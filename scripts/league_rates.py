"""Measure the league's PA outcome rates, which are a prior and a log5 denominator.

``LEAGUE_RATES`` does three jobs, and being wrong costs something different in each:

* it is the Bayesian prior a thin batter window is shrunk toward, at 60 equivalent
  PA, so a stale value drags every low-PA hitter toward a league that never existed;
* it is the **denominator of the log5 matchup**, where ``b * p / lg`` means a value
  that is too small *inflates* that outcome in every matchup in the slate;
* it is the fallback when a pitcher has no sample at all.

So it has to be measured with the same bucketer that produces the observations it is
combined with, which is what this script does -- ``_bucket_counts`` rather than a
reimplementation, on the Statcast frame the engine itself caches.

Exhibition games have to be excluded by date: the engine's cached column set drops
Statcast's ``game_type``, and the widest cache on this box reaches back to March 6.
Leaving spring training in moves walks by 0.06pp, which is small but is not nothing
and is not the league.

    python -m scripts.league_rates
"""

from __future__ import annotations

import argparse
import re
from datetime import date as Date
from pathlib import Path

import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OUTCOMES_ORDER,
    _bucket_counts,
    _pa_rows,
)

CACHE_RE = re.compile(r"statcast_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.pkl")

# Opening day 2026. Identifiable in the cache as the discontinuity it is: eight to
# seventeen exhibition games a day through 2026-03-24, then a single 74-PA game.
SEASON_START = Date(2026, 3, 25)


def widest_cache(cache_dir: Path) -> Path | None:
    """The cached Statcast frame covering the most days."""
    best: tuple[int, Path] | None = None
    for path in cache_dir.glob("statcast_*.pkl"):
        m = CACHE_RE.fullmatch(path.name)
        if m is None:
            continue
        span = (Date.fromisoformat(m.group(2)) - Date.fromisoformat(m.group(1))).days
        if best is None or span > best[0]:
            best = (span, path)
    return best[1] if best else None


def rates(pa_events: pd.Series) -> dict[str, float]:
    counts = _bucket_counts(pa_events)
    n = sum(counts.values())
    return {k: counts[k] / n for k in OUTCOMES_ORDER} if n else dict.fromkeys(OUTCOMES_ORDER, 0.0)


def report(pa: pd.DataFrame) -> str:
    L: list[str] = []
    measured = rates(pa["events"])
    L.append(f"{len(pa):,} plate appearances, {pa.game_date.min()} .. {pa.game_date.max()}\n")
    L.append(f"{'':6}{'shipped':>10}{'measured':>10}{'error':>9}{'relative':>10}")
    for k in OUTCOMES_ORDER:
        err = LEAGUE_RATES[k] - measured[k]
        rel = err / measured[k] if measured[k] else 0.0
        L.append(f"{k:6}{LEAGUE_RATES[k]:10.4f}{measured[k]:10.4f}{err:+9.4f}{rel:+9.1%}")
    L.append(f"\nmeasured sums to {sum(measured.values()):.6f}")

    # The seasonal drift, which is the argument against fitting this to a window.
    # A summer window is warmer on home runs and *colder* on walks, so pinning the
    # prior to one biases two markets in opposite directions.
    by_month = pa.assign(month=pd.to_datetime(pa.game_date).dt.strftime("%Y-%m"))
    L.append("\nBy month -- why this is measured season-to-date and not on a window")
    L.append(f"{'month':>9}{'PA':>9}" + "".join(f"{k:>8}" for k in OUTCOMES_ORDER))
    for month, g in by_month.groupby("month"):
        m = rates(g["events"])
        L.append(f"{month:>9}{len(g):9,}" + "".join(f"{m[k]:8.4f}" for k in OUTCOMES_ORDER))

    L.append("\nLEAGUE_RATES = {")
    for k in OUTCOMES_ORDER:
        L.append(f'    "{k}": {measured[k]:.4f},')
    L.append("}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=None, help="path to a cached Statcast pickle")
    ap.add_argument(
        "--start",
        default=SEASON_START.isoformat(),
        help="drop games before this date, to keep spring training out",
    )
    args = ap.parse_args()

    path = Path(args.frame) if args.frame else widest_cache(load_config().cache_dir)
    if path is None or not path.exists():
        raise SystemExit("no cached Statcast frame found; pass --frame")
    print(f"reading {path.name}")
    pa = _pa_rows(pd.read_pickle(path))
    start = Date.fromisoformat(args.start)
    before = len(pa)
    pa = pa[pd.to_datetime(pa.game_date).dt.date >= start]
    if before > len(pa):
        print(f"dropped {before - len(pa):,} plate appearances before {start} (exhibition)")
    print(report(pa))


if __name__ == "__main__":
    main()
