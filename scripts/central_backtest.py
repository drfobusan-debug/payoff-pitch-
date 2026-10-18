"""Do median inputs, or stabilised starter priors, make the forecasts less overconfident?

Four arms replayed on the same slates (offline harness, standard lines, no odds):

    mean      -- production as shipped
    median    -- MLBE_CENTRAL=median: every continuous Statcast read (EV, bat
                 speed, xwOBA/wOBA on contact, velocity, pitches and BF per start)
                 summarised by its median instead of its mean; proportions untouched
    stable    -- MLBE_STARTER_OUTCOME_PRIOR=1 (per-outcome prior strengths on the
                 starter's rates toward his own season line) and
                 MLBE_STARTER_CONTACT_SHRINK=1 (contact multipliers at measured
                 reliability)
    both      -- median + stable

Compared on the picks every arm produced: calibration by probability bucket,
Brier and log loss per market, and the over-confidence gap (avg p - win%) among
picks the model put at >= .55.

    python -m scripts.central_backtest --start 2026-08-01 --end 2026-09-15 --every 3
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import time
from collections import defaultdict
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.backtest import load_season_frame, run_backtest
from mlb_engine.config import load_config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.recommendations import Recommendation

OUT = Path(os.environ.get("MLBE_DATA_DIR", str(Path.home() / ".mlb_engine"))) / "output"

ARMS: dict[str, dict[str, str]] = {
    "mean": {},
    "median": {"MLBE_CENTRAL": "median"},
    "stable": {"MLBE_STARTER_OUTCOME_PRIOR": "1", "MLBE_STARTER_CONTACT_SHRINK": "1.0"},
    "both": {
        "MLBE_CENTRAL": "median",
        "MLBE_STARTER_OUTCOME_PRIOR": "1",
        "MLBE_STARTER_CONTACT_SHRINK": "1.0",
    },
}
ARM_KEYS = ("MLBE_CENTRAL", "MLBE_STARTER_OUTCOME_PRIOR", "MLBE_STARTER_CONTACT_SHRINK")

Pair = tuple[float, int]


def dates_between(start: Date, end: Date, every: int) -> list[Date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=every)
    return out


def brier(pairs: list[Pair]) -> float:
    return sum((p - w) ** 2 for p, w in pairs) / len(pairs) if pairs else float("nan")


def log_loss(pairs: list[Pair]) -> float:
    if not pairs:
        return float("nan")
    tot = 0.0
    for p, w in pairs:
        q = min(max(p, 1e-6), 1 - 1e-6)
        tot -= math.log(q) if w else math.log(1 - q)
    return tot / len(pairs)


def key(rec: Recommendation) -> tuple:
    return (rec.game_pk, rec.market, rec.selection, rec.side, rec.line)


def pairs_by_market(graded: list[tuple[Recommendation, str]], shared: set) -> dict[str, list[Pair]]:
    out: dict[str, list[Pair]] = defaultdict(list)
    for rec, res in graded:
        if res not in (WIN, LOSS) or key(rec) not in shared:
            continue
        out[rec.market].append((float(rec.model_prob), 1 if res == WIN else 0))
    return out


def overconf(pairs: list[Pair], lo: float = 0.55) -> tuple[int, float]:
    top = [(p, w) for p, w in pairs if p >= lo]
    if not top:
        return 0, float("nan")
    return len(top), sum(p for p, _ in top) / len(top) - sum(w for _, w in top) / len(top)


def calib(pairs: list[Pair]) -> list[tuple[float, int, float, float]]:
    b: dict[int, list[Pair]] = defaultdict(list)
    for p, w in pairs:
        b[min(int(p * 10), 9)].append((p, w))
    return [
        (k / 10, len(v), sum(p for p, _ in v) / len(v), sum(w for _, w in v) / len(v))
        for k, v in sorted(b.items())
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-09-15")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--load", default=None, help="reuse a pickle instead of replaying")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    arms = [a for a in args.arms.split(",") if a in ARMS]
    pkl = OUT / "central_backtest.pkl"
    runs: dict[str, list[tuple[Recommendation, str]]]
    if args.load:
        with Path(args.load).open("rb") as f:
            runs = pickle.load(f)
    else:
        start, end = Date.fromisoformat(args.start), Date.fromisoformat(args.end)
        dates = dates_between(start, end, args.every)
        os.environ["MLBE_MC_SIMS"] = str(args.sims)
        cfg = load_config()
        frame = load_season_frame(cfg.cache_dir, Date(start.year, 3, 1), end)
        print(f"{len(dates)} dates {dates[0]}..{dates[-1]}, {len(frame):,} statcast rows")
        client = MLBStatsClient()
        runs = {}
        for arm in arms:
            for k in ARM_KEYS:
                os.environ.pop(k, None)
            os.environ.update(ARMS[arm])
            cfg_a = load_config()
            t = time.time()
            runs[arm] = run_backtest(cfg_a, frame, dates, stats=client)
            print(f"{arm}: {len(runs[arm])} graded picks in {(time.time() - t) / 60:.1f} min")
            with pkl.open("wb") as f:
                pickle.dump(runs, f)

    arms = [a for a in arms if a in runs]
    shared = set.intersection(*({key(r) for r, _ in runs[a]} for a in arms))
    print(f"\nshared graded picks: {len(shared)}")
    by = {a: pairs_by_market(runs[a], shared) for a in arms}
    markets = sorted({m for a in arms for m in by[a]})

    print("\nbrier by market (lower is better)")
    print(f"{'market':<16}{'n':>6}" + "".join(f"{a:>10}" for a in arms))
    for m in markets:
        n = len(by[arms[0]].get(m, []))
        print(f"{m:<16}{n:>6}" + "".join(f"{brier(by[a].get(m, [])):>10.5f}" for a in arms))
    allp = {a: [p for m in markets for p in by[a].get(m, [])] for a in arms}
    print(f"{'ALL':<16}{len(allp[arms[0]]):>6}" + "".join(f"{brier(allp[a]):>10.5f}" for a in arms))

    print("\nlog loss by market")
    print(f"{'market':<16}{'n':>6}" + "".join(f"{a:>10}" for a in arms))
    for m in markets:
        n = len(by[arms[0]].get(m, []))
        print(f"{m:<16}{n:>6}" + "".join(f"{log_loss(by[a].get(m, [])):>10.5f}" for a in arms))
    print(f"{'ALL':<16}{len(allp[arms[0]]):>6}" + "".join(f"{log_loss(allp[a]):>10.5f}" for a in arms))

    print("\nover-confidence among picks at p >= .55: n / (avg p - win%) in points")
    print(f"{'market':<16}" + "".join(f"{a:>16}" for a in arms))
    for m in markets + ["ALL"]:
        row = ""
        for a in arms:
            n, gap = overconf(allp[a] if m == "ALL" else by[a].get(m, []))
            row += f"{n:>6} {gap * 100:>+8.1f}" if n else f"{'--':>16}"
        print(f"{m:<16}{row}")

    print("\ncalibration, all shared picks: bucket n avgP win% (per arm)")
    for a in arms:
        print(f"-- {a}")
        for lo, n, avg, win in calib(allp[a]):
            print(f"   {lo:.1f}  n={n:<6} avgP={avg * 100:5.1f}  win%={win * 100:5.1f}  gap={(avg - win) * 100:+5.1f}")

    print("\npitcher markets, calibration")
    pm = [m for m in markets if m.startswith("pitcher")]
    for a in arms:
        pp = [p for m in pm for p in by[a].get(m, [])]
        print(f"-- {a}  (n={len(pp)} brier={brier(pp):.5f})")
        for lo, n, avg, win in calib(pp):
            print(f"   {lo:.1f}  n={n:<6} avgP={avg * 100:5.1f}  win%={win * 100:5.1f}  gap={(avg - win) * 100:+5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
