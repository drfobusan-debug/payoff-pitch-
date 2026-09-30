"""How long a look-back does each batter outcome actually need?

The engine used to read a hitter's baseline over whatever the longest split
window happened to be, on the unexamined assumption that recent is better. This
grades that forward instead. For every cutoff: read each hitter's rates over the
W days *before* it, then score that read against the same hitters' next 21 days.

Three questions, three modes:

* ``sweep``  -- out-of-time correlation and holdout RMSE per window, per
  outcome, at the shipped per-outcome prior strengths and multiples of them, so
  the window is separated from the shrinkage. The prior is the league mean,
  which is knowable at the cutoff -- shrinking toward a rest-of-season
  projection dated after the holdout would be reading the future.
* ``joint``  -- the next 21 days regressed on the short *and* the long read
  together. A short-window coefficient near zero means recent form carries
  nothing the season read does not already have.
* ``platoon`` -- the same pair, but the holdout is only the PA against one hand,
  which is what the platoon split is for.

Usage::

    python -m scripts.batter_window_study sweep \\
        --frame ~/.mlb_engine/cache/statcast_2026-03-24_2026-08-20.pkl
    python -m scripts.batter_window_study joint --frame ... --long 90 --short 21
    python -m scripts.batter_window_study platoon --frame ... --hand R
"""

from __future__ import annotations

import argparse
from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OUTCOME_PRIOR_STRENGTH,
    _bucket_counts,
    _pa_rows,
)

WINDOWS = (21, 28, 42, 60, 90, 180)
SCALES = (0.5, 1.0, 2.0, 4.0)
OUTCOMES = ("K", "BB", "1B", "2B", "HR", "OUT")


def _load(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df["d"] = pd.to_datetime(df["game_date"]).dt.date
    cols = ["d", "batter", "events", "inning_topbot", "p_throws"]
    return _pa_rows(df)[cols]


def _cutoffs(pa: pd.DataFrame, holdout: int, n: int, need: int) -> list[Date]:
    last = pa["d"].max()
    first = pa["d"].min()
    out = [last - timedelta(days=holdout * k) for k in range(1, n + 1)]
    return [c for c in out if (c - first).days >= need][::-1]


def _rates(events: pd.Series, floor: int) -> dict[str, float] | None:
    counts = _bucket_counts(events)
    n = sum(counts.values())
    if n < floor:
        return None
    return {k: counts[k] / n for k in counts} | {"pa": float(n)}


def _read(pa: pd.DataFrame, cut: Date, days: int, floor: int,
          hand: str | None = None) -> dict[int, dict[str, float]]:
    sl = pa[(pa["d"] < cut) & (pa["d"] >= cut - timedelta(days=days))]
    if hand:
        sl = sl[sl["p_throws"] == hand]
    out = {}
    for bid, g in sl.groupby("batter"):
        r = _rates(g["events"], floor)
        if r:
            out[int(bid)] = r
    return out


def _holdout(pa: pd.DataFrame, cut: Date, days: int, floor: int,
             hand: str | None = None) -> dict[int, dict[str, float]]:
    sl = pa[(pa["d"] >= cut) & (pa["d"] < cut + timedelta(days=days))]
    if hand:
        sl = sl[sl["p_throws"] == hand]
    out = {}
    for bid, g in sl.groupby("batter"):
        r = _rates(g["events"], floor)
        if r:
            out[int(bid)] = r
    return out


def _fit(x: list[list[float]], y: list[float]) -> np.ndarray:
    a = np.column_stack([np.ones(len(x)), np.array(x)])
    beta, *_ = np.linalg.lstsq(a, np.array(y), rcond=None)
    return beta


def sweep(pa: pd.DataFrame, args: argparse.Namespace) -> None:
    cuts = _cutoffs(pa, args.holdout, args.cutoffs, max(WINDOWS[:-1]))
    print(f"cutoffs: {', '.join(str(c) for c in cuts)}")
    hold = {c: _holdout(pa, c, args.holdout, args.min_pa) for c in cuts}
    reads = {(c, w): _read(pa, c, w, args.read_pa) for c in cuts for w in WINDOWS}

    print("\nout-of-time correlation with the next "
          f"{args.holdout} days, and the read's mean PA")
    print(f"{'window':>7}{'mean PA':>9}" + "".join(f"{k:>8}" for k in OUTCOMES))
    for w in WINDOWS:
        xs: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k in OUTCOMES}
        pas: list[float] = []
        for c in cuts:
            r = reads[(c, w)]
            for bid, h in hold[c].items():
                if bid not in r:
                    continue
                pas.append(r[bid]["pa"])
                for k in OUTCOMES:
                    xs[k][0].append(r[bid][k])
                    xs[k][1].append(h[k])
        line = f"{w:5d}d {np.mean(pas):8.0f}"
        for k in OUTCOMES:
            line += f"{np.corrcoef(*xs[k])[0, 1]:8.3f}"
        print(line)

    print("\nholdout RMSE x1000, league prior at the shipped k x scale")
    for k in OUTCOMES:
        print(f"\n{k}   (k = {OUTCOME_PRIOR_STRENGTH[k]:.0f} PA)")
        print(f"{'window':>7}" + "".join(f"{'x' + str(s):>9}" for s in SCALES))
        for w in WINDOWS:
            line = f"{w:5d}d "
            for s in SCALES:
                kk = OUTCOME_PRIOR_STRENGTH[k] * s
                errs = []
                for c in cuts:
                    r = reads[(c, w)]
                    for bid, h in hold[c].items():
                        if bid not in r:
                            continue
                        n = r[bid]["pa"]
                        est = (r[bid][k] * n + kk * LEAGUE_RATES[k]) / (n + kk)
                        errs.append((est - h[k]) ** 2)
                line += f"{np.sqrt(np.mean(errs)) * 1000:9.1f}"
            print(line)


def joint(pa: pd.DataFrame, args: argparse.Namespace, hand: str | None = None) -> None:
    cuts = _cutoffs(pa, args.holdout, args.cutoffs, args.long)
    label = f" vs {hand}HP" if hand else ""
    print(f"cutoffs: {', '.join(str(c) for c in cuts)}")
    print(f"\nnext {args.holdout} days{label} ~ {args.long}d read + {args.short}d"
          f"{label} read")
    print(f"{'outcome':>8}{f'b({args.long}d)':>12}{f'c({args.short}d)':>12}{'n':>7}")
    hold = {c: _holdout(pa, c, args.holdout, args.min_pa, hand) for c in cuts}
    long_r = {c: _read(pa, c, args.long, args.read_pa) for c in cuts}
    short_r = {c: _read(pa, c, args.short, args.read_pa, hand) for c in cuts}
    for k in OUTCOMES:
        x, y = [], []
        for c in cuts:
            for bid, h in hold[c].items():
                if bid in long_r[c] and bid in short_r[c]:
                    x.append([long_r[c][bid][k], short_r[c][bid][k]])
                    y.append(h[k])
        beta = _fit(x, y)
        print(f"{k:>8}{beta[1]:12.2f}{beta[2]:12.2f}{len(y):7d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["sweep", "joint", "platoon"])
    p.add_argument("--frame", required=True, help="cached Statcast pickle")
    p.add_argument("--holdout", type=int, default=21, help="days scored after the cutoff")
    p.add_argument("--cutoffs", type=int, default=2, help="how many disjoint holdouts")
    p.add_argument("--min-pa", type=int, default=30, help="holdout PA floor")
    p.add_argument("--read-pa", type=int, default=15, help="window PA floor")
    p.add_argument("--long", type=int, default=90)
    p.add_argument("--short", type=int, default=21)
    p.add_argument("--hand", default="R", choices=["R", "L"])
    args = p.parse_args()

    pa = _load(args.frame)
    print(f"{len(pa):,} PA rows, {pa['d'].min()} .. {pa['d'].max()}")
    if args.mode == "sweep":
        sweep(pa, args)
    elif args.mode == "joint":
        joint(pa, args)
    else:
        joint(pa, args, hand=args.hand)


if __name__ == "__main__":
    main()
