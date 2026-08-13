"""Is a batter's rest-of-season projection a better prior than the league mean?

Every batter arrives at the simulator as a 42-day window of his own plate
appearances shrunk 60 equivalent PA toward ``LEAGUE_RATES``. That prior is the
same for Aaron Judge and a backup catcher, so it does the one thing a prior must
not do to a *ranking* model: it pulls the extremes toward each other. The
measured consequence is in the ledger -- realised spread between the worst and
best quartile of bats was 9.3 points on ``batter_h`` while the model priced 3.7 --
and it is the likeliest explanation for the engine's discrimination sitting at
AUC 0.54 across every batter market.

A rest-of-season projection (THE BAT X, via the FanGraphs leaderboard export) is
a per-player talent estimate, already regressed and aged. Shrinking toward *his*
line instead of the league's keeps the difference between hitters while still
controlling small-sample noise.

This script does two things and deliberately does not do a third:

1. ``disperse`` -- measures how much the current prior compresses the lineup,
   by comparing the spread across hitters of the engine's own rate vectors with
   the spread of the projection's. Descriptive, no lookahead.
2. ``prior`` -- writes the priors file the engine reads, keyed by MLBAM id.
3. It does NOT claim an out-of-time validation. A ROS file downloaded today
   contains the games any retrospective test would score, so every backtest of
   it is contaminated in its own favour. The honest test is forward: price with
   it, grade it, and read it off the ledger.

Usage::

    python -m scripts.ros_prior_study disperse \\
        --hitters ~/Downloads/fangraphs-leaderboard-projections.csv \\
        --statcast ~/.mlb_engine/cache/statcast_2026-07-02_2026-08-12.pkl

    python -m scripts.ros_prior_study prior \\
        --hitters ~/Downloads/fangraphs-leaderboard-projections.csv \\
        --out ~/.mlb_engine/ros_hitters.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OUTCOMES_ORDER,
    PRIOR_STRENGTH,
    build_batter_profile,
    raw_window_counts,
    ros_rates_from_projection,
)


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.expanduser(path))


def cmd_prior(args: argparse.Namespace) -> None:
    ros = ros_rates_from_projection(_load(args.hitters))
    out = os.path.expanduser(args.out)
    ros.to_csv(out, index=False)
    print(f"{len(ros)} hitters -> {out}")
    print(ros[list(OUTCOMES_ORDER)].describe().loc[["mean", "std", "min", "max"]].to_string())


def cmd_disperse(args: argparse.Namespace) -> None:
    ros = ros_rates_from_projection(_load(args.hitters)).set_index("mlbam_id")
    df = pd.read_pickle(os.path.expanduser(args.statcast))
    if args.as_of:
        as_of = Date.fromisoformat(args.as_of)
    else:
        last = df["game_date"].max()
        as_of = last if isinstance(last, Date) else last.date()

    ids = [i for i in df["batter"].dropna().unique().astype(int) if i in ros.index]
    rows = []
    for pid in ids:
        prof = build_batter_profile(df, pid, as_of, 21, 21, 42)
        if prof.overall.pa < args.min_pa:
            continue
        rows.append({"mlbam_id": pid, **prof.overall.as_dict(), "pa": prof.overall.pa})
    eng = pd.DataFrame(rows).set_index("mlbam_id")
    joined = eng.join(ros, rsuffix="_ros", how="inner")
    print(f"{len(joined)} hitters with >= {args.min_pa} PA in the window, as of {as_of}")

    print("\nspread across hitters (sd), and the correlation between the two")
    print(f"  {'outcome':<8} {'engine':>8} {'ROS':>8} {'ratio':>7} {'r':>7} {'league':>8}")
    for oc in OUTCOMES_ORDER:
        a, b = joined[oc], joined[f"{oc}_ros"]
        r = float(np.corrcoef(a, b)[0, 1])
        ratio = float(a.std() / b.std()) if b.std() else float("nan")
        print(
            f"  {oc:<8} {a.std():8.4f} {b.std():8.4f} {ratio:7.2f} {r:7.2f} "
            f"{LEAGUE_RATES[oc]:8.3f}"
        )

    print("\nthe hitters the current prior flatters most (engine - ROS, on hits)")
    joined["hit_rate"] = joined[["1B", "2B", "3B", "HR"]].sum(axis=1)
    joined["hit_rate_ros"] = joined[["1B_ros", "2B_ros", "3B_ros", "HR_ros"]].sum(axis=1)
    joined["gap"] = joined["hit_rate"] - joined["hit_rate_ros"]
    cols = ["pa", "hit_rate", "hit_rate_ros", "gap"]
    print(joined.sort_values("gap", ascending=False)[cols].head(8).to_string())
    print("\n... and the ones it understates")
    print(joined.sort_values("gap")[cols].head(8).to_string())


def cmd_shrink(args: argparse.Namespace) -> None:
    """How hard should each outcome be shrunk, given how noisy it is?

    For a rate observed over ``n`` PA, the observed variance across hitters is
    talent variance plus binomial noise ``p(1-p)/n``. The prior strength that
    minimises squared error is the classic regression constant

        k = p(1-p) / var(talent)

    in equivalent PA, and it does not depend on ``n``. Two independent estimates
    of talent variance are reported: the projection's spread, and the window's
    own spread with the binomial noise subtracted off. They should agree, and
    where they do not the larger one is used, which shrinks less.
    """
    ros = ros_rates_from_projection(_load(args.hitters)).set_index("mlbam_id")
    df = pd.read_pickle(os.path.expanduser(args.statcast))
    last = df["game_date"].max()
    as_of = Date.fromisoformat(args.as_of) if args.as_of else (
        last if isinstance(last, Date) else last.date()
    )

    ids = [i for i in df["batter"].dropna().unique().astype(int) if i in ros.index]
    raw, pas = [], []
    for pid in ids:
        counts, n = raw_window_counts(df, pid, as_of, args.days)
        if n < args.min_pa:
            continue
        raw.append({oc: counts[oc] / n for oc in OUTCOMES_ORDER})
        pas.append(n)
    obs = pd.DataFrame(raw)
    n_bar = float(np.mean(pas))
    print(f"{len(obs)} hitters, mean {n_bar:.0f} PA in the last {args.days} days as of {as_of}")

    print("\n  outcome    p    talent sd (ROS)  talent sd (window)   k (PA)   current")
    for oc in OUTCOMES_ORDER:
        p = float(obs[oc].mean())
        sd_ros = float(ros[oc].std())
        var_noise = p * (1.0 - p) / n_bar
        var_win = max(float(obs[oc].var()) - var_noise, 1e-12)
        sd_win = float(np.sqrt(var_win))
        var_talent = max(sd_ros**2, var_win)
        k = p * (1.0 - p) / var_talent
        print(
            f"  {oc:<8} {p:6.3f} {sd_ros:16.4f} {sd_win:19.4f} {k:8.0f} "
            f"{PRIOR_STRENGTH:9.0f}"
        )
    print(
        "\n  k far above the current 60 PA means the window is mostly noise in that\n"
        "  bucket and the prior should carry it; k near 60 means the window is\n"
        "  already weighted about right."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prior", help="write the priors file the engine reads")
    pr.add_argument("--hitters", required=True, help="FanGraphs ROS projection export")
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=cmd_prior)

    ds = sub.add_parser("disperse", help="measure how much the league prior compresses")
    ds.add_argument("--hitters", required=True)
    ds.add_argument("--statcast", required=True, help="pickled Statcast cache")
    ds.add_argument("--as-of", help="window end date (default: last date in the cache)")
    ds.add_argument("--min-pa", type=int, default=60)
    ds.set_defaults(func=cmd_disperse)

    sh = sub.add_parser("shrink", help="fit the prior strength each outcome deserves")
    sh.add_argument("--hitters", required=True)
    sh.add_argument("--statcast", required=True)
    sh.add_argument("--as-of")
    sh.add_argument("--days", type=int, default=42)
    sh.add_argument("--min-pa", type=int, default=60)
    sh.set_defaults(func=cmd_shrink)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
