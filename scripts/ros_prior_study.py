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
    bullpen_relief_frame,
    pa_outcome_counts,
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


def cmd_pen(args: argparse.Namespace) -> None:
    """The same estimator for a bullpen's allowed rates.

    A pen's three-week aggregate is the thinnest sample the engine trusts, so the
    question is whether a projection should carry it. Talent variance is measured
    across the 30 pens two ways: the spread of usage-weighted rest-of-season
    projections for each team's relievers, and the spread of the windows
    themselves with binomial noise subtracted.

    The pitcher projection publishes H and HR but not 2B or 3B, so hits are
    treated as one bucket here -- the hit-type shape is not something this feed
    can prior.
    """
    proj = _load(args.pitchers)
    proj = proj[proj["TBF"] > 0].copy()
    starter_share = proj["GS"] / proj["G"].replace(0, np.nan)
    pen = proj[starter_share.fillna(0.0) < args.max_gs_share]
    pen_rates = []
    for team, g in pen.groupby("Team"):
        w = g["TBF"].astype(float)
        tbf = float(w.sum())
        if tbf < args.min_tbf:
            continue
        pen_rates.append(
            {
                "team": team,
                "tbf": tbf,
                "H": float((g["H"] - g["HR"]).sum()) / tbf,  # hits that are not HR
                "HR": float(g["HR"].sum()) / tbf,
                "BB": float((g["BB"] + g.get("HBP", 0.0)).sum()) / tbf,
                "K": float(g["SO"].sum()) / tbf,
            }
        )
    ros = pd.DataFrame(pen_rates)
    print(f"{len(ros)} pens from the projection ({len(pen)} relief arms)")

    df = pd.read_pickle(os.path.expanduser(args.statcast))
    last = df["game_date"].max()
    as_of = Date.fromisoformat(args.as_of) if args.as_of else (
        last if isinstance(last, Date) else last.date()
    )
    obs, pas = [], []
    for team in sorted(set(df["home_team"].dropna()) | set(df["away_team"].dropna())):
        relief = bullpen_relief_frame(df, str(team), as_of, args.days)
        if relief.empty:
            continue
        counts = pa_outcome_counts(relief)
        n = sum(counts.values())
        if n < args.min_pa:
            continue
        row = {oc: counts[oc] / n for oc in OUTCOMES_ORDER}
        row["H"] = (counts["1B"] + counts["2B"] + counts["3B"]) / n
        obs.append(row)
        pas.append(n)
    win = pd.DataFrame(obs)
    n_bar = float(np.mean(pas))
    print(f"{len(win)} pens with >= {args.min_pa} relief PA in {args.days} days, mean {n_bar:.0f}")

    print("\n  bucket    p     talent sd (ROS)  talent sd (window)    k (PA)   current")
    for oc in ("H", *OUTCOMES_ORDER):
        p = float(win[oc].mean())
        # The pitcher projection has no 2B/3B, so those buckets have only the
        # window's own spread to go on.
        sd_ros = float(ros[oc].std()) if oc in ros else float("nan")
        var_noise = p * (1.0 - p) / n_bar
        var_win = max(float(win[oc].var()) - var_noise, 1e-12)
        var_talent = var_win if np.isnan(sd_ros) else max(sd_ros**2, var_win)
        k = p * (1.0 - p) / var_talent
        shown = "     n/a" if np.isnan(sd_ros) else f"{sd_ros:8.4f}"
        print(
            f"  {oc:<8} {p:6.3f} {shown:>16} {np.sqrt(var_win):19.4f} {k:9.0f} "
            f"{PRIOR_STRENGTH:9.0f}"
        )

    print("\n  what the current weight is actually made of, at the mean window (pp)")
    print(f"  {'bucket':<8}{'w now':>7}{'w fit':>7}{'noise now':>11}{'noise fit':>11}{'talent':>9}")
    for oc in ("H", *OUTCOMES_ORDER):
        p = float(win[oc].mean())
        var_noise = p * (1.0 - p) / n_bar
        var_win = max(float(win[oc].var()) - var_noise, 1e-12)
        sd_ros = float(ros[oc].std()) if oc in ros else float("nan")
        var_talent = var_win if np.isnan(sd_ros) else max(sd_ros**2, var_win)
        k = p * (1.0 - p) / var_talent
        noise_sd = float(np.sqrt(var_noise))
        w_now, w_fit = n_bar / (n_bar + PRIOR_STRENGTH), n_bar / (n_bar + k)
        print(
            f"  {oc:<8}{w_now:7.2f}{w_fit:7.2f}{noise_sd * w_now * 100:11.2f}"
            f"{noise_sd * w_fit * 100:11.2f}{np.sqrt(var_talent) * 100:9.2f}"
        )
    print(
        "\n  Noise now above the talent column means the pen vector the simulator\n"
        "  receives is mostly sampling error rather than pen. Where the two talent\n"
        "  estimates are close, shrinking toward the league pen loses little, so the\n"
        "  gain is in the weight rather than in the projection as a target."
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

    pen = sub.add_parser("pen", help="the same estimator for bullpen allowed rates")
    pen.add_argument("--pitchers", required=True, help="FanGraphs ROS pitcher export")
    pen.add_argument("--statcast", required=True)
    pen.add_argument("--as-of")
    pen.add_argument("--days", type=int, default=21)
    pen.add_argument("--min-pa", type=int, default=100)
    pen.add_argument("--min-tbf", type=float, default=200.0)
    pen.add_argument(
        "--max-gs-share",
        type=float,
        default=0.5,
        help="an arm counts as relief below this share of its games started",
    )
    pen.set_defaults(func=cmd_pen)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
