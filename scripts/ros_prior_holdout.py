"""Does the projection prior predict the *next* three weeks better than the league?

The prior cannot be graded on the window it was fitted to, so this scores it
forward: build each hitter's rate vector from the 42 days up to a cutoff, then
read the log loss of that vector against every plate appearance he actually took
in the 21 days *after* it. Four vectors are compared on identical PA:

* ``league`` -- the league rate for everyone, the floor a hitter model must beat;
* ``now``    -- the window shrunk 60 equivalent PA toward the league mean, which
  is what the engine ships today;
* ``heavy``  -- the window shrunk toward the *league* at those same per-outcome
  strengths, which separates the new weights from the new target;
* ``prior``  -- the window shrunk toward each hitter's own projection at the
  fitted per-outcome strengths.

Hitters are bootstrapped to put an error bar on the difference, because the PA
of one hitter are not independent draws.

To keep the test honest the projection must not know the holdout. Pass a priors
file built from seasons strictly before the cutoff season: a Marcel that has read
this season's line has read the games it is being scored on.

Usage::

    python -m scripts.ros_prior_holdout \\
        --window ~/.mlb_engine/cache/statcast_2026-03-06_2026-07-22.pkl \\
        --holdout ~/.mlb_engine/cache/statcast_2026-07-03_2026-08-13.pkl \\
        --priors ~/.mlb_engine/ros_hitters_2025.csv --cutoff 2026-07-22
"""

from __future__ import annotations

import argparse
import os
from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    OUTCOME_PRIOR_STRENGTH,
    OUTCOMES_ORDER,
    build_batter_profile,
    load_ros_priors,
    pa_outcome_counts,
)

VECTORS = ("league", "now", "heavy", "prior")

FLOOR = 1e-6


def _loss(vec: dict[str, float], counts: dict[str, float]) -> tuple[float, float]:
    total = sum(counts.values())
    ll = sum(counts[oc] * np.log(max(vec[oc], FLOOR)) for oc in OUTCOMES_ORDER)
    return -float(ll), total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", required=True, help="Statcast cache covering the 42 days before")
    p.add_argument("--holdout", required=True, help="Statcast cache covering the days after")
    p.add_argument("--priors", required=True)
    p.add_argument("--cutoff", required=True, help="last day the model may see")
    p.add_argument("--days", type=int, default=42)
    p.add_argument("--forward", type=int, default=21)
    p.add_argument("--min-pa", type=int, default=60)
    p.add_argument("--min-holdout-pa", type=int, default=20)
    args = p.parse_args()

    cutoff = Date.fromisoformat(args.cutoff)
    priors = load_ros_priors(args.priors)
    win = pd.read_pickle(os.path.expanduser(args.window))
    hold = pd.read_pickle(os.path.expanduser(args.holdout))

    dates = pd.to_datetime(hold["game_date"]).dt.date
    hold = hold[(dates > cutoff) & (dates <= cutoff + timedelta(days=args.forward))]
    print(f"holdout: {cutoff} + {args.forward} days, {len(hold)} pitch rows")

    per_hitter: list[dict[str, float]] = []
    covered = 0
    for pid in sorted({int(i) for i in hold["batter"].dropna().unique()}):
        counts = pa_outcome_counts(hold[hold["batter"] == pid])
        if sum(counts.values()) < args.min_holdout_pa:
            continue
        base = build_batter_profile(win, pid, cutoff, 21, 21, args.days)
        if base.overall.pa < args.min_pa:
            continue
        ros = priors.get(pid)
        with_prior = build_batter_profile(win, pid, cutoff, 21, 21, args.days, ros_prior=ros)
        heavy = build_batter_profile(
            win, pid, cutoff, 21, 21, args.days, ros_prior=dict(LEAGUE_RATES)
        )
        covered += ros is not None
        row = {}
        for name, vec in (
            ("league", dict(LEAGUE_RATES)),
            ("now", base.overall.as_dict()),
            ("heavy", heavy.overall.as_dict()),
            ("prior", with_prior.overall.as_dict()),
        ):
            row[name], row["pa"] = _loss(vec, counts)
        per_hitter.append(row)

    d = pd.DataFrame(per_hitter)
    n_pa = float(d["pa"].sum())
    print(
        f"{len(d)} hitters, {covered} with a projection, {n_pa:.0f} holdout PA "
        f"(heavy = league target at the {len(OUTCOME_PRIOR_STRENGTH)} fitted strengths)\n"
    )
    print(f"  {'vector':<10}{'log loss / PA':>15}{'vs league':>12}{'vs today':>10}{'SE':>9}")
    rng = np.random.default_rng(11)
    draws = rng.integers(0, len(d), size=(2000, len(d)))
    base_ll = float(d["league"].sum()) / n_pa
    now_ll = float(d["now"].sum()) / n_pa
    for name in VECTORS:
        ll = float(d[name].sum()) / n_pa
        gap = (d[name] - d["now"]).to_numpy()
        pa = d["pa"].to_numpy()
        se = float(np.std([gap[i].sum() / pa[i].sum() for i in draws]))
        print(f"  {name:<10}{ll:15.5f}{ll - base_ll:12.5f}{ll - now_ll:10.5f}{se:9.5f}")
    print(
        "\n  Lower is better. 'prior' beating 'now' means shrinking a hitter toward\n"
        "  his own projection predicts his next three weeks better than shrinking\n"
        "  him toward the league, on plate appearances neither vector has seen."
    )


if __name__ == "__main__":
    main()
