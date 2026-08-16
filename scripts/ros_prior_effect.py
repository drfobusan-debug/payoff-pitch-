"""What switching the hitter prior on actually does to the rate vectors.

Runs ``build_batter_profile`` twice over the same window -- once shrinking each
hitter toward the league mean at a flat 60 PA, once toward his own projection at
the fitted per-outcome strengths -- and reports the two things that decide
whether the change is safe: how much wider the lineup becomes, and how far any
single hitter moves.

Usage::

    python -m scripts.ros_prior_effect \\
        --statcast ~/.mlb_engine/cache/statcast_2026-07-03_2026-08-13.pkl \\
        --priors ~/.mlb_engine/ros_hitters.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.features.rolling import (
    OUTCOMES_ORDER,
    build_batter_profile,
    load_ros_priors,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statcast", required=True)
    p.add_argument("--priors", required=True)
    p.add_argument("--as-of")
    p.add_argument("--days", type=int, default=42)
    p.add_argument("--min-pa", type=int, default=60)
    args = p.parse_args()

    priors = load_ros_priors(args.priors)
    df = pd.read_pickle(os.path.expanduser(args.statcast))
    last = df["game_date"].max()
    as_of = (
        Date.fromisoformat(args.as_of)
        if args.as_of
        else (last if isinstance(last, Date) else last.date())
    )

    ids = sorted({int(i) for i in df["batter"].dropna().unique()})
    rows = []
    for pid in ids:
        base = build_batter_profile(df, pid, as_of, 21, 21, args.days)
        if base.overall.pa < args.min_pa:
            continue
        prior = priors.get(pid)
        with_prior = build_batter_profile(df, pid, as_of, 21, 21, args.days, ros_prior=prior)
        row: dict[str, float] = {"mlbam_id": pid, "pa": base.overall.pa, "covered": prior is not None}
        for oc in OUTCOMES_ORDER:
            row[f"{oc}_now"] = base.overall.as_dict()[oc]
            row[f"{oc}_new"] = with_prior.overall.as_dict()[oc]
        rows.append(row)
    d = pd.DataFrame(rows)
    cov = float(d["covered"].mean())
    print(
        f"{len(d)} hitters with >= {args.min_pa} PA in the {args.days} days to {as_of}; "
        f"{cov:.1%} have a projection"
    )

    print("\n  spread across hitters (sd, pp), and how far a hitter moves")
    print(f"  {'outcome':<8}{'now':>8}{'with':>8}{'ratio':>8}{'mean |move|':>13}{'p95 |move|':>12}")
    for oc in OUTCOMES_ORDER:
        now, new = d[f"{oc}_now"], d[f"{oc}_new"]
        move = (new - now).abs()
        ratio = float(new.std() / now.std()) if now.std() else float("nan")
        print(
            f"  {oc:<8}{now.std() * 100:8.2f}{new.std() * 100:8.2f}{ratio:8.2f}"
            f"{move.mean() * 100:13.2f}{np.percentile(move, 95) * 100:12.2f}"
        )

    d["obp_now"] = d[[f"{oc}_now" for oc in ("1B", "2B", "3B", "HR", "BB")]].sum(axis=1)
    d["obp_new"] = d[[f"{oc}_new" for oc in ("1B", "2B", "3B", "HR", "BB")]].sum(axis=1)
    print(
        f"\n  on-base: sd {d['obp_now'].std() * 100:.2f}pp -> {d['obp_new'].std() * 100:.2f}pp, "
        f"correlation {float(np.corrcoef(d['obp_now'], d['obp_new'])[0, 1]):.3f}"
    )
    d["move"] = d["obp_new"] - d["obp_now"]
    cols = ["pa", "obp_now", "obp_new", "move"]
    print("\n  the hitters the league prior was flattering most")
    print(d.sort_values("move")[cols].head(8).to_string(index=False))
    print("\n  ... and the ones it was holding down")
    print(d.sort_values("move", ascending=False)[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
