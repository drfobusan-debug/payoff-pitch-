"""Which dropped-in projection should anchor the batter prior: ATC or THE BAT X.

``projection_source`` picks one export out of the folder, so the choice is a
standing decision about how separated the lineup is allowed to be. This measures
the two exports against each other and against what hitters actually did:

* how wide each one spreads the league (sd of projected wOBA, per outcome);
* how much a swap would move one hitter (HR and K per 600 PA);
* the slope of realized rate on projected rate, PA-weighted -- below 1 says the
  export separates hitters further than the season did, which is the shape of
  over-confidence in a prior;
* log loss per PA against the official season lines, floored by giving every
  hitter the league rate.

The contamination is the whole caveat and it cannot be engineered away here: a
rest-of-season export pulled today has read the season it is being scored on, so
every level below is optimistic and a system fitted closer to contemporaneous
batted balls is flattered more. Read the gap between the two files rather than
the distance from the floor, and for a real grade keep a dated copy of the export
and score it against the games that come *after* it (``scripts.ros_prior_holdout``
does that once a forward window exists).

Usage::

    python -m scripts.projection_compare \\
        --atc ~/.mlb_engine/projections/atc_ros_2026-8-18.csv \\
        --batx ~/.mlb_engine/projections/batx_ros_2026-08-18.csv \\
        --season 2026
"""

from __future__ import annotations

import argparse
import os
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.ros_prior import MIN_EXPORT_PA
from mlb_engine.features.rolling import LEAGUE_RATES, OUTCOMES_ORDER, ros_rates_from_projection

FLOOR = 1e-6

# wOBA weights, only ever used to collapse a rate vector to one comparable
# number, so the year's exact scale does not matter.
WOBA = {"1B": 0.883, "2B": 1.244, "3B": 1.569, "HR": 2.004, "BB": 0.692, "K": 0.0, "OUT": 0.0}

RATE_OUTCOMES = ("1B", "2B", "3B", "HR", "BB", "K")


def _rates(path: str) -> pd.DataFrame:
    raw = pd.read_csv(os.path.expanduser(path))
    raw = raw[pd.to_numeric(raw["PA"], errors="coerce") >= MIN_EXPORT_PA]
    return ros_rates_from_projection(raw).set_index("mlbam_id")


def _woba(df: pd.DataFrame) -> pd.Series:
    return sum(df[oc] * WOBA[oc] for oc in OUTCOMES_ORDER)


def _realized(client: MLBStatsClient, season: int, min_pa: int) -> pd.DataFrame:
    """Per-PA outcome counts from the official season lines, hitters with the PA."""
    lines = pd.DataFrame(client.season_hitting(season)).set_index("mlbam_id")
    lines = lines[lines["PA"].astype(float) >= min_pa]
    out = pd.DataFrame(index=lines.index)
    out["2B"] = lines["2B"].astype(float)
    out["3B"] = lines["3B"].astype(float)
    out["HR"] = lines["HR"].astype(float)
    out["1B"] = lines["H"].astype(float) - out["2B"] - out["3B"] - out["HR"]
    out["BB"] = lines["BB"].astype(float) + lines["HBP"].astype(float)
    out["K"] = lines["SO"].astype(float)
    out["PA"] = lines["PA"].astype(float)
    out["OUT"] = (out["PA"] - out[list(RATE_OUTCOMES)].sum(axis=1)).clip(lower=0.0)
    return out


def _loss(vec: pd.Series, counts: pd.Series) -> float:
    return -sum(float(counts[oc]) * np.log(max(float(vec[oc]), FLOOR)) for oc in OUTCOMES_ORDER)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atc", required=True)
    p.add_argument("--batx", required=True)
    p.add_argument("--season", type=int, default=Date.today().year)
    p.add_argument("--min-pa", type=int, default=150)
    args = p.parse_args()

    proj = {"atc": _rates(args.atc), "batx": _rates(args.batx)}
    for key, df in proj.items():
        print(f"{key:5s} {len(df):4d} hitters   sd(projected wOBA) {_woba(df).std():.4f}")

    common = sorted(set(proj["atc"].index) & set(proj["batx"].index))
    a, b = proj["atc"].loc[common], proj["batx"].loc[common]
    print(f"\n{len(common)} hitters in both")
    for oc in RATE_OUTCOMES:
        r = float(np.corrcoef(a[oc], b[oc])[0, 1])
        print(
            f"  {oc:3s} sd atc {a[oc].std():.5f}  sd batx {b[oc].std():.5f}"
            f"  ratio {b[oc].std() / a[oc].std():5.2f}  r {r:.3f}"
        )
    for oc in ("HR", "K"):
        d = (b[oc] - a[oc]) * 600
        print(
            f"  {oc} per 600 PA, batx minus atc: mean {d.mean():+.2f}"
            f"  mean abs {d.abs().mean():.2f}  max {d.abs().max():.2f}"
        )

    real = _realized(MLBStatsClient(), args.season, args.min_pa)
    idx = [i for i in real.index if i in a.index and i in b.index]
    print(f"\n{len(idx)} of them took {args.min_pa}+ PA in {args.season}")

    print("\nrealized rate on projected rate, PA-weighted (slope < 1 = separates too far)")
    print(f"{'outcome':8s} {'slope atc':>10s} {'slope batx':>11s} {'r atc':>8s} {'r batx':>8s}")
    weights = np.sqrt(real.loc[idx, "PA"].astype(float).to_numpy())
    for oc in (*RATE_OUTCOMES, "wOBA"):
        fits: list[float] = []
        for key in ("atc", "batx"):
            df = proj[key].loc[idx]
            x = (_woba(df) if oc == "wOBA" else df[oc].astype(float)).to_numpy()
            y = (
                _woba(real.loc[idx, list(OUTCOMES_ORDER)].div(real.loc[idx, "PA"], axis=0))
                if oc == "wOBA"
                else (real.loc[idx, oc] / real.loc[idx, "PA"]).astype(float)
            ).to_numpy()
            fits.append(float(np.polyfit(x, y, 1, w=weights)[0]))
            fits.append(float(np.corrcoef(x, y)[0, 1]))
        print(f"{oc:8s} {fits[0]:10.3f} {fits[2]:11.3f} {fits[1]:8.3f} {fits[3]:8.3f}")

    total_pa = float(real.loc[idx, list(OUTCOMES_ORDER)].sum(axis=1).sum())
    league = sum(_loss(pd.Series(LEAGUE_RATES), real.loc[i]) for i in idx) / total_pa
    print(f"\nlog loss per PA over {total_pa:.0f} PA (in sample -- see the docstring)")
    print(f"  league rate for everyone   {league:.5f}")
    for key in ("atc", "batx"):
        got = sum(_loss(proj[key].loc[i], real.loc[i]) for i in idx) / total_pa
        print(f"  {key:5s}                      {got:.5f}   {got - league:+.5f} vs league")


if __name__ == "__main__":
    main()
