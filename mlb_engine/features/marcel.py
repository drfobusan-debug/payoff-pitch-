"""A hitter projection built from free season lines, in the shape the prior reads.

``rolling.ros_rates_from_projection`` wants a rest-of-season projection export --
a per-hitter talent estimate to shrink a thin 42-day window toward instead of the
league mean. The export the engine was written against (THE BAT X, via the
FanGraphs leaderboard) is behind a subscription and, from this machine, behind a
Cloudflare interactive challenge, so the prior has never had a file to read and
has been off since it was built.

This is the same object computed from the official MLB Stats API, which is free,
keyed by MLBAM id, and needs no scraping: a Marcel. Marcel is deliberately the
dumbest forecasting system that is not stupid -- three seasons weighted 5/4/3,
regressed to the league by a fixed number of plate appearances, then aged -- and
it is the standard baseline precisely because more elaborate systems beat it by
small margins. It is not as good as THE BAT X. It is enormously better than the
league mean, which is what every hitter is currently shrunk toward, and it fails
in the right direction: a hitter with no history regresses all the way to league
and lands exactly where today's engine already puts him.

The regression constant is Marcel's own 1200 PA of league-average hitting, and
the aging curve is Marcel's: 0.6% a year gained below 29, 0.3% a year lost above
it, applied to the productive outcomes only. Strikeouts are left unaged, because
the direction of the K aging effect is the opposite of the offensive one and
Marcel's single multiplier does not express that.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

# Weight on the most recent season, the one before it, and the one before that.
MARCEL_WEIGHTS = (5.0, 4.0, 3.0)

# Plate appearances of league-average hitting added to every line before rates
# are taken. This is what makes a 40-PA September call-up a league-average
# hitter rather than a superstar.
REGRESSION_PA = 1200.0

# The projection is published as a 600-PA line: only the ratios are read.
PROJECTED_PA = 600.0

AGE_PEAK = 29.0
AGE_GAIN_PER_YEAR = 0.006  # below the peak
AGE_LOSS_PER_YEAR = 0.003  # above it

# Counting stats carried through. ``SO`` is projected but not aged; ``HBP`` is
# folded into walks downstream, as the simulator has no bucket for it.
COUNTS = ("H", "2B", "3B", "HR", "BB", "SO", "HBP")
AGED = ("H", "2B", "3B", "HR", "BB")

SEASON_COLUMNS = ("mlbam_id", "season", "PA", *COUNTS)


def age_factor(age: float) -> float:
    """Marcel's aging curve, as a multiplier on a projected rate."""
    if age < AGE_PEAK:
        return 1.0 + AGE_GAIN_PER_YEAR * (AGE_PEAK - age)
    return 1.0 / (1.0 + AGE_LOSS_PER_YEAR * (age - AGE_PEAK))


def marcel_projection(
    lines: pd.DataFrame,
    ages: Mapping[int, float],
    season: int,
    min_weighted_pa: float = 100.0,
) -> pd.DataFrame:
    """Project each hitter's rate line, in FanGraphs-export shape.

    ``lines`` is one row per hitter per season with the columns in
    ``SEASON_COLUMNS``; ``season`` is the season being projected, whose own
    partial line carries the heaviest weight.

    Hitters below ``min_weighted_pa`` of weighted history are dropped rather
    than published: their projection would be the league mean to three decimal
    places, and an absent id already falls back to exactly that.
    """
    missing = [c for c in SEASON_COLUMNS if c not in lines.columns]
    if missing:
        raise ValueError(f"season lines are missing {missing}")

    d = lines.copy()
    d["age_back"] = season - d["season"].astype(int)
    d = d[(d["age_back"] >= 0) & (d["age_back"] < len(MARCEL_WEIGHTS))]
    d["w"] = d["age_back"].map(lambda b: MARCEL_WEIGHTS[int(b)])

    for col in ("PA", *COUNTS):
        d[col] = d[col].astype(float) * d["w"]
    agg = d.groupby("mlbam_id")[["PA", *COUNTS]].sum()

    league_pa = float(agg["PA"].sum())
    if league_pa <= 0:
        return pd.DataFrame(columns=["MLBAMID", "PA", *COUNTS])
    league = {c: float(agg[c].sum()) / league_pa for c in COUNTS}

    agg = agg[agg["PA"] >= min_weighted_pa]
    rates = pd.DataFrame(index=agg.index)
    for c in COUNTS:
        rates[c] = (agg[c] + REGRESSION_PA * league[c]) / (agg["PA"] + REGRESSION_PA)

    factor = pd.Series(
        [age_factor(float(ages.get(int(pid), AGE_PEAK))) for pid in rates.index],
        index=rates.index,
    )
    # One multiplier across hits and their subtypes, so H >= 2B + 3B + HR holds
    # and the singles residual downstream stays non-negative.
    for c in AGED:
        rates[c] = rates[c] * factor

    out = pd.DataFrame({"MLBAMID": rates.index.astype(int), "PA": PROJECTED_PA})
    for c in COUNTS:
        out[c] = (rates[c] * PROJECTED_PA).to_numpy()
    return out.reset_index(drop=True)
