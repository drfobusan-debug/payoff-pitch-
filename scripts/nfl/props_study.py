"""What a player-prop projection is worth, measured out of time.

Fitted on 2016-``--cutoff`` and scored on the seasons after it, on nflverse weekly
lines, through the same runtime functions the props layer prices with
(:mod:`nfl_engine.models.player`, :mod:`nfl_engine.features.usage`) so the numbers
in their docstrings are reproducible rather than remembered.

Two measurements, and they answer different questions:

*Is usage projectable?* MAE of the shrunk projection against the position mean.

*Is there an edge at the numbers a book posts?* The Brier score of P(over) at
pseudo-lines placed where a book would place them, against the base rate of the
same rows, restricted to projections over the market's usage floor. A pseudo-line
drawn on our own projection is the friendliest test available -- there is no archive
of real prop closes anywhere -- so a tie here is a refutation, not a maybe.

    python -m scripts.nfl.props_study --cutoff 2021
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features.usage import POSITIONS, REGULAR
from nfl_engine.models.player import (
    COUNT_STATS,
    MIN_GAMES,
    STATS,
    USAGE_FLOOR,
    Spread,
    prob_over,
    shrunk_mean,
)

FIRST = 2016
# The Brier gain a market has to show over the base rate of the same rows before
# the verdict reads as anything but a tie. Same bar as the calibration layer's
# MIN_GAIN, for the same reason: a 0.001 gain on a few thousand rows is noise.
MIN_GAIN = 0.0020


def weekly(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        frame = nflverse.player_week(season)
        if frame.empty:
            continue
        if "season_type" in frame.columns:
            frame = frame[frame.season_type == REGULAR]
        keep = ["player_id", "position", "season", "week"] + list(STATS)
        frames.append(frame[[c for c in keep if c in frame.columns]])
    return pd.concat(frames, ignore_index=True).sort_values(["player_id", "season", "week"])


def projected(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Every player-week with the projection the engine would have had for it."""
    rows = frame[frame.position.isin(POSITIONS[stat]) & frame[stat].notna()].copy()
    grouped = rows.groupby(["player_id", "season"])[stat]
    rows["prior_sum"] = grouped.cumsum() - rows[stat]
    rows["prior_n"] = grouped.cumcount()
    season_mean = rows.groupby(["player_id", "season"])[stat].mean().rename("prev_mean")
    previous = season_mean.reset_index()
    previous["season"] = previous.season + 1
    rows = rows.merge(previous, how="left", on=["player_id", "season"])
    position_mean = rows.groupby("position")[stat].mean().rename("pos_mean")
    rows = rows.merge(position_mean, how="left", on="position")
    anchor = rows.prev_mean.fillna(rows.pos_mean)
    rows["proj"] = [
        shrunk_mean(s, int(n), float(a))
        for s, n, a in zip(rows.prior_sum, rows.prior_n, anchor, strict=True)
    ]
    return rows[rows.prior_n >= MIN_GAMES]


def spread_fit(rows: pd.DataFrame, stat: str, cutoff: int) -> Spread:
    """Residual sd as a line in the projection, on the training seasons only."""
    fit = rows[rows.season <= cutoff]
    binned = fit.assign(bin=pd.qcut(fit.proj, 8, duplicates="drop")).groupby("bin", observed=True)
    means, sds = [], []
    for _, part in binned:
        means.append(float(part.proj.mean()))
        sds.append(float(np.sqrt(((part[stat] - part.proj) ** 2).mean())))
    slope, intercept = np.polyfit(means, sds, 1)
    return Spread(float(intercept), float(slope))


def pseudo_line(stat: str, projection: float) -> float:
    """Where a book would hang it: counts on the half, yardage on the nearest five."""
    if stat in COUNT_STATS:
        return float(np.floor(projection) + 0.5)
    return float(max(5.0, round(projection / 5.0) * 5.0 + 0.5))


def measure(rows: pd.DataFrame, stat: str, cutoff: int) -> None:
    spread = spread_fit(rows, stat, cutoff)
    hold = rows[(rows.season > cutoff) & (rows.proj >= USAGE_FLOOR[stat])].copy()
    if hold.empty:
        print(f"{stat:18s} no holdout rows over the usage floor")
        return
    lines = [pseudo_line(stat, p) for p in hold.proj]
    probs = [
        prob_over(stat, mean, line, sd=spread.sd(mean)).conditional
        for mean, line in zip(hold.proj, lines, strict=True)
    ]
    hit = (hold[stat].to_numpy() > np.array(lines)).astype(int)
    predicted = np.array(probs)
    base = float(hit.mean())
    brier = float(((predicted - hit) ** 2).mean())
    base_brier = float(((base - hit) ** 2).mean())
    mae = float((hold[stat] - hold.proj).abs().mean())
    naive = float((hold[stat] - hold.groupby("position")[stat].transform("mean")).abs().mean())
    gain = base_brier - brier
    if gain >= MIN_GAIN:
        verdict = "beats the base rate"
    elif gain <= -MIN_GAIN:
        verdict = "worse than the base rate: retired"
    else:
        verdict = "tie at the line"
    print(
        f"{stat:18s} n={len(hold):6d} sd={spread.a:6.2f}{spread.b:+.3f}*m"
        f" MAE={mae:7.2f} (pos {naive:7.2f}) Brier={brier:.4f} base={base_brier:.4f}"
        f" over={base:.3f}  {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=FIRST)
    parser.add_argument("--cutoff", type=int, default=2021, help="last training season")
    parser.add_argument("--last", type=int, default=2025)
    args = parser.parse_args()

    frame = weekly(list(range(args.first, args.last + 1)))
    for stat in STATS:
        measure(projected(frame, stat), stat, args.cutoff)


if __name__ == "__main__":
    main()
