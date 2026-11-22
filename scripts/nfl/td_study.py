"""What the anytime-touchdown projection is worth, measured out of time.

Same runtime pieces the props layer prices with (``usage.add_touchdowns``,
``player.shrunk_mean``, ``player.anytime_prob``): the in-season touchdown mean after
``MIN_GAMES`` games, shrunk toward the player's prior season (or his position's),
read as Poisson. Scored as the Brier of P(at least one) against the position base
rate fitted on the training seasons. No book's price is in this table, because no
archive of anytime-scorer prices exists before the forward capture.

    python -m scripts.nfl.td_study --cutoff 2021
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features.usage import POSITIONS, REGULAR, add_touchdowns
from nfl_engine.models.player import MIN_GAMES, SHRINK, TOUCHDOWNS, anytime_prob

FIRST, LAST = 2016, 2025


def weekly(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        frame = add_touchdowns(nflverse.player_week(season))
        if "season_type" in frame.columns:
            frame = frame[frame.season_type == REGULAR]
        frame = frame[frame.position.isin(POSITIONS[TOUCHDOWNS])]
        frames.append(frame[["player_id", "position", "season", "week", TOUCHDOWNS]])
    return pd.concat(frames).sort_values(["player_id", "season", "week"])


def projected(frame: pd.DataFrame) -> pd.DataFrame:
    """Each player-week after ``MIN_GAMES`` games, with the shrunk mean before it."""
    by_season = frame.groupby(["player_id", "season"])[TOUCHDOWNS].mean()
    anchor = by_season.rename("anchor").reset_index().assign(season=lambda f: f.season + 1)
    position = frame.groupby(["position", "season"])[TOUCHDOWNS].mean()
    pos = position.rename("pos_mean").reset_index().assign(season=lambda f: f.season + 1)
    rows = frame.merge(anchor, on=["player_id", "season"], how="left")
    rows = rows.merge(pos, on=["position", "season"], how="left")
    group = rows.groupby(["player_id", "season"])
    rows["games"] = group.cumcount()
    rows["prior_sum"] = group[TOUCHDOWNS].cumsum() - rows[TOUCHDOWNS]
    rows = rows[rows.games >= MIN_GAMES].copy()
    base = rows.anchor.fillna(rows.pos_mean)
    rows["mean"] = (rows.prior_sum + SHRINK * base) / (rows.games + SHRINK)
    rows["p"] = rows["mean"].map(anytime_prob)
    rows["hit"] = (rows[TOUCHDOWNS] >= 1).astype(float)
    return rows.dropna(subset=["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=2021)
    args = parser.parse_args()
    rows = projected(weekly(list(range(FIRST - 1, LAST + 1))))
    rows = rows[rows.season >= FIRST]
    train, test = rows[rows.season <= args.cutoff], rows[rows.season > args.cutoff]
    for floor in (0.0, 0.15, 0.25):
        tr, te = train[train["mean"] >= floor], test[test["mean"] >= floor]
        base = te.position.map(tr.groupby("position").hit.mean())
        print(
            f"mean >= {floor:.2f}: n {len(te)}  hit {te.hit.mean():.3f}"
            f"  brier model {np.mean((te.p - te.hit) ** 2):.4f}"
            f"  position base {np.mean((base - te.hit) ** 2):.4f}"
        )


if __name__ == "__main__":
    main()
