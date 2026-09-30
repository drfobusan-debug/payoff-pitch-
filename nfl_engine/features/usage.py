"""Player usage, read only from games that had already been played.

The projections the props layer prices off. The whole file exists to keep one
promise: a projection for ``(season, week)`` is built from weeks *before* it and
from prior seasons, never from the week being priced. A usage model fitted on the
week it is quoting is the leakage the MLB engine spent two rebuilds removing, and
it is invisible in the results -- it just makes everything look sharp.

Names come back from the price feed, not from nflverse, so they are matched on a
normalised form: case, punctuation and generational suffixes dropped. That is
deliberately conservative -- an unmatched player is left unprojected, which the
screens turn into a refusal, rather than matched loosely to somebody else's usage.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.models.player import (
    MIN_GAMES,
    STATS,
    Projection,
    shrunk_mean,
)

log = logging.getLogger(__name__)

REGULAR = "REG"
# Positions whose usage of each stat a book prices. Offensive linemen and
# defenders are dropped: including them makes every rate look reliable, because
# between-position variance is doing the work (see nflverse.rosters).
POSITIONS = {
    "targets": ("WR", "TE", "RB", "FB"),
    "receptions": ("WR", "TE", "RB", "FB"),
    "receiving_yards": ("WR", "TE", "RB", "FB"),
    "carries": ("RB", "QB", "WR", "FB"),
    "rushing_yards": ("RB", "QB", "WR", "FB"),
    "attempts": ("QB",),
    "completions": ("QB",),
    "passing_yards": ("QB",),
}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NON_ALPHA = re.compile(r"[^a-z ]")


def normalise(name: str) -> str:
    """A player name in the form both feeds agree on.

    Periods are deleted rather than spaced out, because a feed writes ``D.K.
    Metcalf`` where the other writes ``DK Metcalf`` and spacing them apart leaves
    two names that never meet.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    plain = _NON_ALPHA.sub(" ", folded.lower().replace(".", ""))
    return " ".join(_SUFFIX.sub(" ", plain).split())


def _weekly(season: int) -> pd.DataFrame:
    frame = nflverse.player_week(season)
    if frame.empty:
        return frame
    if "season_type" in frame.columns:
        frame = frame[frame.season_type == REGULAR]
    keep = ["player_id", "player_display_name", "player_name", "position", "season", "week", "team"]
    columns = [c for c in keep + list(STATS) if c in frame.columns]
    return frame[columns].copy()


def _display(frame: pd.DataFrame) -> pd.Series:
    if "player_display_name" in frame.columns:
        return frame.player_display_name.fillna(frame.get("player_name", ""))
    return frame.player_name


def projections(season: int, week: int) -> dict[tuple[str, str], Projection]:
    """Every projectable ``(normalised name, stat)`` for ``season`` week ``week``.

    Prior weeks of this season carry the mean; the player's own prior season is
    the shrinkage anchor, and his position's league mean stands in when he has no
    prior season. A player with fewer than ``MIN_GAMES`` games this season is
    absent rather than projected off two appearances.
    """
    current = _weekly(season)
    if current.empty:
        return {}
    prior = current[current.week < week]
    if prior.empty:
        return {}
    previous = _weekly(season - 1)
    out: dict[tuple[str, str], Projection] = {}

    for stat in STATS:
        if stat not in prior.columns:
            continue
        rows = prior[prior.position.isin(POSITIONS[stat])]
        rows = rows[rows[stat].notna()]
        if rows.empty:
            continue
        anchor_by_id: dict[str, float] = {}
        position_mean: dict[str, float] = {}
        if not previous.empty and stat in previous.columns:
            past = previous[previous.position.isin(POSITIONS[stat])]
            anchor_by_id = past.groupby("player_id")[stat].mean().to_dict()
            position_mean = past.groupby("position")[stat].mean().to_dict()
        if not position_mean:
            position_mean = rows.groupby("position")[stat].mean().to_dict()

        grouped = rows.assign(display=_display(rows)).groupby("player_id")
        for player_id, part in grouped:
            games = int(len(part))
            if games < MIN_GAMES:
                continue
            position = str(part.position.iloc[-1])
            anchor = anchor_by_id.get(str(player_id))
            fallback = position_mean.get(position, float(rows[stat].mean()))
            mean = shrunk_mean(
                float(part[stat].sum()), games, float(anchor if anchor is not None else fallback)
            )
            name = str(part["display"].iloc[-1])
            out[(normalise(name), stat)] = Projection(
                player=name,
                player_id=str(player_id),
                position=position,
                team=str(part.team.iloc[-1]),
                stat=stat,
                games=games,
                mean=mean,
                prior_mean=anchor,
            )
    return out
