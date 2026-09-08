"""The game around the player: share of team volume, the defence, the line.

:mod:`nfl_engine.features.usage` projects a player from his own past games and
nothing else, and :mod:`scripts.nfl.props_study` showed that such a projection
has no edge at the line because the book knows his usage too. This module adds
the three things a book sees that the usage model does not, measured out of time
in :mod:`scripts.nfl.props_context_study`:

* **share** -- the player's shrunk share of his team's volume times the team's
  projected volume; a share is stabler than a count.
* **opponent** -- what the defence has allowed per game to his position group,
  as a shrunk ratio to the league.
* **script** -- the spread from his team's side and the total, from nflverse
  ``games``, which carries a line for the coming week as well as the closes.

Each is an interaction with the usage projection, so the correction scales with
the role, and the coefficients are fitted once on 2016-2021 and frozen here so a
projection built under this basis is the same projection every week. Every rule
of the usage module holds: a week is projected from weeks before it only.

The basis stamp is different from the usage basis on purpose. A research row
priced off this projection is evidence for this projection, not for the one the
props layer has shipped since the first archive, and the two must never be summed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.features import usage
from nfl_engine.models.player import (
    ATTEMPTS,
    CARRIES,
    COMPLETIONS,
    PASSING_YARDS,
    RECEIVING_YARDS,
    RECEPTIONS,
    RUSHING_YARDS,
    STATS,
    TARGETS,
    Projection,
    shrunk_mean,
)

log = logging.getLogger(__name__)

BASIS = "context-share-opponent-script-2016-2021"

# The team volume a stat is a share of.
TEAM_ATTEMPTS, TEAM_CARRIES, TEAM_PLAYS = "team_attempts", "team_carries", "team_plays"
VOLUME_OF = {
    TARGETS: TEAM_ATTEMPTS,
    RECEPTIONS: TEAM_ATTEMPTS,
    RECEIVING_YARDS: TEAM_ATTEMPTS,
    CARRIES: TEAM_CARRIES,
    RUSHING_YARDS: TEAM_CARRIES,
    ATTEMPTS: TEAM_PLAYS,
    COMPLETIONS: TEAM_ATTEMPTS,
    PASSING_YARDS: TEAM_ATTEMPTS,
}
# Pseudo-games the opponent's allowance is shrunk toward the league by. A defence
# is a smaller sample than a player: seventeen games a season, one opponent each.
OPP_SHRINK = 4.0
# The script terms are centred so a pick'em at the league total is no correction.
SPREAD_SCALE = 7.0
TOTAL_CENTRE = 45.0
TOTAL_SCALE = 10.0


@dataclass(frozen=True)
class Terms:
    """One stat's fitted correction to the usage projection ``u``.

    mean = intercept + usage*u + share*share_volume + opponent*u*(opp_factor - 1)
           + spread*u*team_spread/7 + total*u*(total_line - 45)/10
    """

    intercept: float
    usage: float
    share: float
    opponent: float
    spread: float
    total: float


def features(
    base: np.ndarray,
    share_volume: np.ndarray,
    opp_factor: np.ndarray,
    team_spread: np.ndarray,
    total_line: np.ndarray,
) -> np.ndarray:
    """The design matrix, one column per :class:`Terms` field, in field order."""
    return np.column_stack(
        [
            np.ones(len(base)),
            base,
            share_volume,
            base * (opp_factor - 1.0),
            base * team_spread / SPREAD_SCALE,
            base * (total_line - TOTAL_CENTRE) / TOTAL_SCALE,
        ]
    )


def adjusted_mean(
    terms: Terms,
    *,
    base: float,
    share_volume: float,
    opp_factor: float,
    team_spread: float,
    total_line: float,
) -> float:
    row = features(
        np.array([base]),
        np.array([share_volume]),
        np.array([opp_factor]),
        np.array([team_spread]),
        np.array([total_line]),
    )[0]
    beta = np.array(
        [terms.intercept, terms.usage, terms.share, terms.opponent, terms.spread, terms.total]
    )
    return float(max(0.0, row @ beta))


# Fitted by ``python -m scripts.nfl.props_context_study --cutoff 2021 --coefficients``
# on regular-season 2016-2021, ridge 1e-3, players with four or more prior games.
# Held out on 2022-2025 at the usage projection's pseudo-line, Brier against the
# usage projection: attempts +0.0102, completions +0.0090, passing yards +0.0081,
# rushing yards +0.0020, receptions +0.0018, targets +0.0013, carries +0.0012.
# Receiving yards measured a loss (-0.0034) and has no terms: it keeps the usage
# projection under this basis. That is probability quality at a line of our own,
# not edge against a book's, which is what the forward archive grades.
TERMS: dict[str, Terms] = {
    TARGETS: Terms(0.3124, 0.5268, 0.4009, 0.2247, 0.0003, 0.0277),
    RECEPTIONS: Terms(0.2679, 0.7081, 0.1820, 0.3112, 0.0126, 0.0580),
    CARRIES: Terms(0.0812, 1.1463, -0.1406, 0.6225, -0.0110, -0.0329),
    ATTEMPTS: Terms(8.6227, 0.3440, 0.3872, 0.1910, -0.0061, 0.0386),
    COMPLETIONS: Terms(5.0787, 0.3489, 0.3938, 0.2121, 0.0080, 0.0514),
    RUSHING_YARDS: Terms(0.4962, 1.0038, -0.0133, 0.6232, 0.0155, -0.0091),
    PASSING_YARDS: Terms(60.3075, 0.3300, 0.3824, 0.2102, 0.0150, 0.0721),
}


def _weekly(season: int) -> pd.DataFrame:
    frame = nflverse.player_week(season)
    if frame.empty:
        return frame
    if "season_type" in frame.columns:
        frame = frame[frame.season_type == usage.REGULAR]
    keep = ["player_id", "position", "season", "week", "team", "opponent_team"]
    columns = [c for c in keep + list(STATS) if c in frame.columns]
    return frame[columns].copy()


def team_volume(rows: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's pass attempts, carries and plays, from the box score."""
    vol = rows.groupby(["season", "week", "team"], as_index=False).agg(
        team_attempts=("attempts", "sum"), team_carries=("carries", "sum")
    )
    vol[TEAM_PLAYS] = vol.team_attempts + vol.team_carries
    return vol


def _volume_projection(prior: pd.DataFrame, previous: pd.DataFrame) -> dict[tuple[str, str], float]:
    """``(team, volume) -> projected volume`` from the season's earlier games."""
    vol = team_volume(prior)
    past = team_volume(previous) if not previous.empty else pd.DataFrame()
    out: dict[tuple[str, str], float] = {}
    for col in (TEAM_ATTEMPTS, TEAM_CARRIES, TEAM_PLAYS):
        league = float(past[col].mean()) if not past.empty else float(vol[col].mean())
        anchors = past.groupby("team")[col].mean().to_dict() if not past.empty else {}
        for team, part in vol.groupby("team"):
            anchor = anchors.get(str(team), league)
            out[(str(team), col)] = shrunk_mean(float(part[col].sum()), len(part), anchor)
    return out


def _opponent_factors(prior: pd.DataFrame, stat: str) -> dict[str, float]:
    """``defence -> shrunk ratio`` of what it allowed per game to the stat's positions."""
    pos = prior[prior.position.isin(usage.POSITIONS[stat]) & prior[stat].notna()]
    if pos.empty or "opponent_team" not in pos.columns:
        return {}
    allowed = pos.groupby(["week", "opponent_team"], as_index=False)[stat].sum()
    league = float(allowed[stat].mean())
    if league <= 0:
        return {}
    allowed["ratio"] = allowed[stat] / league
    return {
        str(defence): shrunk_mean(float(part.ratio.sum()), len(part), 1.0, shrink=OPP_SHRINK)
        for defence, part in allowed.groupby("opponent_team")
    }


@dataclass(frozen=True)
class GameContext:
    opponent: str
    team_spread: float  # positive when the player's team is favoured
    total_line: float


def week_context(season: int, week: int) -> dict[str, GameContext]:
    """``team -> opponent, spread, total`` for the week, from nflverse ``games``."""
    games = nflverse.games()
    if games.empty:
        return {}
    rows = games[(games.season == season) & (games.week == week)]
    out: dict[str, GameContext] = {}
    for row in rows.itertuples(index=False):
        spread = float(row.spread_line) if pd.notna(row.spread_line) else 0.0
        total = float(row.total_line) if pd.notna(row.total_line) else TOTAL_CENTRE
        home, away = str(row.home_team), str(row.away_team)
        out[home] = GameContext(away, spread, total)
        out[away] = GameContext(home, -spread, total)
    return out


def projections(season: int, week: int) -> dict[tuple[str, str], Projection]:
    """The usage projections, corrected for share, opponent and script.

    Built on :func:`usage.projections` so the two bases agree on who is
    projectable; a player whose share or game cannot be found keeps his usage
    projection under this basis rather than being dropped, and the fallback is
    counted in the log so a week priced mostly on fallbacks is visible.
    """
    base = usage.projections(season, week)
    if not base:
        return {}
    current = _weekly(season)
    if current.empty:
        return base
    prior = current[current.week < week]
    previous = _weekly(season - 1)
    volume = _volume_projection(prior, previous)
    context = week_context(season, week)
    vol_now = team_volume(prior)
    prior = prior.merge(vol_now, how="left", on=["season", "week", "team"])

    out: dict[tuple[str, str], Projection] = {}
    fallbacks = 0
    for stat in STATS:
        terms = TERMS.get(stat)
        if terms is None or stat not in prior.columns:
            out.update({key: proj for key, proj in base.items() if proj.stat == stat})
            continue
        volume_col = VOLUME_OF[stat]
        rows = prior[prior.position.isin(usage.POSITIONS[stat]) & prior[stat].notna()].copy()
        rows["share"] = rows[stat] / rows[volume_col].clip(lower=1.0)
        share_anchor: dict[str, float] = {}
        if not previous.empty and stat in previous.columns:
            past = previous.merge(team_volume(previous), how="left", on=["season", "week", "team"])
            past = past[past.position.isin(usage.POSITIONS[stat]) & past[stat].notna()]
            past_share = past[stat] / past[volume_col].clip(lower=1.0)
            share_anchor = past_share.groupby(past.player_id).mean().to_dict()
        position_share = rows.groupby("position")["share"].mean().to_dict()
        share_sum = rows.groupby("player_id")["share"].agg(["sum", "count"])
        opponent = _opponent_factors(prior, stat)

        for key, proj in base.items():
            if proj.stat != stat:
                continue
            game = context.get(proj.team)
            share_row = share_sum.loc[proj.player_id] if proj.player_id in share_sum.index else None
            team_volume_proj = volume.get((proj.team, volume_col))
            if game is None or share_row is None or team_volume_proj is None:
                fallbacks += 1
                out[key] = proj
                continue
            anchor = share_anchor.get(proj.player_id, position_share.get(proj.position, 0.0))
            share = shrunk_mean(float(share_row["sum"]), int(share_row["count"]), float(anchor))
            mean = adjusted_mean(
                terms,
                base=proj.mean,
                share_volume=share * team_volume_proj,
                opp_factor=opponent.get(game.opponent, 1.0),
                team_spread=game.team_spread,
                total_line=game.total_line,
            )
            out[key] = Projection(
                player=proj.player,
                player_id=proj.player_id,
                position=proj.position,
                team=proj.team,
                stat=stat,
                games=proj.games,
                mean=mean,
                prior_mean=proj.mean,
            )
    if fallbacks:
        log.info("context projections: %d of %d fell back to usage", fallbacks, len(out))
    return out


__all__ = [
    "BASIS",
    "OPP_SHRINK",
    "TERMS",
    "VOLUME_OF",
    "GameContext",
    "Terms",
    "adjusted_mean",
    "features",
    "projections",
    "week_context",
]
