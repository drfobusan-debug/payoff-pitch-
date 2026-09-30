"""Everything about a live game that the odds feed does not know.

The Odds API sells prices, not context: an event carries two team names and a
kickoff time and nothing else. So a live :class:`~nfl_engine.schemas.Game` came
out of it with no roof, no rest days, no neutral-site flag and no divisional
flag -- and :mod:`nfl_engine.features.adjustments` reads exactly those fields.
Both of its *measured* terms (wind on the total, divisional on the total) were
therefore dead on every live slate, firing only in historical replay, where the
game file supplies the same columns after the fact.

nflverse's ``games.csv`` publishes the whole season's schedule the day it is
released, months before kickoff, with ``roof``, ``surface``, ``location``,
``div_game`` and both rest columns already filled in. None of that is a
forecast; it is the schedule. This module joins it onto the board by team pair.

One wrinkle worth naming: ``roof`` is blank for future games at the five
retractable-roof venues, because whether the roof is open is a game-day
decision. Blank is not "outdoors" -- calling Arizona in September an outdoor
game would hand its total a wind adjustment it will never see -- so an unknown
roof is backfilled with what that venue has actually done in past seasons, and
left unknown if it has no history.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from nfl_engine.data import nflverse, weather
from nfl_engine.schemas import Game

log = logging.getLogger(__name__)

_NEUTRAL = "neutral"


@dataclass(frozen=True)
class ScheduleContext:
    """The schedule's view of one game."""

    roof: str | None = None
    surface: str | None = None
    neutral_site: bool = False
    div_game: bool | None = None
    home_rest: int | None = None
    away_rest: int | None = None
    stadium_id: str | None = None


def _text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _int(value: object) -> int | None:
    """A schedule cell as a whole number, where a blank is a blank.

    nflverse types a column by what is in it, so the same field arrives as
    ``int``, ``float`` with a ``NaN`` for the games not yet played, or ``str``.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _flag(value: object) -> bool | None:
    number = _int(value)
    return None if number is None else bool(number)


def venue_roofs(games: pd.DataFrame) -> dict[str, str]:
    """The roof each venue has most often been played under, by ``stadium_id``.

    Only games that were actually played report a roof, which is what makes this
    a usable prior for the ones that have not been.
    """
    seen: dict[str, Counter[str]] = {}
    for stadium, roof in zip(games.get("stadium_id", []), games.get("roof", []), strict=False):
        key, value = _text(stadium), _text(roof)
        if key is None or value is None:
            continue
        seen.setdefault(key, Counter())[value] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in seen.items()}


def week_context(season: int, week: int) -> dict[str, ScheduleContext]:
    """Context for one week, keyed by ``"AWAY @ HOME"``.

    An empty dict is a normal answer: nflverse can be unreachable, and a slate
    priced without context is the behaviour that existed before this module.
    """
    frame = nflverse.games()
    if frame.empty:
        log.warning("nflverse schedule unavailable: pricing without game context")
        return {}
    roofs = venue_roofs(frame)
    frame = frame[(frame.season == season) & (frame.week == week)]
    out: dict[str, ScheduleContext] = {}
    for row in frame.to_dict("records"):
        home, away = _text(row.get("home_team")), _text(row.get("away_team"))
        if home is None or away is None:
            continue
        stadium = _text(row.get("stadium_id"))
        roof = _text(row.get("roof")) or (roofs.get(stadium) if stadium else None)
        out[f"{away} @ {home}"] = ScheduleContext(
            roof=roof,
            surface=_text(row.get("surface")),
            neutral_site=(_text(row.get("location")) or "").lower() == _NEUTRAL,
            div_game=_flag(row.get("div_game")),
            home_rest=_int(row.get("home_rest")),
            away_rest=_int(row.get("away_rest")),
            stadium_id=stadium,
        )
    return out


def contexts_for(games: list[Game]) -> dict[str, ScheduleContext]:
    """Context for every game on a slate, by matchup, fetching each week once."""
    by_week: dict[tuple[int, int], dict[str, ScheduleContext]] = {}
    out: dict[str, ScheduleContext] = {}
    for game in games:
        key = (game.season, game.week)
        if key not in by_week:
            by_week[key] = week_context(*key)
        context = by_week[key].get(game.matchup())
        if context is not None:
            out[game.matchup()] = context
    if games and len(out) < len(games):
        log.info("schedule context matched %d of %d games", len(out), len(games))
    return out


def enrich(games: list[Game]) -> list[Game]:
    """Schedule context, then a kickoff forecast for whatever plays outdoors.

    The order matters: the roof decides whether the weather is worth asking
    about, so the schedule has to land first.
    """
    if not games:
        return games
    contexts = contexts_for(games)
    placed = [
        apply_context(game, contexts[game.matchup()]) if game.matchup() in contexts else game
        for game in games
    ]
    venues = {
        matchup: context.stadium_id for matchup, context in contexts.items() if context.stadium_id
    }
    return weather.attach_forecast(placed, venues)


def apply_context(game: Game, context: ScheduleContext) -> Game:
    env = game.env.model_copy(
        update={
            "roof": game.env.roof or context.roof,
            "surface": game.env.surface or context.surface,
            "neutral_site": game.env.neutral_site or context.neutral_site,
        }
    )
    return game.model_copy(
        update={
            "env": env,
            "div_game": game.div_game if game.div_game is not None else context.div_game,
            "home_rest": game.home_rest if game.home_rest is not None else context.home_rest,
            "away_rest": game.away_rest if game.away_rest is not None else context.away_rest,
        }
    )
