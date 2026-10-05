"""Each team's schedule state going into a game: rest, back-to-backs, travel, altitude.

Built from the archived finals alone, in date order, so a game's reading uses
only games already played. Distances run from the site of the team's previous
game to tonight's site (a road trip is charged leg by leg, not from home).
Tonight's reading needs only the schedule, so it is known before tip.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_engine.data import arenas
from nba_engine.schemas import GameResult

REST_CAP = 4  # 4+ days off reads as fully rested; a season opener reads as REST_CAP


@dataclass(frozen=True)
class TeamSchedule:
    rest_days: int
    b2b: bool  # second night of a back-to-back
    three_in_four: bool  # third game in four nights
    miles: float | None  # since the previous game; None when a site is unknown
    tz_shift: float | None  # hours, + travelling east
    altitude: bool  # visiting a 4,000 ft+ site from below it
    road_game: int  # 0 at home, else the game number of the current road trip


def season_of(day: Date) -> int:
    """Start year of the season a date belongs to (``2025`` for 2025-26)."""
    return day.year if day.month >= 8 else day.year - 1


def _utc_offset(tz: str, day: Date) -> float:
    off = datetime(day.year, day.month, day.day, 20, tzinfo=ZoneInfo(tz)).utcoffset()
    return off.total_seconds() / 3600.0 if off is not None else 0.0


def schedule(games: Iterable[GameResult]) -> dict[tuple[str, str], TeamSchedule]:
    """``(espn_id, team)`` -> that team's schedule state going into the game."""
    out: dict[tuple[str, str], TeamSchedule] = {}
    played: dict[str, list[Date]] = {}
    where: dict[str, arenas.Site | None] = {}
    trip: dict[str, int] = {}
    for g in sorted(games, key=lambda g: (g.game_date, g.espn_id)):
        site = arenas.game_site(g.home, g.neutral, g.city)
        for team in (g.away, g.home):
            past = [d for d in played.get(team, []) if season_of(d) == season_of(g.game_date)]
            rest = min((g.game_date - past[-1]).days, REST_CAP) if past else REST_CAP
            three = len(past) >= 2 and (g.game_date - past[-2]).days <= 3
            prev = where.get(team) if past else arenas.HOME.get(team)
            dist = arenas.miles(prev, site) if prev is not None and site is not None else None
            shift = (
                _utc_offset(site.tz, g.game_date) - _utc_offset(prev.tz, g.game_date)
                if prev is not None and site is not None
                else None
            )
            home_site = arenas.HOME.get(team)
            lives_high = home_site is not None and home_site.elevation_ft >= arenas.ALTITUDE_FT
            altitude = (
                site is not None and site.elevation_ft >= arenas.ALTITUDE_FT and not lives_high
            )
            away = team == g.away or g.neutral
            road = (trip.get(team, 0) + 1) if away and past else (1 if away else 0)
            out[(g.espn_id, team)] = TeamSchedule(
                rest_days=rest,
                b2b=rest == 1,
                three_in_four=three,
                miles=dist,
                tz_shift=shift,
                altitude=altitude,
                road_game=road,
            )
            played.setdefault(team, []).append(g.game_date)
            where[team] = site
            trip[team] = road
    return out


__all__ = ["REST_CAP", "TeamSchedule", "schedule", "season_of"]
