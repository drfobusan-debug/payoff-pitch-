"""Assemble per-game situational context (rest, travel, weather, venue) from
CFBD schedule/venues/teams/weather feeds, keyed by the unordered team pair.

Every field is optional: a missing feed simply leaves that field ``None`` and the
adjustment layer skips the corresponding nudge. Weather is the one that was
*always* missing -- CFBD's ``/games/weather`` needs a Patreon tier, so on a free
key every wind/precipitation/temperature field came back ``None`` and the totals
nudges keyed on them never ran. Open-Meteo backfills the outdoor games CFBD does
not cover, off the venue coordinates already loaded here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import date as Date
from datetime import datetime, timezone

from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.data.teamnames import school_key
from cfb_engine.schemas import Slate
from mlb_engine.data.openmeteo import fetch_venue_weather

log = logging.getLogger(__name__)

_EARTH_MILES = 3958.8


@dataclass(frozen=True)
class GameContext:
    neutral_site: bool = False
    dome: bool = False
    rest_home: int | None = None
    rest_away: int | None = None
    travel_away_miles: float | None = None
    wind_mph: float | None = None
    precipitation: float | None = None
    temperature_f: float | None = None


ContextBook = dict[frozenset[str], GameContext]


def _pair(home: str, away: str) -> frozenset[str]:
    return frozenset({school_key(home), school_key(away)})


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return _EARTH_MILES * 2 * math.asin(math.sqrt(a))


def _game_day(start_date: str) -> Date | None:
    try:
        return Date.fromisoformat(start_date[:10])
    except ValueError:
        return None


def _kickoff(start_date: str) -> datetime | None:
    try:
        return datetime.fromisoformat(start_date.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def build_context_book(cfbd: CFBDClient, season: int, slate: Slate) -> ContextBook:
    """Build the slate's context book. Returns empty when CFBD is unavailable."""
    if not cfbd.available():
        return {}
    schedule = cfbd.fetch_schedule(season)
    if not schedule:
        return {}
    venues = cfbd.fetch_venues()
    team_locs = cfbd.fetch_team_locations(season)
    weather = {_pair(w.home, w.away): w for w in cfbd.fetch_weather(season)}

    # Per-team kickoff dates (sorted) to derive rest days before each game.
    team_days: dict[str, list[Date]] = {}
    for meta in schedule:
        day = _game_day(meta.start_date)
        if day is None:
            continue
        for team in (meta.home, meta.away):
            team_days.setdefault(school_key(team), []).append(day)
    for days in team_days.values():
        days.sort()

    def rest_days(team: str, day: Date) -> int | None:
        prior = [d for d in team_days.get(school_key(team), []) if d < day]
        return (day - prior[-1]).days if prior else None

    wanted = {_pair(g.home.name, g.away.name) for g in slate.games}
    book: ContextBook = {}
    # Outdoor games CFBD did not report weather for, to backfill in one batch.
    pending: dict[str, tuple[float, float, datetime]] = {}
    keys_by_id: dict[str, frozenset[str]] = {}
    for meta in schedule:
        key = _pair(meta.home, meta.away)
        if key not in wanted or key in book:
            continue
        day = _game_day(meta.start_date)
        venue = venues.get(meta.venue_id) if meta.venue_id is not None else None
        travel = None
        away_loc = team_locs.get(school_key(meta.away))
        if venue is not None and away_loc is not None:
            travel = haversine_miles(
                away_loc.latitude, away_loc.longitude, venue.latitude, venue.longitude
            )
        wx = weather.get(key)
        dome = bool(venue.dome if venue is not None else False) or bool(wx.dome if wx else False)
        book[key] = GameContext(
            neutral_site=meta.neutral_site,
            dome=dome,
            rest_home=rest_days(meta.home, day) if day is not None else None,
            rest_away=rest_days(meta.away, day) if day is not None else None,
            travel_away_miles=travel,
            wind_mph=wx.wind_mph if wx else None,
            precipitation=wx.precipitation if wx else None,
            temperature_f=wx.temperature_f if wx else None,
        )
        kick = _kickoff(meta.start_date)
        if wx is None and not dome and venue is not None and kick is not None:
            ident = f"{meta.home}|{meta.away}|{meta.start_date}"
            pending[ident] = (venue.latitude, venue.longitude, kick)
            keys_by_id[ident] = key

    if pending:
        for ident, obs in fetch_venue_weather(pending).items():
            key = keys_by_id[ident]
            base = book.get(key)
            if base is None:
                continue
            book[key] = replace(
                base,
                wind_mph=obs.wind_mph,
                precipitation=obs.precipitation,
                temperature_f=obs.temperature_f,
            )
        log.info(
            "backfilled weather for %d of %d outdoor games CFBD did not cover",
            sum(1 for k in keys_by_id.values() if book[k].wind_mph is not None),
            len(pending),
        )
    return book


def context_for(book: ContextBook, home_name: str, away_name: str) -> GameContext:
    return book.get(_pair(home_name, away_name), GameContext())
