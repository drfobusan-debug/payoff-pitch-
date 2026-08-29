"""Kickoff wind and temperature for the outdoor games on a live slate.

nflverse reports ``temp`` and ``wind`` only for games that have been played --
they are a post-game report, not a forecast -- so a Sunday-morning board has
neither, and the one measured weather term in
:mod:`nfl_engine.features.adjustments` (-0.20 points of total per mph over an
8.5 mph average) had nothing to read. Open-Meteo fills that in from the venue's
coordinates and the kickoff hour, without a key.

The adjustments module is explicit that this is the right source *and* that it
attenuates: the historical fit is ERA5 reanalysis, while a day-ahead forecast
correlates r=+0.72 with the wind actually reported at kickoff. The term is set
to the shallower market-relative slope partly for that reason.

Indoor games are never asked about, which is why the roof backfill in
:mod:`nfl_engine.data.schedule` matters more than it looks: a game whose roof is
unknown is treated as unknown, not as outdoors, so it gets no reading and no
adjustment rather than a wind number for a room.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from mlb_engine.data.openmeteo import VenueWeather, fetch_venue_weather
from nfl_engine.schemas import Game

log = logging.getLogger(__name__)

# Latitude and longitude by nflverse ``stadium_id``, for every venue that plays
# under the sky -- domes are omitted because they are never queried. The
# international sites are here too: London, Munich, Mexico City and Sao Paulo
# have all hosted regular-season games, and a 6 a.m. Eastern kickoff in a
# European autumn is exactly the game whose wind nobody has checked.
VENUE_COORDS: dict[str, tuple[float, float]] = {
    "BAL00": (39.2780, -76.6227),  # M&T Bank Stadium
    "BOS00": (42.0909, -71.2643),  # Gillette Stadium
    "BUF00": (42.7738, -78.7870),  # Highmark Stadium
    "CAR00": (35.2258, -80.8528),  # Bank of America Stadium
    "CHI98": (41.8623, -87.6167),  # Soldier Field
    "CIN00": (39.0955, -84.5161),  # Paycor Stadium
    "CLE00": (41.5061, -81.6995),  # Huntington Bank Field
    "DEN00": (39.7439, -105.0201),  # Empower Field at Mile High
    "GNB00": (44.5013, -88.0622),  # Lambeau Field
    "JAX00": (30.3239, -81.6373),  # EverBank Stadium
    "KAN00": (39.0489, -94.4839),  # GEHA Field at Arrowhead Stadium
    "MIA00": (25.9580, -80.2389),  # Hard Rock Stadium
    "NAS00": (36.1665, -86.7713),  # Nissan Stadium
    "NYC01": (40.8135, -74.0745),  # MetLife Stadium
    "PHI00": (39.9008, -75.1675),  # Lincoln Financial Field
    "PIT00": (40.4468, -80.0158),  # Acrisure Stadium
    "SEA00": (47.5952, -122.3316),  # Lumen Field
    "SFO01": (37.4030, -121.9698),  # Levi's Stadium
    "TAM00": (27.9759, -82.5033),  # Raymond James Stadium
    "WAS00": (38.9077, -76.8645),  # Northwest Stadium
    "GER00": (48.2188, 11.6247),  # Allianz Arena, Munich
    "LON00": (51.5560, -0.2795),  # Wembley Stadium
    "LON02": (51.6043, -0.0665),  # Tottenham Hotspur Stadium
    "MAD01": (40.4531, -3.6883),  # Santiago Bernabeu
    "MEX00": (19.3029, -99.1505),  # Estadio Banorte
    "RIO00": (-22.9121, -43.2302),  # Maracana
    "SAO00": (-23.5453, -46.4742),  # Arena Corinthians
}


def _kickoff(game: Game) -> datetime | None:
    return parse_kickoff(game.kickoff_utc or "")


def parse_kickoff(stamp: str) -> datetime | None:
    """A board or ledger kickoff stamp as an aware UTC time, or ``None``."""
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_outdoors(roof: str | None) -> bool:
    """Only a stated open roof counts. Unknown is unknown, never outdoors."""
    return (roof or "").strip().lower() in ("outdoors", "open")


def readings(points: dict[str, tuple[str, datetime]]) -> dict[str, VenueWeather]:
    """Weather by key, for ``key -> (stadium_id, kickoff)``, best effort.

    Open-Meteo serves its archive as well as its forecast, so this answers for a
    week already played as readily as for one about to be.
    """
    located = {
        key: (*VENUE_COORDS[stadium], kickoff)
        for key, (stadium, kickoff) in points.items()
        if stadium in VENUE_COORDS
    }
    if not located:
        return {}
    try:
        return fetch_venue_weather(located)
    except Exception as exc:  # noqa: BLE001 - weather is never worth a failed run
        log.warning("kickoff weather unavailable (%s)", exc)
        return {}


def attach_forecast(games: list[Game], venues: dict[str, str]) -> list[Game]:
    """Fill wind and temperature on outdoor games, from ``matchup -> stadium_id``.

    Every failure is silent by design: an unknown venue, an unknown roof, a
    missing kickoff time or an unreachable Open-Meteo all leave the game exactly
    as it arrived, and the wind term then declines to fire rather than guessing.
    """
    points: dict[str, tuple[float, float, datetime]] = {}
    for game in games:
        if game.env.is_indoors() or game.env.wind_mph is not None:
            continue
        # Unknown roof is not outdoors: no reading, no adjustment.
        if not is_outdoors(game.env.roof):
            continue
        coords = VENUE_COORDS.get(venues.get(game.matchup(), ""))
        kickoff = _kickoff(game)
        if coords is None or kickoff is None:
            continue
        points[game.matchup()] = (coords[0], coords[1], kickoff)
    if not points:
        return games
    try:
        observed = fetch_venue_weather(points)
    except Exception as exc:  # noqa: BLE001 - the slate prices without weather
        log.warning("kickoff weather unavailable (%s)", exc)
        return games
    if not observed:
        log.warning("no kickoff weather for %d outdoor games", len(points))
        return games
    out: list[Game] = []
    for game in games:
        reading = observed.get(game.matchup())
        if reading is None:
            out.append(game)
            continue
        env = game.env.model_copy(
            update={
                "wind_mph": game.env.wind_mph
                if game.env.wind_mph is not None
                else reading.wind_mph,
                "temp_f": game.env.temp_f if game.env.temp_f is not None else reading.temperature_f,
            }
        )
        out.append(game.model_copy(update={"env": env}))
    log.info("kickoff weather attached to %d of %d outdoor games", len(observed), len(points))
    return out
