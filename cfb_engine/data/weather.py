"""Kickoff weather from Open-Meteo, for the games CFBD will not report on.

CFBD's ``/games/weather`` sits behind a Patreon tier, so on a free key it
answers 401 and the weather fields have always come back ``None`` -- the totals
nudges keyed on them never fired at all. Open-Meteo needs no key and takes a
venue's coordinates plus a kickoff hour, both of which the context book already
has, so it fills the gap for whatever CFBD does not cover.

Requests are batched by date: the archive/forecast endpoints accept parallel
coordinate lists, so a full Saturday board is one call rather than sixty. Every
failure is soft -- a blocked request or a changed payload leaves the game's
weather ``None`` and the adjustment layer skips it, exactly as before.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mlb_engine.data import http

log = logging.getLogger(__name__)

_FORECAST = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
# Past this gap the nearest available hour is not this game's weather.
_MAX_HOUR_GAP_S = 5400
_HOURLY = "wind_speed_10m,wind_gusts_10m,precipitation,temperature_2m"


@dataclass(frozen=True)
class VenueWeather:
    wind_mph: float | None
    gust_mph: float | None
    precipitation: float | None  # mm in the kickoff hour
    temperature_f: float | None


def _pick_hour(hourly: dict[str, list[object]], kickoff: datetime) -> int | None:
    """Index of the hour closest to kickoff, or ``None`` if none is close."""
    stamps = hourly.get("time")
    if not isinstance(stamps, list) or not stamps:
        return None
    times: list[datetime] = []
    for raw in stamps:
        try:
            times.append(datetime.fromisoformat(str(raw)).replace(tzinfo=timezone.utc))
        except ValueError:
            return None
    idx = min(range(len(times)), key=lambda i: abs((times[i] - kickoff).total_seconds()))
    if abs((times[idx] - kickoff).total_seconds()) > _MAX_HOUR_GAP_S:
        return None
    return idx


def _value(hourly: dict[str, list[object]], field: str, idx: int) -> float | None:
    series = hourly.get(field)
    if not isinstance(series, list) or idx >= len(series):
        return None
    raw = series[idx]
    return float(raw) if isinstance(raw, (int, float)) else None


def fetch_venue_weather(
    points: dict[str, tuple[float, float, datetime]], *, timeout: float = 20.0
) -> dict[str, VenueWeather]:
    """Weather at kickoff for each ``key -> (lat, lon, kickoff UTC)``.

    Keys absent from the result had no usable reading; callers treat that the
    same as never having asked.
    """
    by_date: dict[str, list[str]] = defaultdict(list)
    for key, (_lat, _lon, kick) in points.items():
        by_date[kick.date().isoformat()].append(key)

    out: dict[str, VenueWeather] = {}
    today = datetime.now(timezone.utc).date()
    for day, keys in by_date.items():
        # The forecast endpoint only carries the near future; older dates have
        # to come from the reanalysis archive.
        past = datetime.fromisoformat(day).date() < today - timedelta(days=2)
        url = _ARCHIVE if past else _FORECAST
        lats = [points[k][0] for k in keys]
        lons = [points[k][1] for k in keys]
        params = {
            "latitude": ",".join(f"{v:.4f}" for v in lats),
            "longitude": ",".join(f"{v:.4f}" for v in lons),
            "start_date": day,
            "end_date": day,
            "hourly": _HOURLY,
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
        }
        try:
            resp = http.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - weather is an optional nudge
            log.warning("open-meteo weather unavailable for %s: %s", day, exc)
            continue
        # One location answers as an object, several as a list in request order.
        locs = body if isinstance(body, list) else [body]
        if len(locs) != len(keys):
            log.warning(
                "open-meteo returned %d locations for %d venues on %s: skipping",
                len(locs),
                len(keys),
                day,
            )
            continue
        for key, loc in zip(keys, locs, strict=True):
            hourly = loc.get("hourly") if isinstance(loc, dict) else None
            if not isinstance(hourly, dict):
                continue
            idx = _pick_hour(hourly, points[key][2])
            if idx is None:
                continue
            out[key] = VenueWeather(
                wind_mph=_value(hourly, "wind_speed_10m", idx),
                gust_mph=_value(hourly, "wind_gusts_10m", idx),
                precipitation=_value(hourly, "precipitation", idx),
                temperature_f=_value(hourly, "temperature_2m", idx),
            )
    return out
