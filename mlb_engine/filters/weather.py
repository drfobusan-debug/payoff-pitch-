"""Weather filter using the free Open-Meteo forecast API.

Pulls temperature, humidity, wind speed and direction for the ballpark at game
time, then converts them into bounded multipliers on hit/HR outcomes. Wind is
projected onto the home-plate -> center-field axis so a wind blowing out boosts
power and a wind blowing in suppresses it. Domed / closed-roof parks neutralize
weather.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from mlb_engine.data.parks import Park

log = logging.getLogger(__name__)

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


@dataclass
class WeatherConditions:
    temp_f: float
    humidity_pct: float
    wind_mph: float
    wind_from_deg: float  # meteorological: direction wind blows FROM
    out_to_cf_mph: float  # positive = blowing out to center

    def summary(self) -> str:
        d = "out" if self.out_to_cf_mph >= 0 else "in"
        return (
            f"{self.temp_f:.0f}F {self.humidity_pct:.0f}% "
            f"wind {self.wind_mph:.0f}mph ({abs(self.out_to_cf_mph):.0f} {d} to CF)"
        )


@dataclass
class WeatherEffect:
    conditions: WeatherConditions | None
    hr_mult: float
    hit_mult: float
    note: str = ""

    def multipliers(self) -> dict[str, float]:
        return {"HR": self.hr_mult, "1B": self.hit_mult, "2B": self.hit_mult, "3B": self.hit_mult}


def _wind_out_component(wind_from_deg: float, wind_mph: float, cf_bearing: float) -> float:
    # wind blows TO (from + 180). Out-to-CF component = wind_to . cf_bearing unit.
    wind_to = (wind_from_deg + 180.0) % 360.0
    angle = math.radians(wind_to - cf_bearing)
    return wind_mph * math.cos(angle)


class WeatherProvider:
    def __init__(self, session: requests.Session | None = None, timeout: int = 20) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(self, park: Park, game_dt_utc: str | None) -> WeatherEffect:
        if park.roof in ("closed", "dome"):
            return WeatherEffect(None, 1.0, 1.0, note="closed roof: weather neutral")

        try:
            cond = self._fetch_conditions(park, game_dt_utc)
        except Exception as exc:  # network / API issues shouldn't kill the run
            log.warning("weather fetch failed for %s: %s", park.name, exc)
            return WeatherEffect(None, 1.0, 1.0, note="weather unavailable")

        if park.roof == "retractable":
            # Unknown whether open; damp the effect by half.
            hr, hit = _effect(cond)
            return WeatherEffect(
                cond, 1.0 + (hr - 1.0) * 0.5, 1.0 + (hit - 1.0) * 0.5, note="retractable (damped)"
            )

        hr, hit = _effect(cond)
        return WeatherEffect(cond, hr, hit, note="open")

    def _fetch_conditions(self, park: Park, game_dt_utc: str | None) -> WeatherConditions:
        game_dt = (
            datetime.fromisoformat(game_dt_utc.replace("Z", "+00:00"))
            if game_dt_utc
            else datetime.utcnow()
        )
        date_str = game_dt.date().isoformat()
        params: dict[str, float | str] = {
            "latitude": park.lat,
            "longitude": park.lon,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "UTC",
        }
        # Forecast API only covers the recent past + near future; older dates
        # (backtests) come from the historical archive API.
        days_ago = (datetime.now(timezone.utc).date() - game_dt.date()).days
        url = OPEN_METEO_ARCHIVE if days_ago > 5 else OPEN_METEO
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        h = resp.json()["hourly"]
        times = [datetime.fromisoformat(t) for t in h["time"]]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - game_dt.replace(tzinfo=None)))
        temp = h["temperature_2m"][idx]
        hum = h["relative_humidity_2m"][idx]
        wspd = h["wind_speed_10m"][idx]
        wdir = h["wind_direction_10m"][idx]
        out = _wind_out_component(wdir, wspd, park.orientation_deg)
        return WeatherConditions(temp, hum, wspd, wdir, out)


def _effect(c: WeatherConditions) -> tuple[float, float]:
    """Return (hr_mult, hit_mult) from conditions. Bounded and modest."""
    hr = 1.0
    # temperature: ~+2.5% HR per +10F above 70
    hr *= 1.0 + max(-0.15, min(0.15, (c.temp_f - 70.0) * 0.0025))
    # humidity: humid air is less dense -> slight carry
    hr *= 1.0 + max(-0.03, min(0.03, (c.humidity_pct - 50.0) * 0.0005))
    # wind out to CF: ~+1% HR per mph out
    hr *= 1.0 + max(-0.20, min(0.25, c.out_to_cf_mph * 0.010))
    hr = max(0.75, min(1.35, hr))

    # hits track a damped version of the same drivers
    hit = 1.0 + (hr - 1.0) * 0.35
    hit = max(0.90, min(1.12, hit))
    return hr, hit
