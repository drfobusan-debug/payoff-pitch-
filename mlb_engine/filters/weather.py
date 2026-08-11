"""Weather filter using the free Open-Meteo forecast API.

Pulls temperature, humidity, wind speed and direction for the ballpark at game
time, then converts them into bounded multipliers on hit/HR outcomes via a
WAM-style park-configuration filter: wind is projected onto the home-plate ->
center-field axis and gated by each park's structural wind-receptivity, while
temperature/wind HR payoff is scaled by the park's fence/outfield profile.
Humidity is down-weighted (humidor-neutralized). Domed / closed-roof parks
neutralize weather.

The weather does not touch hits, because it does not measure as touching hits
----------------------------------------------------------------------------
There used to be a hit multiplier, ``1 + (hr_mult - 1) * 0.35``: the carry model
at 35% strength, a figure asserted rather than measured, applied to 1B/2B/3B.
Carry is the wrong mechanism for singles -- it is what turns a single into a
home run -- so the sign was not even obvious, and the wind channel dominated
the number.

Fitted on 3,347 game-halves (128,318 plate appearances), with park and month
fixed effects and each offence's season rate controlled, standard errors
clustered by game:

    singles/PA        coef        t        home runs/PA for comparison
    temperature    +0.000224    +1.73     +0.000424   t +6.61
    wind to CF     +0.000012    +0.06     +0.000317   t +3.29
    crosswind      -0.000101    -0.22     -0.000318   t -1.48
    wind speed     -0.000167    -0.40     +0.000041   t +0.20
    humidity       -0.000034    -0.51     +0.000069   t +2.12

The home-run column reproduces the physics, which is the evidence that the
measurement works. The singles column does not: **wind does nothing to singles**
(t = 0.06 on the very component the old term was driven by), and neither does
humidity. Only temperature shows anything, and it is weak -- t = 1.7 -- and the
coefficient does not replicate across alternate days (+0.00033 against
+0.00007). Extra-base hits are the same story: temperature t = +0.31, wind
t = -0.10.

So the hit term is gone rather than refitted. A multiplier that is nominally
"small" is not harmless when it is unfounded: the old one added up to 10% to a
hitter's singles number on a windy night, which is a phantom edge on a market
whose real park spread is only a few percent. The weather still prices home
runs, where it belongs and where it measures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from mlb_engine.data import http
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
    note: str = ""

    def multipliers(self) -> dict[str, float]:
        return {"HR": self.hr_mult}


def _wind_out_component(wind_from_deg: float, wind_mph: float, cf_bearing: float) -> float:
    # wind blows TO (from + 180). Out-to-CF component = wind_to . cf_bearing unit.
    wind_to = (wind_from_deg + 180.0) % 360.0
    angle = math.radians(wind_to - cf_bearing)
    return wind_mph * math.cos(angle)


class WeatherProvider:
    def __init__(self, session: requests.Session | None = None, timeout: int = 20) -> None:
        self.session = session or http.session()
        self.timeout = timeout

    def fetch(self, park: Park, game_dt_utc: str | None) -> WeatherEffect:
        if park.roof in ("closed", "dome"):
            return WeatherEffect(None, 1.0, note="closed roof: weather neutral")

        try:
            cond = self._fetch_conditions(park, game_dt_utc)
        except Exception as exc:  # network / API issues shouldn't kill the run
            log.warning("weather fetch failed for %s: %s", park.name, exc)
            return WeatherEffect(None, 1.0, note="weather unavailable")

        if park.roof == "retractable":
            # Unknown whether open; damp the effect by half.
            hr = _effect(cond, park)
            return WeatherEffect(
                cond, 1.0 + (hr - 1.0) * 0.5, note="retractable (damped)"
            )

        return WeatherEffect(cond, _effect(cond, park), note="open")

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


def _effect(c: WeatherConditions, park: Park) -> float:
    """Return the home-run multiplier via a WAM-style park-configuration filter.

    Temperature (~3.3 ft / +10F) and wind (~19 ft / 5 mph out) drive carry, but
    the HR payoff is scaled by the park's fence/outfield profile (``carry_factor``)
    and the wind by its structural receptivity (``wind_factor``). Humidity is
    down-weighted to near-noise since humidors neutralize it.
    """
    carry = park.carry_factor
    wind_recv = park.wind_factor

    # temperature: ~+2.8% carry per +10F above 70, amplified/damped by fence profile
    temp_term = max(-0.15, min(0.15, (c.temp_f - 70.0) * 0.0028)) * carry
    # humidity: humidor-neutralized -> treat as low-PPV noise
    humid_term = max(-0.015, min(0.015, (c.humidity_pct - 50.0) * 0.0002))
    # wind out/in to CF: ~+1.1% HR per mph, gated by park wind-receptivity and fences
    wind_term = max(-0.30, min(0.35, c.out_to_cf_mph * 0.011 * wind_recv)) * carry

    hr = (1.0 + temp_term) * (1.0 + humid_term) * (1.0 + wind_term)
    return max(0.70, min(1.40, hr))
