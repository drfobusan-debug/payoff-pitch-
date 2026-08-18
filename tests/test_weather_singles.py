"""The weather prices home runs and nothing else, because that is what it moves."""

from __future__ import annotations

from pathlib import Path

from mlb_engine.data.parks import get_park
from mlb_engine.filters.weather import WeatherConditions, WeatherEffect, WeatherProvider, _effect

WRIGLEY = 17  # open bowl, the most wind-receptive park on the card


def test_the_weather_no_longer_touches_hits() -> None:
    """Fitted against realised singles: wind t = +0.06, humidity t = -0.51.

    The old term echoed 35% of the home-run carry effect onto 1B/2B/3B, which
    added up to 10% to a hitter's singles number on a windy night.
    """
    assert set(WeatherEffect(None, hr_mult=1.25).multipliers()) == {"HR"}


def test_wind_and_temperature_still_price_home_runs() -> None:
    calm = WeatherConditions(70.0, 50.0, 0.0, 0.0, 0.0)
    gale = WeatherConditions(70.0, 50.0, 20.0, 0.0, 20.0)
    hot = WeatherConditions(95.0, 50.0, 0.0, 0.0, 0.0)
    park = get_park(WRIGLEY)
    assert _effect(gale, park) > _effect(calm, park) * 1.15
    assert _effect(hot, park) > _effect(calm, park)


def test_a_closed_roof_is_neutral() -> None:
    assert WeatherEffect(None, 1.0, note="closed roof").multipliers() == {"HR": 1.0}


class _MovingForecast:
    """Open-Meteo as it really behaves: a different answer on every call."""

    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, params: dict[str, float | str], timeout: int) -> _MovingForecast:
        self.calls += 1
        return self

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, dict[str, list[float] | list[str]]]:
        return {
            "hourly": {
                "time": ["2026-08-16T19:00"],
                "temperature_2m": [70.0 + self.calls],
                "relative_humidity_2m": [50.0],
                "wind_speed_10m": [10.0 * self.calls],
                "wind_direction_10m": [180.0],
            }
        }


def test_the_same_slate_is_priced_on_the_same_forecast(tmp_path: Path) -> None:
    """Two runs of one slate must agree; an uncached provider does not.

    A forecast revised between runs moved 6,050 of 6,705 priced rows, mean
    1.23pp, which swamped most of what a before/after comparison is trying to
    measure.
    """
    park = get_park(WRIGLEY)
    when = "2026-08-16T19:00:00Z"

    live = _MovingForecast()
    drifting = WeatherProvider(session=live)
    assert drifting.fetch(park, when).hr_mult != drifting.fetch(park, when).hr_mult

    session = _MovingForecast()
    cached = WeatherProvider(session=session, cache_dir=tmp_path, cache_ttl=1800)
    first = cached.fetch(park, when)
    # A fresh provider on the same cache directory: a re-run, not a warm object.
    again = WeatherProvider(session=session, cache_dir=tmp_path, cache_ttl=1800)
    assert again.fetch(park, when).hr_mult == first.hr_mult
    assert session.calls == 1


def test_a_past_slate_never_refetches(tmp_path: Path) -> None:
    """Archive weather cannot change, so the TTL does not apply to it."""
    session = _MovingForecast()
    provider = WeatherProvider(session=session, cache_dir=tmp_path, cache_ttl=0)
    park = get_park(WRIGLEY)
    old = "2020-08-16T19:00:00Z"
    assert provider.fetch(park, old).hr_mult == provider.fetch(park, old).hr_mult
    assert session.calls == 1
