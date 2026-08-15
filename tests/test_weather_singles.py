"""The weather prices home runs and nothing else, because that is what it moves."""

from __future__ import annotations

from mlb_engine.data.parks import get_park
from mlb_engine.filters.weather import WeatherConditions, WeatherEffect, _effect

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
