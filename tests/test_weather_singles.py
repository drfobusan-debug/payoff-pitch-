"""The singles weather term, fitted rather than echoed off the home-run term."""

from __future__ import annotations

from mlb_engine.data.parks import get_park
from mlb_engine.filters.weather import WeatherConditions, WeatherEffect, _effect

WRIGLEY = 17  # open bowl, the most wind-receptive park on the card


def _hit(temp: float, out_cf: float) -> float:
    cond = WeatherConditions(temp, 50.0, abs(out_cf), 0.0, out_cf)
    return _effect(cond, get_park(WRIGLEY))[1]


def test_wind_moves_home_runs_and_leaves_singles_alone() -> None:
    """Wind to CF measured t = +3.3 on home runs and +0.06 on singles."""
    calm = WeatherConditions(70.0, 50.0, 0.0, 0.0, 0.0)
    gale = WeatherConditions(70.0, 50.0, 20.0, 0.0, 20.0)
    hr_calm, hit_calm = _effect(calm, get_park(WRIGLEY))
    hr_gale, hit_gale = _effect(gale, get_park(WRIGLEY))
    assert hr_gale > hr_calm * 1.15
    assert hit_gale == hit_calm  # the old echo made this ~1.09


def test_temperature_moves_singles_a_little() -> None:
    assert _hit(95.0, 0.0) > _hit(70.0, 0.0) > _hit(45.0, 0.0)
    # Half the fitted slope, so a 25-degree swing is worth ~2%.
    assert 1.015 < _hit(95.0, 0.0) < 1.025


def test_the_singles_term_is_capped_at_three_percent() -> None:
    assert _hit(115.0, 0.0) <= 1.03
    assert _hit(20.0, 0.0) >= 0.97


def test_doubles_and_triples_no_longer_carry_a_weather_term() -> None:
    """Temperature reads t = +0.31 and wind t = -0.10 on extra-base hits."""
    mults = WeatherEffect(None, hr_mult=1.20, hit_mult=1.02).multipliers()
    assert set(mults) == {"HR", "1B"}


def test_humidity_does_not_move_singles() -> None:
    dry = WeatherConditions(80.0, 15.0, 5.0, 0.0, 5.0)
    wet = WeatherConditions(80.0, 95.0, 5.0, 0.0, 5.0)
    assert _effect(dry, get_park(WRIGLEY))[1] == _effect(wet, get_park(WRIGLEY))[1]
