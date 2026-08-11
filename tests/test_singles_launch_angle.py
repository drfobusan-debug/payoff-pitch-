"""Mean launch angle on the singles line.

The ground-ball rate next door is a threshold count of this quantity, so the
test that matters is the last one: two hitters with the *same* ground-ball rate
and different launch angles must not price identically.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_GB_RATE,
    BL_MEAN_LA,
    SINGLES_LA_SLOPE,
    BatterRegression,
    build_batter_regression,
)

NAN = float("nan")


def _reg(mean_la: float = BL_MEAN_LA, gb_rate: float = BL_GB_RATE) -> BatterRegression:
    """A league-average batter apart from launch angle and ground-ball rate."""
    return BatterRegression(
        bbe=100,
        barrel_rate=0.080,
        hard_hit=0.400,
        sweet_spot=0.330,
        bat_speed=71.5,
        max_ev=108.0,
        whiff=0.240,
        zone_contact=0.820,
        xba=0.250,
        xslg=0.400,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
        gb_rate=gb_rate,
        mean_la=mean_la,
    )


def test_the_league_average_launch_angle_is_neutral() -> None:
    assert _reg().multipliers()["1B"] == pytest.approx(1.0, abs=5e-3)


def test_a_lower_launch_angle_lifts_singles_and_a_higher_one_suppresses() -> None:
    # The ball hit lower falls in front of the defence; the one hit in the air
    # is caught, or it is not a single.
    low = _reg(mean_la=6.0).multipliers()
    high = _reg(mean_la=21.0).multipliers()
    assert low["1B"] > 1.0 > high["1B"]


def test_the_term_is_singles_only() -> None:
    # Extra-base and home-run lines read launch angle through their own terms
    # (sweet spot, barrel, pulled air), never through this one.
    low = _reg(mean_la=6.0).multipliers()
    high = _reg(mean_la=21.0).multipliers()
    for key in ("2B", "3B", "HR"):
        assert low[key] == high[key]


def test_the_term_is_bounded() -> None:
    # A 42-day launch angle is a real measurement but not an unlimited one.
    assert _reg(mean_la=-10.0).multipliers()["1B"] <= 1.06
    assert _reg(mean_la=40.0).multipliers()["1B"] >= 0.94


def test_the_term_is_live_by_default_and_switchable() -> None:
    assert Config().singles_la is True
    assert Config().singles_la_slope == SINGLES_LA_SLOPE
    tilted = _reg(mean_la=6.0)
    assert tilted.multipliers()["1B"] > 1.0
    assert tilted.multipliers(singles_la_slope=0.0)["1B"] == pytest.approx(1.0, abs=5e-3)


def test_a_batter_with_no_launch_angle_data_drops_the_term() -> None:
    # NaN means the measurement is missing, which must cost the hitter nothing.
    assert _reg(mean_la=NAN).multipliers()["1B"] == pytest.approx(1.0, abs=5e-3)


def test_launch_angle_separates_hitters_the_ground_ball_rate_calls_identical() -> None:
    # The whole point of the term. Both hitters put 42% of their contact on the
    # ground, so the rate term prices them the same; one averages 4 degrees and
    # the other 20, and the singles line now says so.
    flat = _reg(mean_la=4.0, gb_rate=0.42)
    airborne = _reg(mean_la=20.0, gb_rate=0.42)
    assert flat.gb_rate == airborne.gb_rate
    assert flat.multipliers()["1B"] > airborne.multipliers()["1B"]
    # And the gap is the launch-angle term alone: switch it off and they agree.
    assert flat.multipliers(singles_la_slope=0.0)["1B"] == pytest.approx(
        airborne.multipliers(singles_la_slope=0.0)["1B"]
    )


def test_mean_launch_angle_is_read_off_the_batted_balls() -> None:
    rows = [
        {
            "batter": 1,
            "launch_speed": 90.0,
            "launch_angle": 4.0 if i < 10 else 24.0,
            "launch_speed_angle": 3,
            "bb_type": "ground_ball" if i < 10 else "fly_ball",
            "description": "hit_into_play",
            "events": "single",
            "bat_speed": 71.5,
        }
        for i in range(20)
    ]
    reg = build_batter_regression(pd.DataFrame(rows))
    assert reg.mean_la == pytest.approx(14.0)

    # Swings and misses carry no launch angle and must not drag the mean down.
    whiffs = pd.DataFrame(
        [
            {
                "batter": 1,
                "launch_speed": NAN,
                "launch_angle": NAN,
                "launch_speed_angle": NAN,
                "bb_type": None,
                "description": "swinging_strike",
                "events": "strikeout",
                "bat_speed": 71.5,
            }
        ]
        * 10
    )
    both = build_batter_regression(pd.concat([pd.DataFrame(rows), whiffs]))
    assert both.mean_la == pytest.approx(14.0)


def test_no_launch_angle_column_leaves_the_field_missing_rather_than_zero() -> None:
    # Zero degrees is a real, extreme value; absent data must not masquerade as
    # the most ground-ball-tilted hitter in the league.
    rows = [
        {
            "batter": 1,
            "launch_speed": 90.0,
            "launch_speed_angle": 3,
            "bb_type": "ground_ball",
            "description": "hit_into_play",
            "events": "single",
            "bat_speed": 71.5,
        }
        for _ in range(20)
    ]
    reg = build_batter_regression(pd.DataFrame(rows))
    assert reg.mean_la != reg.mean_la  # NaN
    assert dataclasses.replace(reg, bbe=100).multipliers()["1B"] == pytest.approx(
        dataclasses.replace(reg, bbe=100).multipliers(singles_la_slope=0.0)["1B"]
    )
