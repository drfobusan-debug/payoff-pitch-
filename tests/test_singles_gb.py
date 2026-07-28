"""Ground-ball rate on the singles line (off by default)."""

from __future__ import annotations

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_GB_RATE,
    SINGLES_GB_SLOPE,
    BatterRegression,
    build_batter_regression,
)


def _reg(gb: float) -> BatterRegression:
    """A league-average batter apart from ground-ball rate."""
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
        gb_rate=gb,
    )


def test_gb_term_is_off_unless_asked_for() -> None:
    worm_burner = _reg(0.60)
    assert worm_burner.multipliers()["1B"] == 1.0
    assert Config().singles_gb is False
    assert worm_burner.multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] > 1.0


def test_grounders_help_singles_and_nothing_else() -> None:
    ground = _reg(0.60).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)
    air = _reg(0.25).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)
    assert ground["1B"] > 1.0 > air["1B"]
    assert _reg(BL_GB_RATE).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] == 1.0
    # The batted-ball mix belongs to the singles line alone.
    for key in ("2B", "3B", "HR"):
        assert ground[key] == air[key]


def test_gb_term_is_bounded() -> None:
    assert _reg(1.0).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] <= 1.06
    assert _reg(0.0).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] >= 0.94


def test_gb_rate_is_read_off_batted_ball_type() -> None:
    rows = [
        {
            "batter": 1,
            "launch_speed": 90.0,
            "launch_angle": 5.0,
            "launch_speed_angle": 3,
            "bb_type": "ground_ball" if i < 15 else "fly_ball",
            "description": "hit_into_play",
            "events": "single",
            "bat_speed": 71.5,
        }
        for i in range(20)
    ]
    reg = build_batter_regression(pd.DataFrame(rows))
    assert reg.gb_rate == 0.75

    # No bb_type column at all falls back to the league baseline, not zero.
    bare = pd.DataFrame(rows).drop(columns=["bb_type"])
    assert build_batter_regression(bare).gb_rate == BL_GB_RATE
