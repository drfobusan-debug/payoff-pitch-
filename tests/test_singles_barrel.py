"""Barrel rate as a negative signal on the singles line."""

from __future__ import annotations

import pandas as pd

from mlb_engine.features.regression import (
    BL_BARREL,
    SINGLES_BARREL_SLOPE,
    BatterRegression,
)
from mlb_engine.features.tails import TailAdjuster


def _reg(barrel: float) -> BatterRegression:
    """A league-average batter apart from barrel rate."""
    return BatterRegression(
        bbe=100,
        barrel_rate=barrel,
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
    )


def test_power_hitters_lose_singles_and_the_barrel_rate_stops_there() -> None:
    slugger = _reg(0.16).multipliers()
    slap = _reg(0.02).multipliers()
    assert slugger["1B"] < 1.0 < slap["1B"]
    # The home-run line is not re-read off the same barrels: the rate the
    # simulator holds is already blended toward xHR.
    assert slugger["HR"] == slap["HR"] == 1.0
    # The average batter is untouched on the singles line.
    assert _reg(BL_BARREL).multipliers()["1B"] == 1.0


def test_singles_barrel_term_is_bounded_and_disablable() -> None:
    extreme = _reg(0.40).multipliers()["1B"]
    assert extreme >= 0.94  # the +-6% clip holds however silly the input
    assert _reg(0.40).multipliers(0.0)["1B"] == 1.0  # slope 0 restores old behaviour


def test_slope_prices_only_the_hit_conversion_half() -> None:
    """The raw league slope is ~3.5; half of it is strikeouts the sim already has."""
    assert SINGLES_BARREL_SLOPE == 1.5
    delta = 1.0 - _reg(BL_BARREL + 0.04).multipliers()["1B"]
    assert 0.05 < delta / 0.04 < 2.0  # ~1.5 per unit barrel, before clipping


def _tail_frame() -> pd.DataFrame:
    """40 league-average batters plus one pure-power outlier (id 1)."""
    rng = __import__("numpy").random.default_rng(0)
    rows = []
    for bid in range(1, 41):
        for _ in range(20):
            rows.append(
                {
                    "batter": bid,
                    "pitcher": 1,
                    "launch_speed": float(rng.normal(88, 1.0)),
                    "launch_speed_angle": 3,
                    "estimated_woba_using_speedangle": float(rng.normal(0.32, 0.01)),
                    "description": "hit_into_play",
                    "events": "single",
                }
            )
    for _ in range(20):
        rows.append(
            {
                "batter": 1,
                "pitcher": 1,
                "launch_speed": 108.0,
                "launch_speed_angle": 6,
                "estimated_woba_using_speedangle": 0.32,
                "description": "hit_into_play",
                "events": "home_run",
            }
        )
    return pd.DataFrame(rows)


def test_tail_bonus_no_longer_lifts_singles_on_pure_power() -> None:
    df = _tail_frame()
    split = TailAdjuster.build(df).batter_multiplier(1)
    assert split["HR"] > 1.0
    assert split["1B"] == 1.0  # barrel/hard-hit tails do not touch the singles line

    legacy = TailAdjuster.build(df, power_split=False).batter_multiplier(1)
    assert legacy["1B"] == legacy["HR"] > 1.0
