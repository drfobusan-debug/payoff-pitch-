"""Unit tests for the V1-style RBI/XBH/TB selectors."""

from __future__ import annotations

import pytest

from mlb_engine.data.parks import Park
from mlb_engine.features.regression import BatterRegression
from mlb_engine.models.rbi_rule import RBIFlag
from mlb_engine.models.selectors import RBISelector, TBSelector, XBHSelector


@pytest.fixture
def elite_batter() -> BatterRegression:
    return BatterRegression(
        bbe=50,
        barrel_rate=0.12,
        hard_hit=0.45,
        sweet_spot=0.35,
        bat_speed=75.0,
        max_ev=112.0,
        whiff=0.25,
        zone_contact=0.85,
        xba=0.280,
        xslg=0.500,
        babip=0.300,
        woba=0.350,
        xwoba=0.360,
        sprint_speed=28.0,
    )


@pytest.fixture
def weak_batter() -> BatterRegression:
    return BatterRegression(
        bbe=50,
        barrel_rate=0.02,
        hard_hit=0.25,
        sweet_spot=0.28,
        bat_speed=68.0,
        max_ev=104.0,
        whiff=0.35,
        zone_contact=0.70,
        xba=0.220,
        xslg=0.280,
        babip=0.260,
        woba=0.280,
        xwoba=0.290,
        sprint_speed=25.0,
    )


def test_xbh_selector_buy(elite_batter: BatterRegression) -> None:
    sel = XBHSelector().select(elite_batter)
    assert sel.signal == "buy"
    assert sel.factor > 1.05
    assert sel.outcome_multipliers.get("2B", 1.0) > 1.05
    assert sel.outcome_multipliers.get("3B", 1.0) > 1.05


def test_xbh_selector_sell(weak_batter: BatterRegression) -> None:
    sel = XBHSelector().select(weak_batter)
    assert sel.factor < 0.97
    assert sel.signal == "sell"


def test_xbh_selector_neutral_for_missing_data() -> None:
    sel = XBHSelector().select(None)
    assert sel.signal == "none"
    assert sel.factor == 1.0


def test_tb_selector_buy(elite_batter: BatterRegression) -> None:
    sel = TBSelector().select(elite_batter)
    assert sel.signal == "buy"
    assert sel.factor > 1.05
    assert sel.post_multipliers.get("TB", 1.0) > 1.05


def test_tb_selector_sell(weak_batter: BatterRegression) -> None:
    sel = TBSelector().select(weak_batter)
    assert sel.factor < 0.97
    assert sel.signal == "sell"


def test_tb_selector_max_ev_drives_factor() -> None:
    """Two batters identical but for max EV -> the higher-max-EV bat gets the
    higher TB factor (max EV is the significant separator, now up-weighted)."""
    base = dict(
        bbe=50,
        barrel_rate=0.08,
        hard_hit=0.30,
        sweet_spot=0.31,
        bat_speed=70.0,
        whiff=0.30,
        zone_contact=0.78,
        xba=0.250,
        xslg=0.400,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
        sprint_speed=27.0,
    )
    low = TBSelector().select(BatterRegression(max_ev=105.0, **base))
    high = TBSelector().select(BatterRegression(max_ev=114.0, **base))
    assert high.factor > low.factor
    # The 9 mph gap should move the factor by a meaningful, not negligible, amount.
    assert high.factor - low.factor > 0.03


def test_rbi_selector_flagged_buy() -> None:
    flag = RBIFlag(
        slot=3,
        preceding_obp=0.360,
        flagged=True,
        xslg=0.500,
        zone_contact=0.85,
        bbe=50,
    )
    sel = RBISelector().select(flag)
    assert sel.factor > 1.0
    assert sel.post_multipliers.get("RBI", 1.0) > 1.0


def test_rbi_selector_not_flagged_is_neutral() -> None:
    flag = RBIFlag(
        slot=3,
        preceding_obp=0.360,
        flagged=False,
        xslg=0.500,
        zone_contact=0.85,
        bbe=50,
    )
    sel = RBISelector().select(flag)
    assert sel.factor == 1.0
    assert sel.signal == "none"


def test_park_and_weather_are_applied(elite_batter: BatterRegression) -> None:
    park = Park(
        venue_id=17,
        name="Wrigley Field",
        lat=41.9484,
        lon=-87.6553,
        orientation_deg=30.0,
        roof="open",
        park_factor=103.0,
        wind_factor=1.20,
        carry_factor=1.05,
    )
    weather = {"1B": 1.05, "2B": 1.05, "3B": 1.05, "HR": 1.10}
    xbh = XBHSelector().select(elite_batter, park=park, weather=weather)
    tb = TBSelector().select(elite_batter, park=park, weather=weather)
    assert xbh.factor > 1.0
    assert tb.factor > 1.0
