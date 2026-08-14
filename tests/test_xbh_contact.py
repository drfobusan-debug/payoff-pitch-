"""Contact quality does not move the extra-base rate.

Fitted out of time over 48,120 plate appearances, no contact measure the engine
reads -- sweet-spot rate, xSLG, bat speed, hard-hit rate, xwOBA on contact --
forward-predicts a double or a triple. Only the opposing arm's ground-ball rate
does, and that lives in ``allowed_multipliers``.
"""

from __future__ import annotations

import pytest

from mlb_engine.features.regression import BL_SPRINT, BatterRegression


def _reg(sweet: float, xslg: float, bat_speed: float) -> BatterRegression:
    """A league-average batter apart from his contact quality."""
    return BatterRegression(
        bbe=100,
        barrel_rate=0.080,
        hard_hit=0.400,
        sweet_spot=sweet,
        bat_speed=bat_speed,
        max_ev=108.0,
        whiff=0.240,
        zone_contact=0.820,
        xba=0.250,
        xslg=xslg,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
        sprint_speed=BL_SPRINT,
    )


def test_contact_quality_does_not_move_doubles() -> None:
    elite = _reg(0.420, 0.520, 76.0).multipliers()
    poor = _reg(0.240, 0.300, 67.0).multipliers()
    for outcome in ("2B", "3B"):
        assert elite[outcome] == pytest.approx(poor[outcome])
    # The same contact still separates them on the home-run line, which is where
    # it was measured to belong.
    assert elite["HR"] > poor["HR"]


def test_speed_still_moves_doubles() -> None:
    fast = _reg(0.330, 0.400, 71.5)
    fast.sprint_speed = BL_SPRINT + 2.0
    slow = _reg(0.330, 0.400, 71.5)
    slow.sprint_speed = BL_SPRINT - 2.0
    assert fast.multipliers()["2B"] > slow.multipliers()["2B"]
