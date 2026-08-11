"""A bullpen's contact quality is talent, not luck.

The pen covers ~44% of a hitter's plate appearances and was being read through
the regression-to-the-mean corrections built for a starter's ~100 batted balls.
Pooled over a pen's ~1,240 those corrections invert: the pens allowing the most
hits were the ones handed a suppression.
"""

from __future__ import annotations

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_PEN_XWOBA,
    PitcherRegression,
)


def _reg(xwoba: float, babip: float, *, bullpen: bool) -> PitcherRegression:
    return PitcherRegression(
        bbe=1240 if bullpen else 100,
        pitches=4000 if bullpen else 700,
        babip_allowed=babip,
        woba_allowed=xwoba,  # no luck gap, so dxwOBA cannot muddy the comparison
        xwoba_allowed=xwoba,
        hard_hit_allowed=0.378,
        barrel_allowed=0.070,
        csw=0.28,
        k_pct=0.22,
        bb_pct=0.08,
        two_strike_whiff=0.30,
        bullpen=bullpen,
    )


def test_a_pen_that_gets_hit_hard_is_priced_up_not_down() -> None:
    soft = _reg(0.282, babip=0.245, bullpen=True).allowed_multipliers()
    hard = _reg(0.345, babip=0.315, bullpen=True).allowed_multipliers()
    for outcome in ("1B", "2B", "3B", "HR"):
        assert hard[outcome] > soft[outcome]
    # The old luck reading had these the wrong way round: the .315-BABIP pen
    # allowed the most hits (.2245/PA) and was suppressed to 0.989.
    old_soft = _reg(0.282, babip=0.245, bullpen=False).allowed_multipliers()
    old_hard = _reg(0.345, babip=0.315, bullpen=False).allowed_multipliers()
    assert old_hard["1B"] < old_soft["1B"]


def test_the_starter_path_is_untouched() -> None:
    """Luck corrections stay where the sample is genuinely small."""
    lucky = _reg(0.310, babip=0.245, bullpen=False).allowed_multipliers()
    unlucky = _reg(0.310, babip=0.340, bullpen=False).allowed_multipliers()
    # A starter with a low BABIP allowed is due to give hits back.
    assert lucky["1B"] > unlucky["1B"]
    assert BL_BABIP == 0.290


def test_the_pen_term_is_bounded() -> None:
    extreme = _reg(0.500, babip=0.290, bullpen=True).allowed_multipliers()
    floor = _reg(0.150, babip=0.290, bullpen=True).allowed_multipliers()
    # +/- 6% before the outcome-specific terms, which bind well inside the
    # observed 0.288..0.340 spread of pen xwOBA allowed.
    assert extreme["1B"] / floor["1B"] < 1.30
    neutral = _reg(BL_PEN_XWOBA, babip=0.290, bullpen=True).allowed_multipliers()
    assert abs(neutral["1B"] - 1.0) < 0.05


def test_a_pen_is_only_a_pen_when_it_is_labelled_one() -> None:
    assert PitcherRegression(
        bbe=1240, pitches=4000, babip_allowed=0.29, woba_allowed=0.31,
        xwoba_allowed=0.31, hard_hit_allowed=0.378, barrel_allowed=0.07,
        csw=0.28, k_pct=0.22, bb_pct=0.08, two_strike_whiff=0.30,
    ).bullpen is False


def test_the_switch_turns_the_pen_reading_off(monkeypatch) -> None:
    monkeypatch.delenv("MLBE_PEN_CONTACT_LEVEL", raising=False)
    assert Config().pen_contact_level is True
    monkeypatch.setenv("MLBE_PEN_CONTACT_LEVEL", "0")
    assert Config().pen_contact_level is False
