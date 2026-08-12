"""A bullpen's contact quality is talent, not luck.

The pen covers ~44% of a hitter's plate appearances and was being read through
the regression-to-the-mean corrections built for a starter's ~100 batted balls.
Pooled over a pen's ~1,240 those corrections invert: the pens allowing the most
hits were the ones handed a suppression.
"""

from __future__ import annotations

from dataclasses import replace

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_GB_ALLOWED,
    BL_PEN_XWOBA,
    OPP_GB_SINGLES_SLOPE,
    OPP_GB_XBH_SLOPE,
    PEN_GB_SINGLES_SLOPE,
    PEN_GB_XBH_CLIP,
    PEN_GB_XBH_SLOPE,
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


def test_a_starters_contact_level_does_not_move_his_price_at_all() -> None:
    """No contact-*level* read survives on a starter, in either direction.

    The inverse-BABIP and dxwOBA terms that used to sit here were the pen's
    mistake repeated on a smaller sample: over 2,311 starts neither predicted the
    hits their pitcher went on to allow (t -0.44 and -0.61, both worsening a
    chronological holdout) while together swinging the allowed-hit rate across a
    0.865..1.166 range. A starter's 42-day window holds a median of 95 batted
    balls, where a rate's measurement error is the size of the whole spread
    between pitchers -- so the honest multiplier is exactly 1.0.
    """
    lucky = _reg(0.310, babip=0.245, bullpen=False).allowed_multipliers()
    unlucky = _reg(0.310, babip=0.340, bullpen=False).allowed_multipliers()
    for outcome in ("1B", "2B", "3B", "HR"):
        assert lucky[outcome] == unlucky[outcome]
    # Not merely equal to each other: neutral, so nothing is priced off the level.
    assert lucky["1B"] == 1.0
    assert BL_BABIP == 0.290


def test_a_starter_is_still_read_on_the_trajectory_he_controls() -> None:
    """Dropping the luck terms must not silence the ground-ball term beside them."""
    worm = replace(_reg(0.310, babip=0.290, bullpen=False), gb_allowed=0.520)
    air = replace(_reg(0.310, babip=0.290, bullpen=False), gb_allowed=0.320)
    assert worm.allowed_multipliers()["1B"] > air.allowed_multipliers()["1B"]
    assert worm.allowed_multipliers()["2B"] < air.allowed_multipliers()["2B"]


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


def _gb(gb_allowed: float, *, bullpen: bool) -> dict[str, float]:
    reg = _reg(BL_PEN_XWOBA if bullpen else 0.310, babip=0.290, bullpen=bullpen)
    return replace(reg, gb_allowed=gb_allowed).allowed_multipliers()


def test_a_pen_of_sinkerballers_moves_half_as_far_as_one_starter() -> None:
    """Averaging ~24 arms halves the spread, and the slope with it."""
    worm = _gb(0.500, bullpen=True)
    air = _gb(0.369, bullpen=True)
    pen_span = worm["2B"] / air["2B"]
    sp_span = _gb(0.500, bullpen=False)["2B"] / _gb(0.369, bullpen=False)["2B"]
    assert worm["2B"] < air["2B"]  # grounders still cost extra bases
    assert worm["1B"] > air["1B"]  # and still add singles
    assert pen_span > sp_span  # both suppress, the pen by less
    assert 0.40 < (1 - pen_span) / (1 - sp_span) < 0.60


def test_the_pen_extra_base_term_is_bounded_tighter_than_the_starter() -> None:
    assert PEN_GB_XBH_CLIP == (-0.07, 0.07)
    assert abs(PEN_GB_XBH_SLOPE) < abs(OPP_GB_XBH_SLOPE)
    assert PEN_GB_SINGLES_SLOPE < OPP_GB_SINGLES_SLOPE
    extreme = _gb(0.90, bullpen=True)
    assert extreme["2B"] / _gb(BL_GB_ALLOWED, bullpen=True)["2B"] > 0.92


def test_the_switch_turns_the_pen_reading_off(monkeypatch) -> None:
    monkeypatch.delenv("MLBE_PEN_CONTACT_LEVEL", raising=False)
    assert Config().pen_contact_level is True
    monkeypatch.setenv("MLBE_PEN_CONTACT_LEVEL", "0")
    assert Config().pen_contact_level is False
