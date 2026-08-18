"""The pitch-shape grade, and the strikeout prior that now reads it.

The grade is the one strikeout signal in the engine that is not a result: it
scores the physics of each pitch against how often a pitch shaped like that is
missed, relative to the league's *same pitch type*. That last clause is what
these tests mostly pin, because the study says arsenal composition is worth
nothing and a grade that rewarded throwing sliders would be measuring it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from mlb_engine.calibration import FEATURE_BASIS, FEATURE_BASIS_SINCE
from mlb_engine.features import stuff
from mlb_engine.features.regression import (
    BL_CSW,
    BL_SWSTR,
    XK_SHAPE_COEF,
    PitcherRegression,
    build_pitcher_regression,
)


def _pitches(
    n: int,
    pitch_type: str = "FF",
    velo: float = 93.0,
    ivb_in: float = 15.0,
    spin: float = 2300.0,
) -> pd.DataFrame:
    """A window of one pitch, repeated: enough to grade, nothing else moving.

    Outcomes cycle so CSW% and SwStr% land near league -- the point is to vary the
    *shape* while the results terms of the prior stay put.
    """
    cycle = ["called_strike", "swinging_strike", "foul", "ball", "hit_into_play"]
    return pd.DataFrame(
        {
            "game_date": ["2026-07-20"] * n,
            "pitch_type": [pitch_type] * n,
            "p_throws": ["R"] * n,
            "release_speed": [velo] * n,
            "pfx_z": [ivb_in / 12.0] * n,
            "release_spin_rate": [spin] * n,
            "release_extension": [6.4] * n,
            "release_pos_z": [5.9] * n,
            "release_pos_x": [-1.8] * n,
            "description": [cycle[i % len(cycle)] for i in range(n)],
            "events": ["strikeout" if i % 5 == 1 else None for i in range(n)],
            "woba_denom": [None] * n,
            "woba_value": [None] * n,
            "bb_type": [None] * n,
            "launch_speed": [None] * n,
            "stand": ["R"] * n,
            "balls": [0] * n,
            "strikes": [0] * n,
            "type": ["S"] * n,
        }
    )


def _reg(**kw: float) -> PitcherRegression:
    base: dict[str, float] = dict(
        bbe=200,
        pitches=1500,
        babip_allowed=0.290,
        woba_allowed=0.320,
        xwoba_allowed=0.320,
        hard_hit_allowed=0.380,
        barrel_allowed=0.070,
        csw=BL_CSW,
        k_pct=0.22,
        bb_pct=0.08,
        two_strike_whiff=0.28,
        swstr=BL_SWSTR,
    )
    base.update(kw)
    return PitcherRegression(**base)  # type: ignore[arg-type]


# --- the grade ---------------------------------------------------------------


def test_a_thin_window_gets_no_opinion() -> None:
    """Below the floor the grade is exactly zero, not a small number."""
    assert stuff.shape_plus(_pitches(stuff.MIN_PITCHES - 1)) == 0.0
    assert stuff.shape_plus(_pitches(stuff.MIN_PITCHES)) != 0.0


def test_a_frame_without_shape_columns_gets_no_opinion() -> None:
    bare = _pitches(400).drop(columns=["release_spin_rate"])
    assert stuff.shape_plus(bare) == 0.0


def test_velocity_and_ride_grade_a_fastball_up() -> None:
    slow = stuff.shape_plus(_pitches(400, velo=89.0))
    fast = stuff.shape_plus(_pitches(400, velo=99.0))
    flat = stuff.shape_plus(_pitches(400, ivb_in=9.0))
    ride = stuff.shape_plus(_pitches(400, ivb_in=19.0))
    assert fast > slow
    assert ride > flat


def test_the_grade_is_against_the_same_pitch_type_not_the_league() -> None:
    """A sinker whiffs far less than a slider, and neither is thereby graded down.

    Every pitch type's own average pitch scores ~0, which is the only way arsenal
    composition can be worth nothing: otherwise a sinkerballer would carry a
    negative grade for throwing the pitch he throws, and the study says usage
    predicts nothing (its half fits with the wrong sign).
    """
    for group, model in stuff._GROUPS.items():
        row = pd.DataFrame([dict(zip(stuff._FEATURES, model.mean, strict=True))])
        earned = float(stuff._whiff_rates(model, row)[0]) - model.base_whiff
        assert abs(earned) < 0.02, (group, earned)


# --- the prior that reads it -------------------------------------------------


def test_the_prior_moves_with_the_grade_and_only_by_the_fitted_slope() -> None:
    league = _reg(shape_plus=0.0).expected_k_pct()
    good = _reg(shape_plus=0.02).expected_k_pct()
    bad = _reg(shape_plus=-0.02).expected_k_pct()
    assert good > league > bad
    assert abs((good - league) - XK_SHAPE_COEF * 0.02) < 1e-9
    assert abs((league - bad) - XK_SHAPE_COEF * 0.02) < 1e-9


def test_an_absurd_grade_is_clipped_rather_than_priced() -> None:
    capped = _reg(shape_plus=0.40).expected_k_pct()
    at_clip = _reg(shape_plus=stuff.GRADE_CLIP).expected_k_pct()
    assert capped == at_clip


def test_a_pitcher_with_no_grade_prices_exactly_as_before() -> None:
    """The term is additive on the grade, so 0.0 leaves the shipped line alone."""
    reg = _reg(shape_plus=0.0)
    assert abs(reg.expected_k_pct() - 0.220) < 1e-9


def test_pricing_on_the_grade_means_a_new_calibration_basis() -> None:
    """A live grade and the previous basis string cannot both be true.

    The term shipped without bumping ``FEATURE_BASIS``, which a live board caught:
    it moves 3,906 of 4,023 priced rows, so a map refit on the previous engine
    would have matched the basis and been applied silently to prices it was never
    fitted on. Pinned here rather than in the calibration tests because it is the
    grade being priced that invalidates the old map.
    """
    assert XK_SHAPE_COEF > 0.0
    assert FEATURE_BASIS != "no-stuff-multiplier-2026.08"
    assert FEATURE_BASIS_SINCE >= dt.date(2026, 8, 18)


def test_the_window_is_graded_end_to_end() -> None:
    """``build_pitcher_regression`` fills the grade in from the pitch frame."""
    hard = build_pitcher_regression(_pitches(400, velo=99.0))
    soft = build_pitcher_regression(_pitches(400, velo=89.0))
    assert hard.shape_plus > soft.shape_plus
    assert hard.expected_k_pct() > soft.expected_k_pct()
    # and a window too thin to grade leaves the prior where it was
    thin = build_pitcher_regression(_pitches(20, velo=99.0))
    assert thin.shape_plus == 0.0
