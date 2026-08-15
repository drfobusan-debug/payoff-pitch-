"""The opposing starter's home-run profile: where the ball goes, not just how hard.

The HR line already read the contact quality a starter allows (barrels, hard
hit). It could not see *direction* -- the sinkerballer who keeps the ball in the
dirt and the fly-ball arm whose air contact is also hard -- nor which side of the
plate his home-run risk actually lives on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_engine.features.pitch_mix import (
    CLASSES,
    LEAGUE_SWSTR,
    LEAGUE_WHIFF,
    LEAGUE_XWOBA,
    pitch_class,
)
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_GB_ALLOWED,
    BL_HARD_HIT,
    BL_XBA,
    PitcherRegression,
    build_pitcher_regression,
)


def _pitcher(**kw: float) -> PitcherRegression:
    base = dict(
        bbe=250,
        pitches=1500,
        babip_allowed=BL_BABIP,
        woba_allowed=BL_XBA,
        xwoba_allowed=BL_XBA,
        hard_hit_allowed=BL_HARD_HIT,
        barrel_allowed=0.080,
        csw=0.28,
        k_pct=0.22,
        bb_pct=0.08,
        two_strike_whiff=0.28,
    )
    base.update(kw)
    return PitcherRegression(**base)  # type: ignore[arg-type]


def _hr(**kw: float) -> float:
    return _pitcher(**kw).allowed_multipliers()["HR"]


# --- ground balls: the deep negative modifier --------------------------------


def test_a_ground_ball_starter_suppresses_the_home_run() -> None:
    """You cannot hit one on the ground, however hard you swing."""
    assert _hr(gb_allowed=0.58, fb_allowed=0.26) < 0.85
    assert _hr(gb_allowed=0.63, fb_allowed=0.22) < _hr(gb_allowed=0.58, fb_allowed=0.26)


def test_the_ground_ball_brake_is_dormant_below_the_ceiling() -> None:
    """A merely above-average ground-ball rate is not a profile."""
    assert _hr(gb_allowed=0.47) == _hr(gb_allowed=0.42)
    assert _hr() == 1.0  # league-average arm is untouched


def test_the_brake_survives_an_otherwise_hittable_starter() -> None:
    """The point of the modifier: it suppresses even against real damage."""
    hittable = dict(barrel_allowed=0.105, hard_hit_allowed=0.44)
    assert _hr(**hittable) > 1.08
    assert _hr(gb_allowed=0.60, fb_allowed=0.24, **hittable) < _hr(**hittable)


# --- fly balls: only a liability when they are hit hard ----------------------


def test_fly_balls_alone_are_not_a_home_run_problem() -> None:
    """A starter can give up all the fly balls he likes if they are hit softly."""
    soft = _hr(gb_allowed=0.33, fb_allowed=0.48, hard_hit_allowed=0.355)
    assert soft < 1.0  # the soft contact itself, not the fly balls


def test_fly_balls_plus_hard_contact_is_a_strong_positive() -> None:
    hard = dict(hard_hit_allowed=0.455, barrel_allowed=0.105)
    assert _hr(gb_allowed=0.33, fb_allowed=0.48, **hard) > _hr(**hard)


def test_the_fly_ball_term_needs_both_halves() -> None:
    """It is a product of two excesses, so either at baseline means no credit."""
    assert _hr(fb_allowed=0.48) == _hr(fb_allowed=0.36)  # air but not hard
    assert _hr(hard_hit_allowed=0.455, fb_allowed=0.36) == _hr(
        hard_hit_allowed=0.455, fb_allowed=0.30
    )  # hard but not in the air


# --- ride on the four-seamer -------------------------------------------------


def test_a_high_ride_fastball_costs_more_than_a_heavy_one() -> None:
    assert _hr(ivb=20.0) > _hr(ivb=15.0) > _hr(ivb=11.0)
    assert _hr(ivb=float("nan")) == _hr(ivb=15.0)  # unknown -> baseline


def test_ivb_is_measured_on_four_seamers_not_the_whole_arsenal() -> None:
    """Breaking balls carry negative break; averaging everything measured mix."""
    rows = [{"pitch_type": "FF", "pfx_z": 1.4} for _ in range(60)]
    rows += [{"pitch_type": "CU", "pfx_z": -0.8} for _ in range(60)]
    pdf = pd.DataFrame(rows).assign(
        launch_speed=None, events=None, description=None,
        release_extension=None, release_pos_x=None, release_pos_z=None,
        release_spin_rate=None,
    )
    reg = build_pitcher_regression(pdf)
    assert abs(reg.ivb - 16.8) < 0.1  # the four-seamers only (1.4ft x 12)


def test_a_thin_four_seam_sample_reports_no_ride() -> None:
    pdf = pd.DataFrame({"pitch_type": ["FF"] * 5, "pfx_z": [1.4] * 5}).assign(
        launch_speed=None, events=None, description=None,
        release_extension=None, release_pos_x=None, release_pos_z=None,
        release_spin_rate=None,
    )
    reg = build_pitcher_regression(pdf)
    assert reg.ivb != reg.ivb  # NaN


# --- platoon: home-run risk lives on one side of the plate -------------------


def test_a_reverse_split_starter_stops_looking_ordinary() -> None:
    reg = _pitcher(
        barrel_allowed=0.080,
        hard_hit_allowed=0.400,
        barrel_allowed_vs_l=0.115,
        hard_hit_allowed_vs_l=0.440,
        barrel_allowed_vs_r=0.055,
        hard_hit_allowed_vs_r=0.370,
    )
    assert reg.platoon_power_multiplier("L") > 1.15
    assert reg.platoon_power_multiplier("R") < 0.90
    # ...while his overall rate is exactly league average.
    assert abs(reg.allowed_multipliers()["HR"] - 1.0) < 1e-9


def test_the_power_split_is_independent_of_the_strikeout_split() -> None:
    """Missing bats and suppressing contact quality are different skills."""
    reg = _pitcher(
        k_pct=0.22,
        k_pct_vs_l=0.22,
        k_pct_vs_r=0.22,  # no K platoon split at all
        barrel_allowed_vs_l=0.120,
        hard_hit_allowed_vs_l=0.450,
    )
    assert reg.platoon_k_multiplier("L") == 1.0
    assert reg.platoon_power_multiplier("L") > 1.10


def test_the_power_split_is_neutral_without_a_split_sample() -> None:
    reg = _pitcher()
    for bats in ("L", "R", "S", None):
        assert reg.platoon_power_multiplier(bats) == 1.0


def test_platoon_splits_need_enough_batted_balls() -> None:
    rows = []
    for hand, n in (("L", 10), ("R", 200)):  # LHB sample below the floor
        rows += [
            # ``type``/``description`` mark these as balls in play: exit velocity
            # alone is also recorded on foul balls, which are not batted balls.
            {
                "stand": hand,
                "launch_speed": 100.0,
                "launch_speed_angle": 6,
                "type": "X",
                "description": "hit_into_play",
            }
            for _ in range(n)
        ]
    pdf = pd.DataFrame(rows).assign(
        events=None, pitch_type=None, pfx_z=None,
        release_extension=None, release_pos_x=None, release_pos_z=None,
        release_spin_rate=None, bb_type=None,
    )
    reg = build_pitcher_regression(pdf)
    assert reg.barrel_allowed_vs_l != reg.barrel_allowed_vs_l  # NaN, too thin
    assert reg.barrel_allowed_vs_r == 1.0


# --- pitch classification ----------------------------------------------------


def test_the_sinker_is_no_longer_a_four_seamer() -> None:
    """The distinction the whole shape argument rests on."""
    assert pitch_class("FF") == "FB"
    assert pitch_class("SI") == "SNK"
    assert pitch_class("FT") == "SNK"
    assert pitch_class("FC") == "FB"  # the cutter rides, it is not a sinker
    assert pitch_class("SL") == "BRK"
    assert pitch_class(None) is None


def test_every_class_has_a_league_baseline() -> None:
    for cls in CLASSES:
        assert cls in LEAGUE_WHIFF
        assert cls in LEAGUE_SWSTR
        assert cls in LEAGUE_XWOBA
    # A sinker is hit for less damage than a four-seamer, which is the point.
    assert LEAGUE_XWOBA["SNK"] < LEAGUE_XWOBA["FB"]
    assert LEAGUE_SWSTR["SNK"] < LEAGUE_SWSTR["FB"]


# --- the whole multiplier stays bounded --------------------------------------


def test_the_allowed_hr_multiplier_stays_bounded() -> None:
    worst = _hr(
        gb_allowed=0.20, fb_allowed=0.60, hard_hit_allowed=0.55,
        barrel_allowed=0.18, ivb=24.0, babip_allowed=0.240,
    )
    best = _hr(
        gb_allowed=0.70, fb_allowed=0.15, hard_hit_allowed=0.28,
        barrel_allowed=0.02, ivb=6.0, babip_allowed=0.360,
    )
    assert 0.78 <= best < 1.0 < worst <= 1.35


def test_a_thin_batted_ball_sample_yields_no_multipliers() -> None:
    assert _pitcher(bbe=3).allowed_multipliers() == {}


# --- ground balls on the singles line ----------------------------------------


def _allowed(**kw: float) -> dict[str, float]:
    return _pitcher(**kw).allowed_multipliers()


def test_a_sinkerballer_concedes_singles_while_suppressing_the_rest() -> None:
    """The same grounder that cannot clear a fence very often falls in.

    Split-half reliability puts GB% allowed at .658 against BABIP allowed's
    .126, so the trajectory is the part of his contact profile a starter
    actually repeats.
    """
    worm = _allowed(gb_allowed=0.56)
    flyball = _allowed(gb_allowed=0.30)
    assert worm["1B"] > flyball["1B"]
    assert worm["HR"] < flyball["HR"]


def test_grounders_move_the_two_channels_in_opposite_directions() -> None:
    """A grounder is a single *instead of* an extra-base hit.

    The singles channel gains what the extra-base channel loses, so the term
    must not carry the same sign into 2B/3B -- that would assert the opposite
    of what it means.
    """
    worm = _allowed(gb_allowed=0.56)
    flyball = _allowed(gb_allowed=0.30)
    for key in ("2B", "3B"):
        assert worm[key] < flyball[key]
    assert worm["1B"] > flyball["1B"]


def test_the_extra_base_ground_ball_term_is_bounded() -> None:
    # An extreme sinkerballer is not allowed to erase the doubles line: the
    # clip binds outside roughly the 5th and 95th percentile of GB% allowed.
    extreme_worm = _allowed(gb_allowed=0.70)["2B"]
    extreme_air = _allowed(gb_allowed=0.20)["2B"]
    neutral = _allowed(gb_allowed=0.42)["2B"]
    assert extreme_worm == pytest.approx(0.86 * neutral)
    assert extreme_air == pytest.approx(1.14 * neutral)
    assert extreme_worm < neutral < extreme_air


def test_a_league_average_ground_ball_rate_is_neutral_on_singles() -> None:
    # An otherwise-neutral starter -- league BABIP allowed, no dxwOBA gap -- must
    # come out at exactly 1.0, so the term charges nothing for being ordinary.
    assert _allowed(gb_allowed=BL_GB_ALLOWED)["1B"] == pytest.approx(1.0, abs=1e-9)


def test_the_singles_ground_ball_term_is_bounded() -> None:
    assert _allowed(gb_allowed=0.75)["1B"] <= 1.14 * 1.035
    assert _allowed(gb_allowed=0.15)["1B"] >= 0.88 * 0.965
