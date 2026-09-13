"""The home-run contact read: barrel rate allowed, or where the ball goes.

Barrel% allowed has driven the HR multiplier since it was written and does not
forecast -- added to the next start's home runs it fits at t=+3.3 and moves
held-out deviance the wrong way. Fly-ball rate over his last four starts does
(-.00033). The crossfade is off by default, so the shipped price is unchanged
until the ledger has graded it (``scripts.hr_contact_study``).
"""

from __future__ import annotations

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_BARREL_ALLOWED,
    BL_FB_HR,
    MIN_FB_HR_BBE,
    PitcherRegression,
    build_pitcher_regression,
)


def _rows(days: dict[str, tuple[str, int]], barrel: float = BL_BARREL_ALLOWED):
    """One frame of batted balls: ``{game_date: (bb_type, count)}``."""
    out = []
    for day, (bb_type, n) in days.items():
        for i in range(n):
            out.append(
                {
                    "game_date": day,
                    "pitch_type": "FF",
                    "release_speed": 94.0,
                    "description": "hit_into_play",
                    "events": "field_out",
                    "woba_denom": 1.0,
                    "woba_value": 0.0,
                    "estimated_woba_using_speedangle": 0.320,
                    "bb_type": bb_type,
                    "launch_speed": 90.0,
                    "launch_speed_angle": 6 if i / max(n, 1) < barrel else 3,
                    "pfx_z": None,
                    "release_extension": None,
                    "release_pos_x": None,
                    "release_pos_z": None,
                    "release_spin_rate": None,
                }
            )
    return pd.DataFrame(out)


def _reg(days: dict[str, tuple[str, int]], hr_flyball: float) -> PitcherRegression:
    return build_pitcher_regression(_rows(days), hr_flyball=hr_flyball)


def _flat(rate: float, hr_flyball: float) -> PitcherRegression:
    """A regression with the recent fly-ball rate set directly."""
    return PitcherRegression(
        bbe=200,
        pitches=1200,
        babip_allowed=0.290,
        woba_allowed=0.320,
        xwoba_allowed=0.320,
        hard_hit_allowed=0.380,
        barrel_allowed=BL_BARREL_ALLOWED,
        csw=0.280,
        k_pct=0.220,
        bb_pct=0.075,
        two_strike_whiff=0.300,
        fb_allowed_recent=rate,
        hr_flyball=hr_flyball,
    )


def test_the_flyball_read_is_off_until_it_is_switched_on() -> None:
    assert Config().windows.hr_flyball_weight == 0.0
    off = _flat(0.50, 0.0)
    assert off.hr_contact_multiplier() == 1.0  # barrel sits at its baseline
    assert _flat(0.50, 1.0).hr_contact_multiplier() > 1.0


def test_a_flyball_arm_allows_more_home_runs_and_a_sinkerballer_fewer() -> None:
    lofty = _flat(BL_FB_HR + 0.10, 1.0).hr_contact_multiplier()
    grounder = _flat(BL_FB_HR - 0.10, 1.0).hr_contact_multiplier()

    assert lofty > 1.0 > grounder
    assert round(lofty - 1.0, 4) == round(1.0 - grounder, 4)  # two-sided


def test_barrel_keeps_its_say_at_partial_weight() -> None:
    """Half weight is half of each read, so the term can be graded in stages."""
    barrelled = 0.140  # a long way above the .080 baseline
    full_barrel = _flat(BL_FB_HR, 0.0)
    full_barrel.barrel_allowed = barrelled
    half = _flat(BL_FB_HR, 0.5)
    half.barrel_allowed = barrelled

    assert full_barrel.hr_contact_multiplier() > half.hr_contact_multiplier() > 1.0


def test_a_thin_recent_sample_falls_back_to_neutral_not_to_barrel() -> None:
    """Barrel is not the fallback: it forecast worse than nothing."""
    thin = _flat(float("nan"), 1.0)
    thin.barrel_allowed = 0.140

    assert thin.hr_contact_multiplier() == 1.0


def test_the_read_is_his_last_four_starts_not_his_whole_window() -> None:
    """Six of them: four fly-ball outings after two on the ground."""
    days = {
        "2026-06-01": ("ground_ball", 40),
        "2026-06-07": ("ground_ball", 40),
        "2026-07-01": ("fly_ball", 40),
        "2026-07-07": ("fly_ball", 40),
        "2026-07-13": ("fly_ball", 40),
        "2026-07-19": ("fly_ball", 40),
    }
    reg = _reg(days, 1.0)

    assert reg.fb_allowed_recent == 1.0  # the grounders are outside the window
    assert reg.fb_allowed > 0.60  # the whole-window rate still sees them


def test_too_few_recent_batted_balls_is_no_read_at_all() -> None:
    days = {"2026-07-19": ("fly_ball", MIN_FB_HR_BBE - 1)}
    reg = _reg(days, 1.0)

    assert reg.fb_allowed_recent != reg.fb_allowed_recent  # NaN
    assert reg.hr_contact_multiplier() == 1.0


def test_a_bullpen_is_left_alone() -> None:
    """The term was fitted on starts; a pen's four 'starts' are not outings."""
    pen = _flat(BL_FB_HR + 0.12, 1.0)
    pen.bullpen = True

    assert pen.hr_contact_multiplier() == 1.0


def test_only_the_home_run_rate_moves() -> None:
    off = _flat(BL_FB_HR + 0.12, 0.0).allowed_multipliers()
    on = _flat(BL_FB_HR + 0.12, 1.0).allowed_multipliers()

    assert on["HR"] > off["HR"]
    for key in ("1B", "2B", "3B"):
        assert on[key] == off[key]
