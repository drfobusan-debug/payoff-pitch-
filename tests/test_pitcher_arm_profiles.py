"""The pitcher regression card carries the delivery beside the luck term."""

from __future__ import annotations

import math
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from mlb_engine.features import arm
from mlb_engine.output import regression_profiles as rp

DAY0 = Date(2026, 7, 1)


def _slice(n: int, *, velo: float = 96.0, tracked: bool = True) -> pd.DataFrame:
    """``n`` fastballs put in play, with or without the release columns."""
    frame = pd.DataFrame(
        {
            "pitcher": [7] * n,
            "batter": range(n),
            "p_throws": ["R"] * n,
            "pitch_type": ["FF"] * n,
            "game_date": [DAY0 + timedelta(days=i % 40) for i in range(n)],
            "description": ["hit_into_play"] * n,
            "type": ["X"] * n,
            "events": ["single"] * n,
            "launch_speed": [98.0] * n,
            "launch_angle": [14.0] * n,
            "launch_speed_angle": [4] * n,
            "estimated_ba_using_speedangle": [0.400] * n,
            "estimated_woba_using_speedangle": [0.420] * n,
            "woba_value": [0.9] * n,
            "woba_denom": [1] * n,
            "zone": [5] * n,
        }
    )
    if not tracked:
        return frame
    return frame.assign(
        release_speed=velo,
        release_extension=6.6,
        release_pos_x=-1.9,
        release_pos_z=5.9,
        release_spin_rate=2300.0,
        pfx_z=1.25,
        pfx_x=-0.85,
    )


def test_the_card_prints_the_delivery_and_its_verdict() -> None:
    prof = rp.analyze("Hunter Greene", 7, _slice(120, velo=98.0), DAY0 + timedelta(days=20))

    assert prof["arm_pitches"] == 120
    assert math.isclose(prof["arm_velo"], 98.0)
    assert math.isclose(prof["arm_pvelo"], 98.0 + 1.1 * 6.6 - 6.0)
    assert prof["stuff_z"] > 0
    assert prof["arm_stage2"] in (arm.CONFIRMED, arm.CONTRADICTED)
    # the stat-card biomechanics come off the same read, not a second calculation
    assert math.isclose(prof["biomech"]["ivb"], prof["arm_ivb"])


def test_an_untracked_starter_is_unmeasured_on_the_card_too() -> None:
    """A slice with no release columns leaves stage one standing, and never raises.

    The batter side hit exactly this as a live crash -- a frame cached before
    ``bat_speed`` existed took the article down with a ``KeyError`` -- so the
    pitcher card is pinned against the same failure on the release columns.
    """
    prof = rp.analyze("No Tracking", 7, _slice(120, tracked=False), DAY0 + timedelta(days=20))

    assert prof["arm_stage2"] == arm.UNMEASURED
    assert math.isnan(prof["arm_pvelo"]) and math.isnan(prof["stuff_z"])
    assert not math.isnan(prof["babip"])  # stage one is unaffected


def test_the_verdict_is_crossed_against_the_luck_term_not_the_results() -> None:
    """A hot line under a good arm disagrees; the same arm under a cold line agrees."""
    good = arm.ArmProfile(pitches=arm.WINDOW, pvelo=arm.LEAGUE["pvelo"][0] + 3.0)

    hot = rp._arm_fields(good, +0.030)  # xwOBA above wOBA: results ran hot
    cold = rp._arm_fields(good, -0.030)

    assert hot["arm_stage2"] == arm.CONTRADICTED
    assert cold["arm_stage2"] == arm.CONFIRMED


def test_no_physical_trend_is_written_to_the_card() -> None:
    """The card's only deltas stay SIERA, expected K% and velocity, as before."""
    prof = rp.analyze("Hunter Greene", 7, _slice(120), DAY0 + timedelta(days=20))

    deltas = {k for k in prof if k.startswith("d_")}
    assert deltas == {"d_siera", "d_xk", "d_vfa"}
