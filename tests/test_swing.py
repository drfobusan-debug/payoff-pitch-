"""Bat tracking as the screen's second stage: windows, levels, and the 2x2.

The pins here are the ones a refactor could invert without looking wrong: that a
level is read over its own window of swings rather than the whole slice, that a
thin hitter reads as unmeasured rather than as league average, that no trend is
exposed, and that the confirm/contradict cross keeps its four corners.
"""

from __future__ import annotations

import math

import pandas as pd

from mlb_engine.features import swing
from mlb_engine.features.swing import (
    CONFIRMED,
    CONTRADICTED,
    LEAGUE,
    UNMEASURED,
    WINDOW,
    SwingProfile,
    build_swing_profile,
    readable,
    reliability,
    stage_two,
)


def _swings(
    n: int, *, bat: float, ev: float = 95.0, length: float = 7.3, day0: int = 0
) -> pd.DataFrame:
    """``n`` tracked competitive swings at one bat speed, oldest first."""
    start = pd.Timestamp("2026-04-01") + pd.Timedelta(days=day0)
    return pd.DataFrame(
        [
            {
                "game_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                "batter": 1,
                "description": "hit_into_play",
                "bat_speed": bat,
                "swing_length": length,
                "launch_speed": ev,
                "release_speed": 93.0,
            }
            for i in range(n)
        ]
    )


# --- reliability ----------------------------------------------------------


def test_the_windows_are_the_measured_crossings_not_a_round_number() -> None:
    """Bat speed repeats in a handful of swings and blast rate needs dozens.

    The whole point of reading each measure on its own window is that these are
    two orders of magnitude apart; a single window would either starve bat speed
    of nothing or read blast rate off an at-bat.
    """
    assert swing.SWINGS_FOR_READABLE["bat_speed"] < 5
    assert swing.SWINGS_FOR_READABLE["fast"] < 5
    assert 40 < swing.SWINGS_FOR_READABLE["blast"] < 60
    assert 55 < swing.SWINGS_FOR_READABLE["squared_up"] < 80
    for metric, n in WINDOW.items():
        assert reliability(metric, n) >= 0.75, metric


def test_reliability_rises_with_the_sample_and_holds_outside_the_grid() -> None:
    assert reliability("blast", 10) < reliability("blast", 50) < reliability("blast", 250)
    assert reliability("blast", 5000) == reliability("blast", 250)
    assert reliability("blast", 1) == reliability("blast", 3)
    assert reliability("attack_angle", 10) == 1.0  # unmeasured metrics are not scored
    assert readable("bat_speed", 10)
    assert not readable("squared_up", 10)


# --- levels ---------------------------------------------------------------


def test_a_level_is_read_over_its_own_window_of_recent_swings() -> None:
    """Bat speed comes off the last twelve swings, not the whole season."""
    slow = _swings(200, bat=68.0)
    fast = _swings(WINDOW["bat_speed"], bat=80.0, day0=200)
    prof = build_swing_profile(pd.concat([slow, fast], ignore_index=True))
    assert prof.bat_speed == 80.0
    assert prof.swings == 200 + WINDOW["bat_speed"]


def test_a_thin_hitter_reads_as_unmeasured_rather_than_as_league() -> None:
    prof = build_swing_profile(_swings(10, bat=75.0))
    assert prof.bat_speed == 75.0  # three swings is enough for this one
    assert math.isnan(prof.blast)
    assert math.isnan(prof.squared_up)
    assert math.isnan(prof.power_z)  # so the swing carries no decision
    assert stage_two(0.080, prof) == UNMEASURED


def test_an_untracked_swing_is_dropped_rather_than_read_as_a_slow_one() -> None:
    rows = _swings(60, bat=75.0)
    rows.loc[rows.index[:30], "bat_speed"] = None
    prof = build_swing_profile(rows)
    assert prof.swings == 30
    assert prof.bat_speed == 75.0


def test_takes_and_bunts_are_not_competitive_swings() -> None:
    rows = _swings(60, bat=75.0)
    rows.loc[rows.index[:30], "description"] = "ball"
    assert build_swing_profile(rows).swings == 30


def test_an_empty_slice_is_a_profile_of_nothing() -> None:
    prof = build_swing_profile(pd.DataFrame())
    assert prof.swings == 0
    assert math.isnan(prof.bat_speed)
    assert stage_two(-0.030, prof) == UNMEASURED


def test_a_z_score_reads_against_league_measured_the_same_way() -> None:
    mu, sd = LEAGUE["bat_speed"]
    prof = SwingProfile(swings=500, bat_speed=mu + sd)
    assert abs(prof.z("bat_speed") - 1.0) < 1e-9
    assert math.isnan(prof.z("squared_up"))


def test_no_trend_is_exposed_at_all() -> None:
    """The recent-versus-prior move in these measures predicts nothing.

    Measured on the same 3,175 batter-windows the levels were validated on: bat
    speed t +1.4, blast t -0.3 on next-fortnight total bases. The article and the
    screen cannot print or price what the module does not compute.
    """
    fields = set(SwingProfile(swings=0).__dict__)
    assert not any("delta" in f or f.startswith("d_") or f.endswith("_prior") for f in fields)
    assert not hasattr(SwingProfile(swings=0), "attack_angle")


# --- the cross ------------------------------------------------------------


def _prof(power: float) -> SwingProfile:
    """A profile whose bat speed and blast rate both sit ``power`` SD from league."""
    bmu, bsd = LEAGUE["bat_speed"]
    zmu, zsd = LEAGUE["blast"]
    return SwingProfile(
        swings=400, bat_speed=bmu + power * bsd, blast=zmu + power * zsd
    )


def test_the_four_corners_of_the_cross() -> None:
    good, bad = _prof(1.0), _prof(-1.0)
    # results above the contact: a good swing says the gap is about to be wrong
    assert stage_two(0.080, good) == CONTRADICTED
    assert stage_two(0.080, bad) == CONFIRMED
    # results below it: a good swing says the rebound has a bat behind it
    assert stage_two(-0.080, good) == CONFIRMED
    assert stage_two(-0.080, bad) == CONTRADICTED


def test_squared_up_is_not_allowed_into_the_power_read() -> None:
    """It predicts hits and is negatively signed on home runs (t -9.5).

    Folding it into the same index as blast rate would cancel the signal the
    rescue is built on, so a hitter who only squares the ball up is not rescued.
    """
    mu, sd = LEAGUE["squared_up"]
    prof = SwingProfile(swings=400, squared_up=mu + 2 * sd)
    assert prof.contact_z > 1.5
    assert math.isnan(prof.power_z)
    assert stage_two(0.080, prof) == UNMEASURED
