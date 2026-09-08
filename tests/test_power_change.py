"""Peak exit velocity and the fastball whiff are read over the windows they earn.

The point of these tests is not the arithmetic, it is the honesty: a level is
quoted only over a sample the measurement says supports it, a hitter short of
that reads as unmeasured rather than as the league, and a move is never printed
without the noise band a hitter who did not change would produce.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

pc = pytest.importorskip("mlb_engine.features.power_change")
art = pytest.importorskip("mlb_engine.output.regression_article")
cr = pytest.importorskip("scripts.comprehensive_report")


def _pitches(pa: int, *, max_ev: float, fb_whiff: float, start_pa: int = 0) -> pd.DataFrame:
    """One fastball swing plus one batted ball per plate appearance.

    Each plate appearance gets its own inning so the ordering is unambiguous. The
    whiff share lands exactly because the only fastball swings are the first
    pitches, and the peak lands on the *last* plate appearance of the range, which
    is the one a trailing window is guaranteed to contain.
    """
    rows = []
    last = start_pa + pa - 1
    for i in range(start_pa, start_pa + pa):
        first = {
            "batter": 1,
            "game_date": "2026-06-01",
            "inning": i + 1,
            "inning_topbot": "Top",
            "pitch_type": "FF",
            "description": "swinging_strike" if (i - start_pa) < round(fb_whiff * pa) else "foul",
            "type": "S",
            "events": None,
            "launch_speed": math.nan,
        }
        rows.append(first)
        rows.append(
            {
                **first,
                "pitch_type": "SL",
                "description": "hit_into_play",
                "type": "X",
                "events": "field_out",
                "launch_speed": max_ev if i == last else 80.0,
            }
        )
    return pd.DataFrame(rows)


def _profile(**over) -> dict:
    p = {
        "power_pa": 400,
        "max_ev": 111.4,
        "max_ev_pa": pc.WINDOW["max_ev"],
        "d_max_ev": 1.2,
        "max_ev_moved": False,
        "fb_whiff": 0.24,
        "fb_whiff_pa": pc.WINDOW["fb_whiff"],
        "fb_swings": 120,
        "d_fb_whiff": 0.03,
        "fb_whiff_moved": False,
        "power_block_pa": pc.MOVE_BLOCK["fb_whiff"],
    }
    p.update(over)
    return p


def test_each_metric_is_read_over_the_sample_it_stabilises_at() -> None:
    """The two windows differ because the two reliability curves differ."""
    assert pc.WINDOW["fb_whiff"] < pc.WINDOW["max_ev"]
    assert pc.FLOOR["max_ev"] < pc.WINDOW["max_ev"]
    got = pc.build_power_change(_pitches(400, max_ev=112.3, fb_whiff=0.25))
    assert got.max_ev_pa == pc.WINDOW["max_ev"]
    assert got.fb_whiff_pa == pc.WINDOW["fb_whiff"]
    assert got.max_ev == pytest.approx(112.3)
    assert got.fb_whiff == 0.0  # every whiff of this hitter's is outside the window
    assert got.stable("max_ev") and got.stable("fb_whiff")


def test_a_hitter_under_the_floor_is_unmeasured_and_not_the_league() -> None:
    got = pc.build_power_change(_pitches(30, max_ev=113.0, fb_whiff=0.30))
    assert got.pa == 30
    assert got.max_ev != got.max_ev
    assert got.fb_whiff != got.fb_whiff
    assert got.max_ev_pa == 0
    assert got.moved("max_ev") is False


def test_a_fastball_rate_needs_fastball_swings_and_not_only_plate_appearances() -> None:
    """The denominator is swings at velocity, so a hitter who takes them reads blank."""
    df = _pitches(200, max_ev=110.0, fb_whiff=0.20)
    got = pc.build_power_change(df.assign(pitch_type="SL"))
    assert got.max_ev == got.max_ev
    assert got.fb_whiff != got.fb_whiff
    assert got.fb_swings < pc.MIN_FB_SWINGS


def test_the_move_is_read_over_two_equal_blocks_and_only_when_both_exist() -> None:
    short = pc.build_power_change(_pitches(80, max_ev=110.0, fb_whiff=0.20))
    assert short.block_pa == 0
    assert short.d_max_ev != short.d_max_ev

    block = max(pc.MOVE_BLOCK.values())
    prior = _pitches(block, max_ev=104.0, fb_whiff=0.10)
    recent = _pitches(block, max_ev=112.0, fb_whiff=0.40, start_pa=block)
    got = pc.build_power_change(pd.concat([prior, recent], ignore_index=True))
    assert got.block_pa == block
    assert got.d_max_ev == pytest.approx(8.0)
    assert got.d_fb_whiff == pytest.approx(0.30, abs=0.02)
    assert got.fb_whiff_recent > got.fb_whiff_prior


def test_a_move_inside_its_own_noise_is_not_called_a_move() -> None:
    block = max(pc.MOVE_BLOCK.values())
    small = pc.band("max_ev", block) - 1.0
    prior = _pitches(block, max_ev=104.0, fb_whiff=0.20)
    recent = _pitches(block, max_ev=104.0 + small, fb_whiff=0.22, start_pa=block)
    got = pc.build_power_change(pd.concat([prior, recent], ignore_index=True))
    assert got.d_max_ev == pytest.approx(small)
    assert got.moved("max_ev") is False
    assert got.moved("fb_whiff") is False


def test_the_band_tightens_as_the_block_grows_and_never_extrapolates_off_the_grid() -> None:
    assert pc.band("max_ev", 25) > pc.band("max_ev", 60) > pc.band("max_ev", 130)
    assert pc.band("max_ev", 5) == pc.band("max_ev", 25)
    assert pc.band("fb_whiff", 900) == pc.band("fb_whiff", 130)


def test_the_article_prints_both_levels_with_the_windows_they_were_read_over() -> None:
    line = art._power_line(_profile())
    assert "MaxEV 111.4 mph over 192 PA" in line
    assert "FB whiff 24% over 108 PA on 120 fastball swings" in line
    assert "r=.70" in line


def test_the_article_prints_the_move_against_its_band_and_calls_it_a_diagnostic() -> None:
    line = art._power_line(_profile())
    assert "MaxEV +1.2 mph vs prior 52 PA (inside the &plusmn;" in line
    assert "FB whiff +3.0pp vs prior 52 PA (inside the &plusmn;" in line
    assert "diagnostics" in line
    moved = art._power_line(_profile(d_max_ev=9.0, max_ev_moved=True))
    assert "MaxEV +9.0 mph vs prior 52 PA (clears the &plusmn;" in moved


def test_the_article_says_what_a_missing_read_is_missing() -> None:
    line = art._power_line(
        _profile(
            power_pa=40,
            max_ev=math.nan,
            max_ev_pa=0,
            fb_whiff=math.nan,
            fb_whiff_pa=0,
            fb_swings=3,
            d_max_ev=math.nan,
            d_fb_whiff=math.nan,
            power_block_pa=0,
        )
    )
    assert "MaxEV &mdash; (40 PA, under the 49-PA floor)" in line
    assert f"needs {pc.FLOOR['fb_whiff']} PA and {pc.MIN_FB_SWINGS} fastball swings" in line
    assert "MaxEV move unmeasured (needs 98 PA)" in line
    assert "FB whiff move unmeasured (needs 104 PA)" in line
    assert "111.4" not in line


def test_a_hitter_with_no_plate_appearances_prints_no_line_at_all() -> None:
    assert art._power_line(_profile(power_pa=0)) == ""
    assert cr._power_cells(_profile(power_pa=0)) == ""


def test_a_level_read_short_of_its_window_is_labelled_provisional() -> None:
    """Past the floor is worth printing; short of r=.70 is not worth trusting."""
    short = _profile(max_ev_pa=114, fb_whiff_pa=80)
    line = art._power_line(short)
    assert f"over 114 PA (provisional, {pc.WINDOW['max_ev']} PA to settle)" in line
    assert f"swings (provisional, {pc.WINDOW['fb_whiff']} PA to settle)" in line
    assert "provisional" in cr._power_cells(short)
    assert "provisional" not in art._power_line(_profile())
    assert "provisional" not in cr._power_cells(_profile())


def test_the_stat_card_carries_the_same_numbers_as_the_article() -> None:
    cells = cr._power_cells(_profile())
    assert "max EV <b>111.4</b> (192 PA)" in cells
    assert "FB whiff <b>24%</b> (108 PA, 120 FB swings)" in cells
    assert "noise, band &plusmn;" in cells
    assert f"league {pc.BL_MAX_EV:.1f}" in cells
