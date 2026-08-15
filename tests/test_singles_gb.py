"""Batted-ball mix and shape on the singles line."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_GB_PULL,
    BL_GB_RATE,
    BL_LD_OPPOMID,
    BL_LD_RATE,
    BL_LD_SOFT,
    GB_RATE_CEILING,
    SINGLES_GB_SLOPE,
    BatterRegression,
    build_batter_regression,
)

NAN = float("nan")


def _reg(gb: float) -> BatterRegression:
    """A league-average batter apart from ground-ball rate."""
    return BatterRegression(
        bbe=100,
        barrel_rate=0.080,
        hard_hit=0.400,
        sweet_spot=0.330,
        bat_speed=71.5,
        max_ev=108.0,
        whiff=0.240,
        zone_contact=0.820,
        xba=0.250,
        xslg=0.400,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
        gb_rate=gb,
    )


def test_gb_term_is_live_by_default() -> None:
    # Was off, because an eight-slate fit could not separate the slope from
    # zero. It is now fitted out of time at p<1e-4, so it runs by default --
    # and the slope stays switchable for a counterfactual.
    worm_burner = _reg(0.60)
    assert Config().singles_gb is True
    assert worm_burner.multipliers()["1B"] > 1.0
    assert worm_burner.multipliers(singles_gb_slope=0.0)["1B"] == 1.0


def test_grounders_help_singles_and_hurt_home_runs() -> None:
    ground = _reg(0.60).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)
    air = _reg(0.25).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)
    assert ground["1B"] > 1.0 > air["1B"]
    assert _reg(BL_GB_RATE).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] == 1.0
    # Extra-base hits read contact quality, not the batted-ball mix.
    for key in ("2B", "3B"):
        assert ground[key] == air[key]
    # Home runs do: you cannot hit one on the ground, so a ground-ball hitter is
    # braked past the 50% ceiling while the fly-ball hitter is untouched.
    assert ground["HR"] < air["HR"]
    assert air["HR"] == _reg(GB_RATE_CEILING).multipliers()["HR"]


def test_gb_term_is_bounded() -> None:
    assert _reg(1.0).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] <= 1.06
    assert _reg(0.0).multipliers(singles_gb_slope=SINGLES_GB_SLOPE)["1B"] >= 0.94


def _shaped(
    gb_rate: float = BL_GB_RATE,
    ld_pct: float = BL_LD_RATE,
    gb_pull_pct: float = BL_GB_PULL,
    ld_soft_pct: float = BL_LD_SOFT,
    ld_oppomid_pct: float = BL_LD_OPPOMID,
) -> BatterRegression:
    """A league-average batter apart from the named batted-ball fields."""
    return dataclasses.replace(
        _reg(gb_rate),
        ld_pct=ld_pct,
        gb_pull_pct=gb_pull_pct,
        ld_soft_pct=ld_soft_pct,
        ld_oppomid_pct=ld_oppomid_pct,
    )


def test_the_league_average_batted_ball_profile_is_neutral() -> None:
    # Every mix and shape baseline is measured back through
    # ``build_batter_regression``, so a hitter sitting on all of them must come
    # out at exactly 1.0 -- no free offset for being ordinary.
    assert _shaped().multipliers()["1B"] == pytest.approx(1.0, abs=5e-3)


def test_line_drives_lift_singles_and_are_singles_only() -> None:
    slasher = _shaped(ld_pct=0.32).multipliers()
    flat = _shaped(ld_pct=0.16).multipliers()
    assert slasher["1B"] > 1.0 > flat["1B"]
    # Line-drive rate tested at p=.36 against total hits and is not wired into
    # the power lines at all.
    for key in ("2B", "3B", "HR"):
        assert slasher[key] == flat[key]


def test_pulled_grounders_are_a_penalty_not_a_bonus() -> None:
    # The rollover: a pulled grounder is the hardest-hit and most-defended one.
    rollover = _shaped(gb_pull_pct=0.85).multipliers()
    sprayer = _shaped(gb_pull_pct=0.50).multipliers()
    assert rollover["1B"] < 1.0 < sprayer["1B"]


def test_soft_and_cut_off_line_drives_are_the_singles_producing_ones() -> None:
    flare = _shaped(ld_soft_pct=0.60, ld_oppomid_pct=0.70).multipliers()
    scorched = _shaped(ld_soft_pct=0.24, ld_oppomid_pct=0.35).multipliers()
    assert flare["1B"] > 1.0 > scorched["1B"]


def test_shape_terms_have_a_kill_switch_and_leave_mix_alone() -> None:
    rollover = _shaped(gb_pull_pct=0.85)
    assert rollover.multipliers(singles_shape=False)["1B"] == pytest.approx(
        _shaped().multipliers()["1B"]
    )


def test_a_thin_batted_ball_class_drops_its_own_term_only() -> None:
    # Too few grounders to read spray, but enough line drives: the line-drive
    # shape still prices and the ground-ball one silently drops out.
    partial = _shaped(ld_soft_pct=0.60, gb_pull_pct=NAN)
    assert partial.multipliers()["1B"] > 1.0
    blind = _shaped(gb_pull_pct=NAN, ld_soft_pct=NAN, ld_oppomid_pct=NAN)
    assert blind.multipliers()["1B"] == pytest.approx(1.0, abs=5e-3)


def test_a_fly_ball_slugger_is_marked_down_on_singles_but_not_on_power() -> None:
    # The Schwarber case, and the reason this is a singles-only change: on the
    # real cache he prices at the 6th percentile for singles while keeping a
    # 1.23 home-run multiplier. A fly ball that leaves the yard is still a hit.
    slugger = _shaped(gb_rate=0.23, ld_pct=0.20, gb_pull_pct=0.85, ld_soft_pct=0.25)
    contact = _shaped(gb_rate=0.50, ld_pct=0.28, gb_pull_pct=0.55, ld_soft_pct=0.55)
    assert slugger.multipliers()["1B"] < 0.95
    assert contact.multipliers()["1B"] > 1.05
    # ... and the extra-base lines do not read any of it.
    for key in ("2B", "3B"):
        assert slugger.multipliers()[key] == contact.multipliers()[key]


def test_gb_rate_is_read_off_batted_ball_type() -> None:
    rows = [
        {
            "batter": 1,
            "launch_speed": 90.0,
            "launch_angle": 5.0,
            "launch_speed_angle": 3,
            "bb_type": "ground_ball" if i < 15 else "fly_ball",
            "description": "hit_into_play",
            "events": "single",
            "bat_speed": 71.5,
        }
        for i in range(20)
    ]
    reg = build_batter_regression(pd.DataFrame(rows))
    assert reg.gb_rate == 0.75

    # No bb_type column at all falls back to the league baseline, not zero.
    bare = pd.DataFrame(rows).drop(columns=["bb_type"])
    assert build_batter_regression(bare).gb_rate == BL_GB_RATE
