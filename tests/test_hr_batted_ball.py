"""Batted-ball profile terms on the home-run line.

Covers the air-contact split (EV filtered to fly balls/line drives), the pulled-
air PPV term, and the ground-ball / pop-up / soft-air NPV brakes.
"""

from __future__ import annotations

import pandas as pd

from mlb_engine.features.hr_gate import HRPowerGate
from mlb_engine.features.regression import (
    BL_PULL_AIR,
    BatterRegression,
    build_batter_regression,
)


def _reg(**kw: float | int) -> BatterRegression:
    """A league-average batter, overridden field by field."""
    base: dict[str, float | int] = {
        "bbe": 100,
        "barrel_rate": 0.080,
        "hard_hit": 0.400,
        "sweet_spot": 0.330,
        "bat_speed": 71.5,
        "max_ev": 108.0,
        "whiff": 0.240,
        "zone_contact": 0.820,
        "xba": 0.250,
        "xslg": 0.400,
        "babip": 0.290,
        "woba": 0.320,
        "xwoba": 0.320,
    }
    base.update(kw)
    return BatterRegression(**base)  # type: ignore[arg-type]


# --- air-contact split -------------------------------------------------------


def test_hr_reads_max_ev_on_air_contact_not_ground_balls() -> None:
    """A 115 mph ground ball must not credit the home-run line."""
    grounder = _reg(max_ev=115.0, fb_ld_max_ev=104.0, fb_ld_ev=93.0)
    flyball = _reg(max_ev=115.0, fb_ld_max_ev=115.0, fb_ld_ev=93.0)
    assert grounder.multipliers()["HR"] < flyball.multipliers()["HR"]


def test_air_metrics_fall_back_to_all_batted_balls() -> None:
    """Without launch-angle data the old unfiltered values are used."""
    reg = _reg(max_ev=112.0, hard_hit=0.44)
    assert reg.air_max_ev == 112.0
    assert reg.air_hard_hit == 0.44


def test_soft_air_hard_hit_still_brakes() -> None:
    soft = _reg(fb_ld_hard_hit=0.20, fb_ld_ev=92.0)
    ok = _reg(fb_ld_hard_hit=0.45, fb_ld_ev=92.0)
    assert soft.multipliers()["HR"] < ok.multipliers()["HR"]


# --- pulled air (PPV) --------------------------------------------------------


def test_pulled_air_lifts_home_runs() -> None:
    puller = _reg(pull_air_pct=0.34)
    oppo = _reg(pull_air_pct=0.10)
    assert puller.multipliers()["HR"] > oppo.multipliers()["HR"]
    # League-average pull-air is neutral, and a missing value changes nothing.
    assert _reg(pull_air_pct=BL_PULL_AIR).multipliers()["HR"] == _reg().multipliers()["HR"]


# --- NPV brakes --------------------------------------------------------------


def test_soft_fly_ball_ev_is_an_absolute_brake() -> None:
    """Under 90 mph on air contact, the hitter cannot clear a fence."""
    soft = _reg(fb_ld_ev=85.0)
    ok = _reg(fb_ld_ev=95.0)
    assert soft.multipliers()["HR"] < ok.multipliers()["HR"]
    # Exactly at the floor is not braked.
    assert _reg(fb_ld_ev=90.0).multipliers()["HR"] == _reg().multipliers()["HR"]


def test_pop_ups_brake_home_runs() -> None:
    popper = _reg(iffb_pct=0.30)
    normal = _reg(iffb_pct=0.05)
    assert popper.multipliers()["HR"] < normal.multipliers()["HR"]
    assert _reg(iffb_pct=0.15).multipliers()["HR"] == _reg().multipliers()["HR"]


def test_brakes_are_bounded_and_stack() -> None:
    worst = _reg(gb_rate=0.70, iffb_pct=0.40, fb_ld_ev=80.0, fb_ld_hard_hit=0.10)
    hr = worst.multipliers()["HR"]
    assert 0.50 <= hr < 0.75
    # Elite power is still capped at the same ceiling as before.
    best = _reg(barrel_rate=0.20, bat_speed=78.0, fb_ld_max_ev=118.0, pull_air_pct=0.40)
    assert best.multipliers()["HR"] <= 1.32


def test_missing_batted_ball_data_is_neutral() -> None:
    """NaN metrics must not be read as zero and brake the line."""
    assert _reg().multipliers()["HR"] == _reg(gb_rate=0.42).multipliers()["HR"]


# --- barrels per PA ----------------------------------------------------------


def test_barrel_per_pa_folds_in_contact_frequency() -> None:
    """Same barrel rate per batted ball, very different contact rates."""
    whiffer = _reg(bbe=50, barrel_rate=0.16, pa=200)
    contact = _reg(bbe=130, barrel_rate=0.16, pa=200)
    assert whiffer.barrel_per_pa == 0.16 * 50 / 200
    assert contact.barrel_per_pa > whiffer.barrel_per_pa
    # Unknown PA count reports NaN rather than a misleading zero.
    assert _reg(pa=0).barrel_per_pa != _reg(pa=0).barrel_per_pa


# --- metrics off a real Statcast frame ---------------------------------------


def _rows(n: int, la: float, speed: float, bb_type: str) -> list[dict[str, object]]:
    return [
        {
            "batter": 1,
            "launch_speed": speed,
            "launch_angle": la,
            "launch_speed_angle": 3,
            "bb_type": bb_type,
            "description": "hit_into_play",
            "events": "single",
            "bat_speed": 71.5,
        }
        for _ in range(n)
    ]


def test_air_split_and_iffb_are_read_off_the_frame() -> None:
    rows = (
        _rows(10, la=-5.0, speed=110.0, bb_type="ground_ball")  # scorched grounders
        + _rows(10, la=25.0, speed=88.0, bb_type="fly_ball")  # soft fly balls
        + _rows(5, la=55.0, speed=70.0, bb_type="popup")
    )
    reg = build_batter_regression(pd.DataFrame(rows))

    # Unfiltered max EV is set by the grounders; the air split is not.
    assert reg.max_ev == 110.0
    assert reg.fb_ld_max_ev == 88.0
    assert reg.fb_ld_ev < 90.0
    assert reg.fb_ld_hard_hit == 0.0
    # Pop-ups as a share of fly balls (5 of 15), and PAs are counted.
    assert reg.iffb_pct == 5 / 15
    assert reg.pa == 25


def test_frame_without_launch_angle_leaves_air_metrics_missing() -> None:
    rows = _rows(10, la=25.0, speed=100.0, bb_type="fly_ball")
    bare = pd.DataFrame(rows).drop(columns=["launch_angle"])
    reg = build_batter_regression(bare)
    assert reg.fb_ld_ev != reg.fb_ld_ev  # NaN
    assert reg.air_max_ev == reg.max_ev


# --- HR gate -----------------------------------------------------------------


def _gate(**kw: float | int | bool) -> HRPowerGate:
    return HRPowerGate(**kw)  # type: ignore[arg-type]


def test_gate_keeps_a_hitter_who_is_elite_per_pa_only() -> None:
    """Barrel below the per-BBE gate but elite per PA is still a buy."""
    keep, reason = _gate().allows(
        max_ev=112.0, barrel=0.120, bbe=120, barrel_pa=0.080, fb_ld_ev=95.0
    )
    assert keep, reason


def test_gate_drops_the_whiff_prone_slugger_on_a_thin_per_pa_rate() -> None:
    keep, reason = _gate().allows(
        max_ev=112.0, barrel=0.120, bbe=120, barrel_pa=0.030, fb_ld_ev=95.0
    )
    assert not keep
    assert "barrel" in reason


def test_gate_drops_soft_air_contact() -> None:
    keep, reason = _gate().allows(
        max_ev=112.0, barrel=0.200, bbe=120, barrel_pa=0.090, fb_ld_ev=86.0
    )
    assert not keep
    assert "FB/LD EV" in reason


def test_gate_is_neutral_without_the_new_inputs() -> None:
    """Omitting barrel_pa/fb_ld_ev must not change the existing behaviour."""
    keep, _ = _gate().allows(max_ev=112.0, barrel=0.200, bbe=120)
    assert keep


def test_new_gate_thresholds_are_env_tunable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_HR_BARREL_PA_GATE", "0")
    monkeypatch.setenv("MLBE_HR_MIN_FB_LD_EV", "0")
    gate = HRPowerGate.from_env()
    assert gate.barrel_pa_gate == 0.0
    # Both new tests disabled -> soft air contact no longer drops the buy.
    keep, _ = gate.allows(
        max_ev=112.0, barrel=0.200, bbe=120, barrel_pa=0.001, fb_ld_ev=70.0
    )
    assert keep
