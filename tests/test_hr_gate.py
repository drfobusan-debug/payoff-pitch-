"""Unit tests for the home-run power gate."""

from __future__ import annotations

from mlb_engine.features.hr_gate import HRPowerGate


def test_gate_keeps_elite_barrel_hitter() -> None:
    # Barrel above the standing gate keeps the buy without needing a trend.
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(max_ev=112.0, barrel=0.16, bbe=50)
    assert keep is True
    assert "OK" in reason


def test_gate_demotes_weak_power_max_ev() -> None:
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(max_ev=104.0, barrel=0.10, bbe=50)
    assert keep is False
    assert "PASS" in reason


def test_gate_demotes_weak_power_barrel() -> None:
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(max_ev=110.0, barrel=0.02, bbe=50)
    assert keep is False
    assert "PASS" in reason


def test_gate_neutral_on_thin_sample() -> None:
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(max_ev=104.0, barrel=0.02, bbe=5)
    assert keep is True
    assert "neutral" in reason


def test_gate_neutral_when_missing_data() -> None:
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, _ = gate.allows(max_ev=None, barrel=None, bbe=50)
    assert keep is True


def test_gate_disabled_keeps_everything() -> None:
    gate = HRPowerGate(enabled=False)
    keep, reason = gate.allows(max_ev=90.0, barrel=0.0, bbe=100)
    assert keep is True
    assert reason == ""


def test_barrel_gate_demotes_mid_barrel_when_flat() -> None:
    # Clears the power floor but barrel < 0.15 and no rising trend -> demote,
    # even though max EV is strong (the gate fires regardless of EV).
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(max_ev=112.0, barrel=0.10, bbe=50)
    assert keep is False
    assert "PASS" in reason and "not rising" in reason


def test_barrel_gate_keeps_rising_barrel() -> None:
    # Sub-0.15 barrel is kept when the last 3 weeks are above the 6-week rate.
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(
        max_ev=112.0, barrel=0.10, bbe=50, barrel_3w=0.14, barrel_6w=0.09
    )
    assert keep is True
    assert "rising" in reason


def test_barrel_gate_demotes_falling_barrel() -> None:
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(
        max_ev=112.0, barrel=0.10, bbe=50, barrel_3w=0.08, barrel_6w=0.12
    )
    assert keep is False
    assert "PASS" in reason


def test_barrel_gate_disabled_via_zero_threshold() -> None:
    # barrel_gate=0 keeps the power floor but disables the level/trend gate.
    gate = HRPowerGate(
        enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15, barrel_gate=0.0
    )
    keep, reason = gate.allows(max_ev=112.0, barrel=0.10, bbe=50)
    assert keep is True
    assert "OK" in reason


def test_from_env_defaults(monkeypatch) -> None:
    for k in (
        "MLBE_HR_POWER_GATE", "MLBE_HR_MAX_EV", "MLBE_HR_BARREL",
        "MLBE_HR_MIN_BBE", "MLBE_HR_BARREL_GATE",
    ):
        monkeypatch.delenv(k, raising=False)
    gate = HRPowerGate.from_env()
    assert gate.enabled is True
    assert gate.min_max_ev == 109.0
    assert gate.min_barrel == 0.07
    assert gate.barrel_gate == 0.15


def test_from_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_HR_POWER_GATE", "0")
    assert HRPowerGate.from_env().enabled is False
