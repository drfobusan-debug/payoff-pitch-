"""Unit tests for the home-run power gate."""

from __future__ import annotations

from mlb_engine.features.hr_gate import HRPowerGate


def test_gate_keeps_power_hitter() -> None:
    gate = HRPowerGate(enabled=True, min_max_ev=108.0, min_barrel=0.06, min_bbe=15)
    keep, reason = gate.allows(max_ev=112.0, barrel=0.12, bbe=50)
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


def test_from_env_defaults(monkeypatch) -> None:
    for k in ("MLBE_HR_POWER_GATE", "MLBE_HR_MAX_EV", "MLBE_HR_BARREL", "MLBE_HR_MIN_BBE"):
        monkeypatch.delenv(k, raising=False)
    gate = HRPowerGate.from_env()
    assert gate.enabled is True
    assert gate.min_max_ev == 109.0
    assert gate.min_barrel == 0.07


def test_from_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_HR_POWER_GATE", "0")
    assert HRPowerGate.from_env().enabled is False
