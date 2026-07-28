"""Unit tests for the moneyline sharp-money confirmation gate."""

from __future__ import annotations

from mlb_engine.features.ml_gate import MLSharpGate


def test_gate_keeps_sharp_confirmed_side() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=0.0)
    keep, reason = gate.allows(handle_pct=70.0, bets_pct=50.0)
    assert keep is True
    assert "OK" in reason


def test_gate_demotes_when_money_against() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=0.0)
    keep, reason = gate.allows(handle_pct=40.0, bets_pct=60.0)
    assert keep is False
    assert "PASS" in reason


def test_gate_respects_positive_threshold() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=10.0)
    keep, _ = gate.allows(handle_pct=55.0, bets_pct=50.0)
    assert keep is False
    keep, _ = gate.allows(handle_pct=65.0, bets_pct=50.0)
    assert keep is True


def test_gate_neutral_when_missing_split() -> None:
    gate = MLSharpGate(enabled=True, min_divergence=0.0)
    keep, reason = gate.allows(handle_pct=None, bets_pct=None)
    assert keep is True
    assert "neutral" in reason


def test_gate_disabled_keeps_everything() -> None:
    gate = MLSharpGate(enabled=False)
    keep, reason = gate.allows(handle_pct=10.0, bets_pct=90.0)
    assert keep is True
    assert reason == ""


def test_from_env_defaults(monkeypatch) -> None:
    for k in ("MLBE_ML_SHARP_GATE", "MLBE_ML_MIN_DIVERGENCE"):
        monkeypatch.delenv(k, raising=False)
    gate = MLSharpGate.from_env()
    assert gate.enabled is True
    assert gate.min_divergence == 0.0


def test_from_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_ML_SHARP_GATE", "0")
    assert MLSharpGate.from_env().enabled is False
