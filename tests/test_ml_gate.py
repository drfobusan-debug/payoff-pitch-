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


def test_upgrade_promotes_sharp_backed_pass() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0, max_fair_prob=0.65)
    up, reason = gate.upgrades(handle_pct=65.0, bets_pct=50.0, fair_prob=0.48)
    assert up is True
    assert "BUY" in reason


def test_upgrade_skipped_when_divergence_too_small() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0)
    up, _ = gate.upgrades(handle_pct=52.0, bets_pct=50.0, fair_prob=0.48)
    assert up is False


def test_upgrade_skips_heavy_chalk() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0, max_fair_prob=0.65)
    up, reason = gate.upgrades(handle_pct=70.0, bets_pct=50.0, fair_prob=0.80)
    assert up is False
    assert "chalk" in reason


def test_upgrade_disabled_never_promotes() -> None:
    gate = MLSharpGate(upgrade_enabled=False)
    up, reason = gate.upgrades(handle_pct=80.0, bets_pct=40.0, fair_prob=0.40)
    assert up is False
    assert reason == ""


def test_upgrade_requires_split() -> None:
    gate = MLSharpGate(upgrade_divergence=5.0)
    up, _ = gate.upgrades(handle_pct=None, bets_pct=None, fair_prob=0.48)
    assert up is False


def test_from_env_defaults(monkeypatch) -> None:
    for k in (
        "MLBE_ML_SHARP_GATE",
        "MLBE_ML_MIN_DIVERGENCE",
        "MLBE_ML_SHARP_UPGRADE",
        "MLBE_ML_UPGRADE_DIVERGENCE",
        "MLBE_ML_UPGRADE_MAX_FAIR",
    ):
        monkeypatch.delenv(k, raising=False)
    gate = MLSharpGate.from_env()
    assert gate.enabled is True
    assert gate.min_divergence == 0.0
    assert gate.upgrade_enabled is True
    assert gate.upgrade_divergence == 5.0
    assert gate.max_fair_prob == 0.65


def test_from_env_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_ML_SHARP_GATE", "0")
    assert MLSharpGate.from_env().enabled is False


def test_from_env_upgrade_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_ML_SHARP_UPGRADE", "0")
    assert MLSharpGate.from_env().upgrade_enabled is False
