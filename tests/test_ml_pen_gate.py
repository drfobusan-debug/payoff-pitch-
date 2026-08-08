"""Unit tests for the moneyline bullpen-depletion gate."""

from __future__ import annotations

from mlb_engine.features.ml_gate import MLPenGate


def test_keeps_buy_with_rested_pen() -> None:
    gate = MLPenGate()
    keep, reason = gate.allows(own_fatigue=30.0, opp_fatigue=20.0)
    assert keep is True
    assert "OK" in reason


def test_demotes_buy_on_depleted_pen() -> None:
    gate = MLPenGate()
    keep, reason = gate.allows(own_fatigue=80.0, opp_fatigue=20.0)
    assert keep is False
    assert "depleted" in reason


def test_keeps_buy_when_both_pens_worked() -> None:
    """Depletion is only actionable relative to the opponent's pen."""
    gate = MLPenGate()
    keep, reason = gate.allows(own_fatigue=75.0, opp_fatigue=70.0)
    assert keep is True
    assert "both pens worked" in reason


def test_demotes_when_opponent_fatigue_unknown() -> None:
    gate = MLPenGate()
    keep, reason = gate.allows(own_fatigue=90.0, opp_fatigue=None)
    assert keep is False
    assert "opp unknown" in reason


def test_neutral_without_workload_read() -> None:
    gate = MLPenGate()
    keep, reason = gate.allows(own_fatigue=None, opp_fatigue=40.0)
    assert keep is True
    assert "neutral" in reason


def test_availability_feed_overrides_proxy() -> None:
    gate = MLPenGate()
    keep, reason = gate.allows(own_fatigue=10.0, opp_fatigue=90.0, own_availability=0.1)
    assert keep is False
    assert "availability" in reason


def test_availability_above_floor_keeps_buy() -> None:
    gate = MLPenGate()
    keep, _ = gate.allows(own_fatigue=10.0, opp_fatigue=90.0, own_availability=0.8)
    assert keep is True


def test_disabled_gate_keeps_everything() -> None:
    gate = MLPenGate(enabled=False)
    keep, reason = gate.allows(own_fatigue=100.0, opp_fatigue=0.0, own_availability=0.0)
    assert keep is True
    assert reason == ""


def test_thresholds_are_tunable() -> None:
    gate = MLPenGate(depleted=40.0, min_edge=5.0)
    keep, _ = gate.allows(own_fatigue=50.0, opp_fatigue=40.0)
    assert keep is False
