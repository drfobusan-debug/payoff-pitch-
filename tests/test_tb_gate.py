"""Unit tests for the total-bases power and opposing-starter gates."""

from __future__ import annotations

from mlb_engine.features.regression import BatterRegression, PitcherRegression
from mlb_engine.features.tb_gate import TBGate
from mlb_engine.models.selectors import TBSelector


def _gate(**kw) -> TBGate:
    return TBGate(**kw)


def test_power_gate_keeps_a_real_power_bat() -> None:
    assert _gate().power_reason(0.110, 112.0, 80) is None


def test_power_gate_drops_a_low_barrel_bat() -> None:
    reason = _gate().power_reason(0.030, 112.0, 80)
    assert reason is not None
    assert "barrel" in reason


def test_power_gate_drops_a_low_max_ev_bat() -> None:
    reason = _gate().power_reason(0.110, 103.0, 80)
    assert reason is not None
    assert "max_ev" in reason


def test_power_gate_is_neutral_on_a_thin_sample() -> None:
    # Below min_bbe the gate must not punish small-sample noise.
    assert _gate().power_reason(0.010, 99.0, 5) is None


def test_power_gate_is_neutral_without_power_data() -> None:
    assert _gate().power_reason(None, None, 80) is None


def test_opponent_gate_drops_over_vs_contact_suppressor() -> None:
    reason = _gate().opponent_reason(0.045, 0.330, 200)
    assert reason is not None
    assert "suppressor" in reason


def test_opponent_gate_needs_both_signals_low() -> None:
    # Low barrel but league-average hard contact is not enough to veto.
    assert _gate().opponent_reason(0.045, 0.410, 200) is None
    assert _gate().opponent_reason(0.090, 0.330, 200) is None


def test_opponent_gate_is_neutral_on_a_thin_sample() -> None:
    assert _gate().opponent_reason(0.010, 0.200, 5) is None


def test_kill_switches_disable_the_gates() -> None:
    off = _gate(enabled=False)
    assert off.power_reason(0.010, 99.0, 80) is None
    assert off.opponent_reason(0.010, 0.200, 200) is None
    # The opponent half can be disabled on its own.
    opp_off = _gate(opp_enabled=False)
    assert opp_off.opponent_reason(0.010, 0.200, 200) is None
    assert opp_off.power_reason(0.010, 99.0, 80) is not None


def test_from_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_TB_MIN_BARREL", "0.2")
    monkeypatch.setenv("MLBE_TB_OPP_GATE", "0")
    gate = TBGate.from_env()
    assert gate.min_barrel == 0.2
    assert gate.opp_enabled is False


def _reg(hard_hit: float, barrel: float) -> PitcherRegression:
    return PitcherRegression(
        bbe=200,
        pitches=1500,
        babip_allowed=0.290,
        woba_allowed=0.320,
        xwoba_allowed=0.320,
        hard_hit_allowed=hard_hit,
        barrel_allowed=barrel,
        csw=0.280,
        k_pct=0.220,
        bb_pct=0.080,
        two_strike_whiff=0.280,
    )


def test_hard_hit_allowed_moves_extra_base_multipliers() -> None:
    """Hard contact allowed must lift 2B/3B/HR and leave singles alone."""
    soft = _reg(hard_hit=0.320, barrel=0.080).allowed_multipliers()
    hard = _reg(hard_hit=0.480, barrel=0.080).allowed_multipliers()

    assert hard["2B"] > soft["2B"]
    assert hard["3B"] > soft["3B"]
    assert hard["HR"] > soft["HR"]
    # Singles are driven by BABIP/dxwOBA only, so they are unchanged.
    assert hard["1B"] == soft["1B"]


def test_league_average_hard_hit_is_neutral() -> None:
    mult = _reg(hard_hit=0.400, barrel=0.080).allowed_multipliers()
    assert mult["2B"] == mult["1B"]


def _breg(barrel: float, max_ev: float, bbe: int = 80) -> BatterRegression:
    return BatterRegression(
        bbe=bbe,
        barrel_rate=barrel,
        hard_hit=0.400,
        sweet_spot=0.330,
        bat_speed=71.5,
        max_ev=max_ev,
        whiff=0.240,
        zone_contact=0.820,
        xba=0.250,
        xslg=0.400,
        babip=0.290,
        woba=0.320,
        xwoba=0.320,
    )


def test_selector_carries_the_inputs_the_gate_reads() -> None:
    """The TB selector must expose barrel/max-EV/BBE for the pipeline gate."""
    sel = TBSelector().select(_breg(barrel=0.030, max_ev=103.0))
    gate = _gate()

    assert sel.hr_barrel == 0.030
    assert sel.hr_max_ev == 103.0
    assert sel.hr_bbe == 80
    # A weak-power bat routed through the selector is excluded.
    assert gate.power_reason(sel.hr_barrel, sel.hr_max_ev, sel.hr_bbe) is not None

    strong = TBSelector().select(_breg(barrel=0.110, max_ev=112.0))
    assert gate.power_reason(strong.hr_barrel, strong.hr_max_ev, strong.hr_bbe) is None


def test_selector_leaves_gate_neutral_on_thin_sample() -> None:
    # Too few batted balls -> selector reports no power data -> no exclusion.
    sel = TBSelector().select(_breg(barrel=0.010, max_ev=95.0, bbe=3))
    assert _gate().power_reason(sel.hr_barrel, sel.hr_max_ev, sel.hr_bbe) is None
