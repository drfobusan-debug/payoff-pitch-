"""The price band: what it refuses, and what it only writes down."""

from __future__ import annotations

import pytest

from cfb_engine.market.priceband import LONG_GATE, SHORT_GATE, PriceBand


def test_defaults_refuse_nothing() -> None:
    """Shipping off is the point: no graded CFB row has set this band yet."""
    band = PriceBand.from_env()
    assert band.enabled is False
    keep, reason, gate = band.verdict(1200.0)
    assert keep is True
    assert gate is None
    assert "measuring" in reason


def test_an_out_of_band_price_is_annotated_even_with_the_band_off() -> None:
    band = PriceBand()
    keep, reason, gate = band.verdict(-400.0)
    assert (keep, gate) == (True, None)
    assert "shorter than -250" in reason


def test_enabled_refuses_both_tails_and_attributes_the_gate() -> None:
    band = PriceBand(enabled=True)
    keep, reason, gate = band.verdict(-400.0)
    assert (keep, gate) == (False, SHORT_GATE)
    assert reason.endswith("-> PASS")
    assert band.verdict(400.0)[2] == LONG_GATE


def test_the_band_is_inclusive_of_its_own_boundaries() -> None:
    """A threshold that refuses its own number makes every backtest of it a
    point estimate at the boundary."""
    band = PriceBand(enabled=True)
    assert band.verdict(-250.0) == (True, "", None)
    assert band.verdict(200.0) == (True, "", None)
    assert band.verdict(-251.0)[0] is False
    assert band.verdict(201.0)[0] is False


def test_a_missing_price_is_a_data_hole_not_a_refusal() -> None:
    assert PriceBand(enabled=True).verdict(None) == (True, "", None)


def test_a_market_override_can_arm_one_board_and_leave_the_others_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single band cannot serve both boards -- armed engine-wide at -250/+200
    the spread survives only because it never leaves -120/+100."""
    monkeypatch.setenv("CFBE_PRICE_BAND_GAME_ML", "1")
    monkeypatch.setenv("CFBE_PRICE_MIN_GAME_ML", "-150")
    band = PriceBand.from_env()
    ml = band.for_market("game_ml")
    assert (ml.enabled, ml.min_american) == (True, -150.0)
    assert ml.verdict(-200.0)[0] is False
    ats = band.for_market("game_ats")
    assert ats.enabled is False
    assert ats.verdict(-200.0)[0] is True


def test_engine_wide_settings_are_inherited_by_every_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFBE_PRICE_BAND", "1")
    monkeypatch.setenv("CFBE_PRICE_MAX", "120")
    band = PriceBand.from_env().for_market("game_total")
    assert band.enabled is True
    assert band.verdict(130.0)[2] == LONG_GATE


def test_the_moneyline_ships_refusing_dogs_longer_than_plus_200() -> None:
    """The one live tail, with nothing configured."""
    ml = PriceBand.from_env().for_market("game_ml")
    assert ml.enabled is True
    keep, reason, gate = ml.verdict(260.0)
    assert (keep, gate) == (False, LONG_GATE)
    assert "longer than +200" in reason


def test_the_moneylines_short_tail_ships_disarmed() -> None:
    """Long-side only: a short favourite is graded off the ledger's odds column,
    not refused on MLB's number."""
    ml = PriceBand.from_env().for_market("game_ml")
    assert ml.min_american is None
    assert ml.verdict(-3000.0) == (True, "", None)


def test_the_spread_and_total_boards_still_refuse_nothing() -> None:
    band = PriceBand.from_env()
    for market in ("game_ats", "game_total"):
        assert band.for_market(market).enabled is False


def test_the_engine_wide_flag_cannot_silently_disarm_a_market_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CFBE_PRICE_BAND=0`` is the engine-wide default, not an override of a
    tail that ships live -- otherwise the off switch is one someone sets by
    accident."""
    monkeypatch.setenv("CFBE_PRICE_BAND", "0")
    assert PriceBand.from_env().for_market("game_ml").enabled is True
    monkeypatch.setenv("CFBE_PRICE_BAND_GAME_ML", "0")
    assert PriceBand.from_env().for_market("game_ml").enabled is False


def test_a_tail_can_be_disarmed_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFBE_PRICE_MAX_GAME_ML", "off")
    ml = PriceBand.from_env().for_market("game_ml")
    assert ml.max_american is None
    assert ml.verdict(2500.0) == (True, "", None)


def test_a_market_override_can_disarm_a_band_armed_engine_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFBE_PRICE_BAND", "1")
    monkeypatch.setenv("CFBE_PRICE_BAND_GAME_TOTAL", "0")
    band = PriceBand.from_env()
    assert band.for_market("game_ml").enabled is True
    assert band.for_market("game_total").enabled is False
