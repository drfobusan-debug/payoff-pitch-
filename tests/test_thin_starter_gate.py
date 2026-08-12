"""Tests for the thin-Statcast starter gate.

A probable starter with too few tracked pitches (e.g. a debut/call-up with no
MLB data) is priced off an optimistic prior, which manufactures phantom edges on
that game's starter-driven markets. The gate vetoes those to Pass.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline


class _IdentityCalibrator:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline(cfg: Config) -> Pipeline:
    # Exercise the pure gate helper without the heavy Pipeline.__init__.
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    return p


def _mk_pipeline(cfg: Config) -> Pipeline:
    p = _pipeline(cfg)
    p._calibrator = _IdentityCalibrator()
    p._shrink = None
    p._splits = {}
    return p


def test_thin_starter_flagged_below_floor() -> None:
    cfg = Config(thin_starter_gate=True, thin_starter_min_pitches=150)
    reason = _pipeline(cfg)._thin_starter_reason("Quinn Mathews", 0)
    assert reason is not None
    assert "Quinn Mathews" in reason and "0p" in reason


def test_established_starter_not_flagged() -> None:
    cfg = Config(thin_starter_gate=True, thin_starter_min_pitches=150)
    assert _pipeline(cfg)._thin_starter_reason("Kevin Gausman", 580) is None


def test_gate_disabled_returns_none() -> None:
    cfg = Config(thin_starter_gate=False, thin_starter_min_pitches=150)
    assert _pipeline(cfg)._thin_starter_reason("Quinn Mathews", 0) is None


def test_exactly_at_floor_is_allowed() -> None:
    cfg = Config(thin_starter_gate=True, thin_starter_min_pitches=150)
    assert _pipeline(cfg)._thin_starter_reason("Border Case", 150) is None


def _f5_ml_rec(cfg: Config, gate_reason: str | None):
    """Price one clearly +EV F5 ML selection through _mk with/without the gate.

    The buy is the *home* side: a road moneyline dog is refused outright on
    price, which would mask whatever this test is trying to say.
    """
    p = _mk_pipeline(cfg)
    game = SimpleNamespace(game_date="2026-08-01", game_pk=822781)
    # model 44% at +150 (devigged fair ~39%) is a Strong buy absent any gate: a
    # 5-point edge, inside the implausible-edge cap.
    quotes = {
        ("STL @ TOR", "f5_ml", "TOR F5 ML"): [
            MarketQuote(book="draftkings", american=150.0, opposite_american=-170.0)
        ]
    }
    return p._mk(
        game, "STL @ TOR", "f5", "f5_ml", "TOR F5 ML", 0.44,
        team_side="home", side="win", quotes=quotes, gate_reason=gate_reason,
    )


def test_gate_flips_a_would_be_buy_to_pass() -> None:
    cfg = Config(thin_starter_gate=True, thin_starter_min_pitches=150)
    # Sanity: without the gate this selection is a buy.
    assert _f5_ml_rec(cfg, None).tier in (Tier.STRONG, Tier.MODERATE)
    # With the thin-starter gate it is vetoed to Pass, reason recorded.
    gated = _f5_ml_rec(cfg, "thin Statcast: Quinn Mathews 0p < 150")
    assert gated.tier is Tier.PASS
    assert any("thin Statcast" in r for r in gated.reasons)
