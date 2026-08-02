"""Tests for the over-confident-prop buy floors and the pitcher-K line cap.

The cumulative audit repeatedly flags a handful of counting-prop overs
(batter TB/R/H+R+RBI/1B and high pitcher-K lines) as false-positive pockets:
the model's edge is real on paper but the realized hit rate sits below the
break-even, so a global EV floor still lets marginal, unprofitable buys through.

Two orthogonal guards, both pure selection-tightening (no probability change):
  * per-market buy floors (raised strong/moderate/min_edge) for the leaky overs
  * a hard gate on pitcher-K lines above ``pitcher_k_max_buy_line``
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlb_engine.config import Config, EVThresholds
from mlb_engine.market import keys
from mlb_engine.market.ev import EVResult, MarketQuote
from mlb_engine.market.tiers import Tier, classify
from mlb_engine.pipeline import Pipeline


class _IdentityCalibrator:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _mk_pipeline(cfg: Config) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg
    p._calibrator = _IdentityCalibrator()
    p._shrink = None
    p._splits = {}
    return p


# ---- per-market buy floors -------------------------------------------------
def test_overbet_markets_get_a_raised_floor() -> None:
    base = EVThresholds(strong_buy=0.08, moderate_buy=0.03, min_edge=0.02)
    tb = base.for_market("batter_tb")
    assert (tb.strong_buy, tb.moderate_buy, tb.min_edge) == (0.12, 0.08, 0.05)
    hrr = base.for_market("batter_hrr")
    assert (hrr.strong_buy, hrr.moderate_buy, hrr.min_edge) == (0.10, 0.06, 0.04)


def test_non_flagged_market_keeps_global_floor() -> None:
    base = EVThresholds(strong_buy=0.08, moderate_buy=0.03, min_edge=0.02)
    ml = base.for_market("game_ml")
    assert (ml.strong_buy, ml.moderate_buy, ml.min_edge) == (0.08, 0.03, 0.02)


def test_env_override_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_EV_STRONG_BATTER_TB", "0.20")
    base = EVThresholds()
    assert base.for_market("batter_tb").strong_buy == 0.20
    # unspecified fields still fall back to the raised code floor
    assert base.for_market("batter_tb").moderate_buy == 0.08


def test_raised_floor_flips_a_marginal_buy_to_pass() -> None:
    q = MarketQuote(book="bk", american=-110)
    # EV +0.05 / edge +0.05 is a Moderate buy under the global floor...
    res = EVResult(
        model_prob=0.5, best_quote=q, decimal=1.91, ev=0.05,
        fair_prob=0.45, edge=0.05, sharp_divergence=None,
    )
    base = EVThresholds()
    assert classify(res, base.for_market("game_ml"))[0] is Tier.MODERATE
    # ...but Pass for batter_tb, whose moderate floor is 0.08.
    assert classify(res, base.for_market("batter_tb"))[0] is Tier.PASS


# ---- pitcher-K line cap ----------------------------------------------------
def _pitcher_props(cfg: Config, quote_line: float):
    p = _mk_pipeline(cfg)
    game = SimpleNamespace(game_date="2026-08-01", game_pk=1)
    pitcher = SimpleNamespace(name="Some Pitcher", mlbam_id=42)
    n = 200
    res = SimpleNamespace(
        pit={"home": {k: np.full(n, 8.0) for k in ("K", "outs", "H", "BB", "ER")}}
    )
    sel = keys.pitcher_prop(pitcher.name, "Ks", quote_line)
    quotes = {
        ("MATCH", "pitcher_k", sel): [MarketQuote(book="dk", american=120.0)]
    }
    recs = p._pitcher_props(game, "MATCH", res, "home", pitcher, quotes)
    return next(r for r in recs if r.selection == sel)


def test_high_k_line_is_gated_to_pass() -> None:
    cfg = Config(pitcher_k_max_buy_line=5.5)
    rec = _pitcher_props(cfg, 6.5)  # o6.5 > cap
    assert rec.tier is Tier.PASS
    assert any("buy cap" in r for r in rec.reasons)


def test_low_k_line_is_not_gated() -> None:
    cfg = Config(pitcher_k_max_buy_line=5.5)
    rec = _pitcher_props(cfg, 5.5)  # o5.5 == cap, allowed
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)
    assert not any("buy cap" in r for r in rec.reasons)
