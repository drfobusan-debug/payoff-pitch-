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
    base = EVThresholds(min_edge=0.02)
    assert base.for_market("batter_tb").min_edge == 0.05
    assert base.for_market("batter_hrr").min_edge == 0.04


def test_non_flagged_market_keeps_global_floor() -> None:
    base = EVThresholds(min_ev=0.0, min_edge=0.02)
    ml = base.for_market("game_ml")
    assert (ml.min_ev, ml.min_edge) == (0.0, 0.02)


def test_env_override_still_wins(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MIN_EDGE_BATTER_TB", "0.09")
    base = EVThresholds()
    assert base.for_market("batter_tb").min_edge == 0.09
    # unspecified fields still fall back to the global default
    assert base.for_market("batter_tb").max_edge == base.max_edge


def test_flagged_floor_never_loosens_a_tightened_global_guard() -> None:
    tight = EVThresholds(min_edge=0.06)
    assert tight.for_market("batter_tb").min_edge == 0.06  # not the 0.05 floor
    assert tight.for_market("game_ml").min_edge == 0.06


def test_raised_edge_floor_keeps_a_moderate_band() -> None:
    """A flagged market must not grade every surviving buy Strong."""
    tb = EVThresholds().for_market("batter_tb")
    assert tb.min_edge + tb.strong_edge_gap > tb.min_edge


def test_raised_floor_flips_a_marginal_buy_to_pass() -> None:
    q = MarketQuote(book="bk", american=-110)
    # A 4.5-point edge at a positive-EV price is a buy under the global floor...
    # The level sits above the conviction floor so the edge floor is what bites.
    res = EVResult(
        model_prob=0.625, best_quote=q, decimal=1.91, ev=0.194,
        fair_prob=0.58, edge=0.045, sharp_divergence=None,
    )
    base = EVThresholds()
    assert classify(res, base.for_market("game_ml"))[0] is Tier.STRONG
    # ...but Pass for batter_tb, which has to show 5 points.
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
    # Half the sims over any line at or below 8, so the model sits at 50% rather
    # than a certainty the implausible-edge cap would reject outright.
    res.pit["home"]["K"][: n // 2] = 4.0
    sel = keys.pitcher_prop(pitcher.name, "Ks", quote_line)
    quotes = {
        ("MATCH", "pitcher_k", sel): [
            MarketQuote(book="dk", american=120.0, opposite_american=-140.0)
        ]
    }
    recs = p._pitcher_props(game, "MATCH", res, "home", pitcher, quotes)
    return next(r for r in recs if r.selection == sel)


def test_high_k_line_is_gated_to_pass() -> None:
    # A 50% model at +120 is refused by the conviction floor and the EV ceiling
    # on shipped settings; the line cap is the screen under test.
    cfg = Config(ev=EVThresholds(min_prob=0.0, max_ev=1.0), pitcher_k_max_buy_line=5.5)
    rec = _pitcher_props(cfg, 6.5)  # o6.5 > cap
    assert rec.tier is Tier.PASS
    assert any("buy cap" in r for r in rec.reasons)


def test_low_k_line_is_not_gated() -> None:
    cfg = Config(ev=EVThresholds(min_prob=0.0, max_ev=1.0), pitcher_k_max_buy_line=5.5)
    rec = _pitcher_props(cfg, 5.5)  # o5.5 == cap, allowed
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)
    assert not any("buy cap" in r for r in rec.reasons)
