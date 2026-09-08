"""The edge ceiling is relaxed on strikeouts and left alone everywhere else.

``edge_ceiling`` refuses the picks the model likes most, so it can only be
judged on its own refused rows. On ``pitcher_k`` they won: 57 graded refusals
went 56.1% against a 53.8% breakeven (+7.9% ROI), and the profit is at the far
end -- 8 to 20 points of disagreement is flat (+1.3%, n=28) while 25 to 35
points went 64.3% (n=14). ``pitcher_outs`` is the control, where the same screen
removed real losers (35.0% against a 49.1% breakeven, -32.0%, n=40).
"""

from __future__ import annotations

from mlb_engine.config import EVThresholds
from mlb_engine.market.ev import EVResult, MarketQuote
from mlb_engine.market.tiers import Tier, classify, price_screen

GLOBAL_MAX_EDGE = 0.08
K_MAX_EDGE = 0.30


def _result(edge: float) -> EVResult:
    """A priced selection whose edge over the no-vig market is ``edge``."""
    fair = 0.50
    return EVResult(
        model_prob=fair + edge,
        best_quote=MarketQuote(book="dk", american=-110.0, opposite_american=-110.0),
        decimal=1.909,
        ev=0.10,
        fair_prob=fair,
        edge=edge,
        sharp_divergence=None,
    )


def test_strikeouts_keep_a_wider_ceiling_than_the_rest_of_the_sheet() -> None:
    assert EVThresholds().for_market("pitcher_k").max_edge == K_MAX_EDGE
    assert EVThresholds().max_edge == GLOBAL_MAX_EDGE


def test_outs_are_the_control_and_keep_the_global_ceiling() -> None:
    """Its refusals lost, so nothing about them argues for relaxing it."""
    assert EVThresholds().for_market("pitcher_outs").max_edge == GLOBAL_MAX_EDGE
    assert price_screen(_result(0.12), EVThresholds().for_market("pitcher_outs")) == (
        "edge_ceiling",
        "edge +0.120 > 0.08 -> pass",
    )


def test_a_strikeout_edge_the_old_ceiling_refused_now_buys() -> None:
    thr = EVThresholds().for_market("pitcher_k")
    assert price_screen(_result(0.25), thr) is None
    tier, _ = classify(_result(0.25), thr)
    assert tier is not Tier.PASS


def test_the_relaxed_ceiling_still_has_a_far_end() -> None:
    """Past a third of a probability the model is reading a start it cannot see."""
    thr = EVThresholds().for_market("pitcher_k")
    gate, reason = price_screen(_result(0.35), thr) or ("", "")
    assert gate == "edge_ceiling"
    assert "0.3" in reason


def test_the_relaxed_ceiling_is_overridable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MAX_EDGE_PITCHER_K", "0.08")
    assert EVThresholds().for_market("pitcher_k").max_edge == GLOBAL_MAX_EDGE


def test_no_other_market_moved() -> None:
    for market in ("batter_h", "batter_2b", "pitcher_h", "pitcher_bb", "game_total"):
        assert EVThresholds().for_market(market).max_edge == GLOBAL_MAX_EDGE
