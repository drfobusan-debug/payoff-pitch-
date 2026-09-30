"""Card ordering: what the price does to the running order."""

from __future__ import annotations

from datetime import date

import pytest

from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.ordering import (
    EDGE,
    EV,
    KELLY,
    conviction,
    order_buys,
    order_mode,
    order_recs,
)
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation

DAY = date(2025, 11, 1)


def _rec(
    selection: str,
    american: float,
    edge: float,
    tier: Tier = Tier.STRONG,
    fair_prob: float = 0.5,
) -> Recommendation:
    """A row priced consistently: EV is what the edge actually earns at that price."""
    prob = fair_prob + edge
    ev = prob * (american_to_decimal(american) - 1.0) - (1.0 - prob)
    return Recommendation(
        game_date=DAY,
        game_id="g1",
        matchup="Alabama vs Georgia",
        market="game_ml",
        selection=selection,
        model_prob=prob,
        market_american=american,
        ev=ev,
        edge=edge,
        fair_prob=fair_prob,
        tier=tier,
    )


def test_the_same_edge_at_a_longer_price_carries_more_ev_and_less_kelly() -> None:
    """The arithmetic the ordering exists to answer: EV rises with price length
    because ``EV = (decimal - 1) x p - q``, so ranking on it ranks on price."""
    short = _rec("favourite", -200.0, 0.03, fair_prob=0.66)
    long_ = _rec("dog", 400.0, 0.03, fair_prob=0.20)
    assert (long_.ev or 0.0) > (short.ev or 0.0)
    assert conviction(short, KELLY) > conviction(long_, KELLY)
    assert order_recs([long_, short], KELLY)[0] is short
    assert order_recs([short, long_], EV)[0] is long_


def test_edge_ordering_is_price_blind() -> None:
    short = _rec("favourite", -200.0, 0.03, fair_prob=0.66)
    long_ = _rec("dog", 400.0, 0.04, fair_prob=0.20)
    assert order_recs([short, long_], EDGE)[0] is long_


def test_kelly_is_the_growth_optimal_fraction_of_the_price() -> None:
    rec = _rec("dog", 150.0, 0.05, fair_prob=0.40)
    kelly = rec.kelly
    assert kelly is not None
    # f* = EV / (decimal - 1) = (p - breakeven) / (1 - breakeven).
    breakeven = 1.0 / american_to_decimal(150.0)
    assert kelly == pytest.approx((rec.model_prob - breakeven) / (1.0 - breakeven))


def test_an_unpriced_buy_falls_back_to_its_edge_rather_than_sinking() -> None:
    priced = _rec("priced", -110.0, 0.02)
    unpriced = _rec("unpriced", -110.0, 0.09)
    unpriced.market_american = None
    unpriced.ev = None
    assert unpriced.kelly is None
    assert order_recs([priced, unpriced], KELLY)[0] is unpriced


def test_tiers_outrank_conviction_and_passes_sort_last() -> None:
    """A Moderate buy never leads a card over a Strong one, whatever it pays."""
    strong = _rec("strong", -110.0, 0.02)
    moderate = _rec("moderate", -110.0, 0.08, tier=Tier.MODERATE)
    passed = _rec("passed", -110.0, 0.08, tier=Tier.PASS)
    assert [r.selection for r in order_recs([passed, moderate, strong])] == [
        "strong",
        "moderate",
        "passed",
    ]
    assert [r.selection for r in order_buys([passed, moderate, strong])] == [
        "strong",
        "moderate",
    ]


def test_a_pass_is_ranked_on_the_model_disagreeing_with_the_side() -> None:
    """A refused row has no stake to size, so the fades lead with the strongest
    disagreement rather than the longest price."""
    mild = _rec("mild", 500.0, 0.0, tier=Tier.PASS, fair_prob=0.20)
    mild.model_prob = 0.18
    strong = _rec("strong-fade", -110.0, 0.0, tier=Tier.PASS, fair_prob=0.50)
    strong.model_prob = 0.30
    assert order_recs([mild, strong])[0] is strong


def test_the_mode_comes_from_the_environment_and_ignores_nonsense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CFBE_CARD_ORDER", "EV")
    assert order_mode() == EV
    monkeypatch.setenv("CFBE_CARD_ORDER", "sharpe")
    assert order_mode() == KELLY
    monkeypatch.delenv("CFBE_CARD_ORDER")
    assert order_mode() == KELLY
