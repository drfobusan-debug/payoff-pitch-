"""Odds conversion, devig, and EV/tier screening."""

from __future__ import annotations

import math

from cfb_engine.config import EVThresholds
from cfb_engine.market.ev import MarketQuote, ev_per_dollar, evaluate
from cfb_engine.market.odds import (
    american_to_decimal,
    american_to_prob,
    no_vig_two_way,
    prob_to_american,
    remove_vig,
)
from cfb_engine.market.tiers import Tier, classify


def test_american_to_decimal_both_signs():
    assert math.isclose(american_to_decimal(-110), 1.9090909, rel_tol=1e-6)
    assert math.isclose(american_to_decimal(+150), 2.5, rel_tol=1e-9)


def test_american_prob_roundtrip():
    for p in (0.25, 0.5, 0.73):
        assert math.isclose(american_to_prob(prob_to_american(p)), p, rel_tol=1e-6)


def test_remove_vig_sums_to_one():
    probs = remove_vig([american_to_prob(-110), american_to_prob(-110)])
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
    assert math.isclose(probs[0], 0.5, rel_tol=1e-9)


def test_no_vig_two_way_favorite():
    fav, dog = no_vig_two_way(-200, +170)
    assert fav > dog
    assert math.isclose(fav + dog, 1.0, rel_tol=1e-9)


def test_ev_per_dollar_fair_bet_is_zero():
    # A true 50% at +100 is a break-even bet.
    assert math.isclose(ev_per_dollar(0.5, 100), 0.0, abs_tol=1e-9)


def test_evaluate_devigs_and_edges():
    quotes = [
        MarketQuote("pinnacle", -110, opposite_american=-110),
        MarketQuote("draftkings", +100, opposite_american=-120),
    ]
    res = evaluate(0.60, quotes)
    # best price is the +100 (higher decimal payout)
    assert res.best_quote.american == 100
    assert 0.45 < res.fair_prob < 0.55
    assert res.edge > 0
    assert res.devig_coverage == 1.0


def test_classify_ranks_on_edge_not_ev():
    thr = EVThresholds()
    # Comfortably inside the edge cap: 4.5 points over the devigged price.
    strong = evaluate(0.523, [MarketQuote("pinnacle", +100, -120)])
    tier, reasons = classify(strong, thr)
    assert tier == Tier.STRONG
    assert reasons

    flat = evaluate(0.50, [MarketQuote("pinnacle", -110, -110)])
    tier2, _ = classify(flat, thr)
    assert tier2 == Tier.PASS


def test_a_long_price_no_longer_reaches_strong_on_a_thin_edge():
    """``EV = decimal x edge``, so an EV cutoff was a cheaper bar on long prices.

    A +300 dog cleared the old 0.06 Strong threshold on 1.5 points of
    disagreement while a -400 favorite needed 4.8. Tiering on edge charges both
    the same, so this dog lands one tier lower than the EV rule put it.
    """
    thr = EVThresholds()
    dog = evaluate(0.2675, [MarketQuote("pinnacle", +300, -400)])
    assert dog.ev > 0.06  # would have been Strong under the EV rule
    tier, _ = classify(dog, thr)
    assert tier == Tier.MODERATE


def test_an_implausible_edge_is_a_model_error_not_a_bigger_bet():
    thr = EVThresholds()
    wild = evaluate(0.80, [MarketQuote("pinnacle", +100, -120)])
    assert wild.edge > thr.max_edge
    tier, reasons = classify(wild, thr)
    assert tier == Tier.PASS
    assert any("> 0.08" in r for r in reasons)
