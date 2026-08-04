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


def test_classify_tiers_by_ev():
    thr = EVThresholds()
    strong = evaluate(0.80, [MarketQuote("pinnacle", +100, -120)])
    tier, reasons = classify(strong, thr)
    assert tier == Tier.STRONG
    assert reasons

    flat = evaluate(0.50, [MarketQuote("pinnacle", -110, -110)])
    tier2, _ = classify(flat, thr)
    assert tier2 == Tier.PASS
