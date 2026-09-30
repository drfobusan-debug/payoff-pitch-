"""Which of two buys is Strong is the market's call, not our edge's.

Over 2,354 deduped graded buys the edge gap sorted the tiers backwards: Strong
went 47.7% for -9.9% ROI (n=1,435) against Moderate's 51.1% for -2.6% (n=919),
because the model's biggest departures from the price are its worst rows -- the
realized rate falls .623 -> .408 as the claimed edge grows. The devigged price
runs the other way over the same buys: 33.1% below .45 fair, 56.5% at .55-.60,
62.8% from .58 up, 65.9% at .65-.75.

Nothing here changes *whether* a row is bought; only which of the two buy tiers
it lands in.
"""

from __future__ import annotations

from mlb_engine.config import EVThresholds
from mlb_engine.market.ev import EVResult, MarketQuote
from mlb_engine.market.tiers import Tier, classify, price_screen


def _res(
    *,
    fair: float,
    edge: float = 0.05,
    devigged: bool = True,
    american: float = -110.0,
) -> EVResult:
    quote = MarketQuote(
        book="bk",
        american=american,
        opposite_american=-110.0 if devigged else None,
    )
    return EVResult(
        model_prob=min(fair + edge, 0.99),
        best_quote=quote,
        decimal=1.91,
        ev=0.05,
        fair_prob=fair,
        edge=edge,
        sharp_divergence=None,
        devig_coverage=1.0 if devigged else 0.0,
    )


def test_the_market_liking_it_is_what_makes_a_buy_strong() -> None:
    tier, reasons = classify(_res(fair=0.62), EVThresholds())
    assert tier is Tier.STRONG
    assert any("fair 0.620 >= 0.58 -> strong" in r for r in reasons)


def test_a_price_the_market_is_unsure_of_is_only_a_moderate_buy() -> None:
    """Even carrying the wider edge of the two, which is the whole point."""
    thr = EVThresholds()
    modest = classify(_res(fair=0.62, edge=0.02), thr)[0]
    wide = classify(_res(fair=0.52, edge=0.07), thr)[0]
    assert (modest, wide) == (Tier.STRONG, Tier.MODERATE)


def test_the_edge_gap_no_longer_promotes_anything_the_market_priced() -> None:
    """The old rule made this Strong on a .52 market at a 7-point edge."""
    thr = EVThresholds()
    assert 0.07 >= thr.min_edge + thr.strong_edge_gap
    assert classify(_res(fair=0.52, edge=0.07), thr)[0] is Tier.MODERATE


def test_a_one_way_quote_is_never_a_buy() -> None:
    """The edge on a one-way price is measured against the hold, not a fair number."""
    thr = EVThresholds()
    for edge in (0.07, 0.02):
        tier, reasons = classify(_res(fair=0.62, edge=edge, devigged=False), thr)
        assert tier is Tier.PASS
        assert any("one-way quote: PASS" in r for r in reasons)
    assert price_screen(_res(fair=0.62, edge=0.07, devigged=False), thr) is not None
    assert price_screen(_res(fair=0.62, edge=0.07, devigged=False), thr)[0] == "one_way_quote"


def test_the_two_sided_screen_runs_before_every_other_screen() -> None:
    """Named first so a one-way row is attributed to its price, not its edge."""
    res = _res(fair=0.62, edge=0.20, devigged=False)
    assert price_screen(res, EVThresholds())[0] == "one_way_quote"
    assert price_screen(res, EVThresholds(two_sided=False))[0] == "edge_ceiling"


def test_the_two_sided_screen_is_reversible_and_per_market(monkeypatch) -> None:
    """Off, the row keeps the rule fitted without a devigged price.

    A raw implied price overstates the market by about half the hold, so read
    as a fair probability it would promote one-sided longshots; the edge gap
    tiers it instead.
    """
    thr = EVThresholds(two_sided=False)
    tier, reasons = classify(_res(fair=0.62, edge=0.07, devigged=False), thr)
    assert tier is Tier.STRONG
    assert any("no devigged price" in r for r in reasons)
    assert classify(_res(fair=0.62, edge=0.02, devigged=False), thr)[0] is Tier.MODERATE

    monkeypatch.setenv("MLBE_TWO_SIDED_ONLY", "0")
    assert EVThresholds().two_sided is False
    monkeypatch.delenv("MLBE_TWO_SIDED_ONLY")
    monkeypatch.setenv("MLBE_TWO_SIDED_ONLY_BATTER_1B", "0")
    assert EVThresholds().two_sided is True
    assert EVThresholds().for_market("batter_1b").two_sided is False
    assert EVThresholds().for_market("game_ml").two_sided is True


def test_the_basis_is_reversible() -> None:
    thr = EVThresholds(strong_fair_prob=1.0)
    assert classify(_res(fair=0.52, edge=0.07), thr)[0] is Tier.STRONG
    assert classify(_res(fair=0.62, edge=0.02), thr)[0] is Tier.MODERATE


def test_the_tier_basis_does_not_decide_whether_to_buy() -> None:
    """The screens do that, and a demoted row is still a bet."""
    thr = EVThresholds()
    assert classify(_res(fair=0.52, edge=0.07), thr)[0] is not Tier.PASS
    # ... unless strict selection is on, where Moderate has always meant no bet.
    assert classify(_res(fair=0.52, edge=0.07), EVThresholds(strong_only=True))[0] is Tier.PASS
