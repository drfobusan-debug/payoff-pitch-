"""The buy list is ordered by conviction and price, not by edge or EV."""

from __future__ import annotations

from mlb_engine.market.ranking import LONGSHOT_AMERICAN, bet_sort_key, decimal_odds


def test_decimal_odds_reads_both_sides_of_even_money() -> None:
    assert decimal_odds(-200) == 1.5
    assert decimal_odds(100) == 2.0
    assert decimal_odds(150) == 2.5


def test_an_unpriced_row_sorts_last() -> None:
    assert decimal_odds(None) == float("inf")


def test_longshots_share_one_bucket() -> None:
    """Past +300 the graded cells are 15 bets, so they are not ordered."""
    assert decimal_odds(400) == decimal_odds(LONGSHOT_AMERICAN)
    assert decimal_odds(299) < decimal_odds(400)


def test_conviction_outranks_price() -> None:
    strong_long = bet_sort_key(strong=True, american=250, edge=0.01)
    moderate_short = bet_sort_key(strong=False, american=-300, edge=0.07)
    assert strong_long < moderate_short


def test_the_shorter_price_leads_within_a_tier() -> None:
    short = bet_sort_key(strong=True, american=-250, edge=0.02)
    long_ = bet_sort_key(strong=True, american=180, edge=0.079)
    assert short < long_


def test_edge_only_breaks_a_tie() -> None:
    """The ceiling-hugging row no longer reaches the top of the page."""
    same_price_thin = bet_sort_key(strong=True, american=-150, edge=0.03)
    same_price_wide = bet_sort_key(strong=True, american=-150, edge=0.079)
    assert same_price_wide < same_price_thin

    wide_but_long = bet_sort_key(strong=True, american=500, edge=0.08)
    thin_but_short = bet_sort_key(strong=True, american=-150, edge=0.021)
    assert thin_but_short < wide_but_long


def test_ordering_a_slate_puts_the_longest_prices_last() -> None:
    rows = [
        ("longshot", True, 640, 0.077),
        ("chalk", True, -620, 0.080),
        ("moderate chalk", False, -400, 0.079),
        ("even", True, 100, 0.078),
    ]
    order = [
        name
        for name, strong, odds, edge in sorted(
            rows, key=lambda r: bet_sort_key(strong=r[1], american=r[2], edge=r[3])
        )
    ]
    assert order == ["chalk", "even", "longshot", "moderate chalk"]
