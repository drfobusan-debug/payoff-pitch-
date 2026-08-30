"""The buy list is ordered by conviction and price, not by edge or EV."""

from __future__ import annotations

from mlb_engine.market.ranking import (
    LONGSHOT_AMERICAN,
    bet_sort_key,
    decimal_odds,
    fair_decimal,
    price_rank,
)


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


# --- the devigged price, which is the one the ledger orders on -------------


def test_the_hold_comes_out_before_two_markets_are_compared() -> None:
    """Two -140s are not the same bet when one book keeps twice the hold.

    A raw price cannot order a 3% prop market against a 9% one, and it is the
    market's true probability that tracks the money (33.1% wins below .45 fair,
    62.8% at .60-.65), so the ranking reads the devigged number.
    """
    tight = fair_decimal(-140, 0.573)  # -140/+120: hold comes out at .573
    fat = fair_decimal(-140, 0.535)  # -140/+105: the same price, more juice
    assert tight < fat


def test_a_row_nothing_devigged_keeps_its_posted_price() -> None:
    """An unknown hold is not a zero hold, but the posted price still ranks."""
    assert fair_decimal(-140, None) == decimal_odds(-140)
    assert fair_decimal(-140, 0.0) == decimal_odds(-140)


def test_the_devigged_scale_shares_the_longshot_bucket() -> None:
    assert fair_decimal(None, 0.05) == fair_decimal(None, 0.10)
    assert fair_decimal(None, 0.05) == decimal_odds(LONGSHOT_AMERICAN)


def test_the_devigged_price_beats_the_posted_one_at_ordering_a_pair() -> None:
    """The row the market thinks more of leads, whichever row we price higher."""
    market_likes_it = price_rank(-120, 0.60, 0.01)
    we_like_it = price_rank(-120, 0.44, 0.30)
    assert market_likes_it < we_like_it


def test_a_row_with_no_price_at_all_sorts_behind_every_priced_row() -> None:
    """And behind them in its own order rather than as if it were even money."""
    unpriced_good = price_rank(None, None, 0.30)
    unpriced_bad = price_rank(None, None, 0.01)
    assert price_rank(400, None, 0.9) < unpriced_good < unpriced_bad


def test_the_sort_key_reads_the_devigged_price_when_it_has_one() -> None:
    juiced = bet_sort_key(strong=True, american=-140, edge=0.05, fair_prob=0.52)
    clean = bet_sort_key(strong=True, american=-140, edge=0.02, fair_prob=0.60)
    assert clean < juiced, "the wider edge does not lead the shorter true price"
