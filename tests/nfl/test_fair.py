"""De-vig arithmetic, and the guards against inventing a price."""

from __future__ import annotations

import math

import pytest

from nfl_engine.market.board import MarketQuote
from nfl_engine.market.fair import (
    METHODS,
    devig,
    fair_from_quotes,
    hold,
)
from nfl_engine.market.odds import american_to_prob


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_returns_a_probability_distribution(method: str) -> None:
    implied = [american_to_prob(-140), american_to_prob(120)]
    fair = devig(implied, method)
    assert math.isclose(sum(fair), 1.0, abs_tol=1e-9)
    assert all(0.0 < p < 1.0 for p in fair)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_a_fair_market_is_left_alone(method: str) -> None:
    fair = devig([0.5, 0.5], method)
    assert fair == pytest.approx([0.5, 0.5], abs=1e-9)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_devig_lowers_both_sides(method: str) -> None:
    implied = [american_to_prob(-110), american_to_prob(-110)]
    assert hold(implied) > 0
    fair = devig(implied, method)
    assert all(f < p for f, p in zip(fair, implied, strict=True))


def test_proportional_shaves_the_favourite_harder_than_power() -> None:
    """The measured bias: proportional understates heavy favourites.

    2006-2025 closing moneylines, favourites booked above 0.9 realised 0.9185;
    proportional called them 0.8954 (-2.3pp) and power 0.9152 (-0.3pp). The
    ordering of the two methods on a single heavy favourite is what produces that
    table, so it is asserted directly.
    """
    implied = [american_to_prob(-1200), american_to_prob(750)]
    assert devig(implied, "proportional")[0] < devig(implied, "power")[0]


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown de-vig"):
        devig([0.5, 0.5], "vibes")


def test_unpaired_quotes_are_never_priced() -> None:
    fair = fair_from_quotes([MarketQuote(book="dk", american=-110)])
    assert fair is not None
    assert fair.paired_books == 0
    assert not fair.is_trustworthy()
    assert fair.prob == 0.0


def test_one_paired_book_is_not_a_consensus() -> None:
    fair = fair_from_quotes(
        [MarketQuote(book="dk", american=-110, opposite_american=-110)]
    )
    assert fair is not None
    assert fair.paired_books == 1
    assert not fair.is_trustworthy()
    assert fair.prob == pytest.approx(0.5, abs=1e-9)


def test_sharp_book_carries_more_weight() -> None:
    """Pinnacle is weighted 2.0, so the blend sits nearer its number."""
    quotes = [
        MarketQuote(book="pinnacle", american=-150, opposite_american=130),
        MarketQuote(book="fanduel", american=-110, opposite_american=-110),
    ]
    fair = fair_from_quotes(quotes)
    assert fair is not None
    assert fair.paired_books == 2
    assert fair.is_trustworthy()
    unweighted = (0.5 + devig(
        [american_to_prob(-150), american_to_prob(130)], "power"
    )[0]) / 2
    assert fair.prob > unweighted


def test_median_hold_is_reported() -> None:
    quotes = [
        MarketQuote(book="dk", american=-110, opposite_american=-110),
        MarketQuote(book="mgm", american=-120, opposite_american=100),
    ]
    fair = fair_from_quotes(quotes)
    assert fair is not None
    assert 0.0 < fair.median_hold < 0.1
