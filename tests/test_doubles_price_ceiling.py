"""Doubles overs are refused at +300 and longer.

The doubles book is calibrated everywhere except where it bets: across 6,656
graded ``batter_2b o0.5`` rows the model's .140 band hits 14.0% and its .188
band 17.7%, but its .258 band -- the one the buy list is drawn from -- hits
15.0% on 346 rows. The selection consequently adds nothing, with bought rows at
14.3% (n=70) against passed rows' 14.2% (n=6,586), so the screen is a price
ceiling rather than a band: no price pocket survived, and at these lengths a
fraction of a point of probability error is a fifth of the stake.
"""

from __future__ import annotations

from mlb_engine.config import Config
from mlb_engine.features.market_gates import price_ceiling_allows

REFUSE_AT = 300.0


def test_a_long_doubles_price_is_refused() -> None:
    keep, reason = price_ceiling_allows(455, REFUSE_AT, "doubles-price-ceiling")
    assert not keep
    assert "+455" in reason


def test_the_ceiling_is_exclusive() -> None:
    """'+300 or longer' is the rule, so +300 itself is refused."""
    assert not price_ceiling_allows(300, REFUSE_AT)[0]
    assert price_ceiling_allows(299, REFUSE_AT)[0]


def test_a_short_doubles_price_is_still_buyable() -> None:
    """The door is left open: a hitter the book prices near even is not this bet."""
    assert price_ceiling_allows(150, REFUSE_AT)[0]


def test_an_unpriced_selection_is_left_to_the_other_screens() -> None:
    keep, reason = price_ceiling_allows(None, REFUSE_AT)
    assert keep
    assert reason == ""


def test_the_shipped_default_is_the_measured_cutoff() -> None:
    assert Config().doubles_max_buy_odds == REFUSE_AT


def test_the_screen_can_be_lifted_from_the_environment(monkeypatch) -> None:
    """A screen the ledger condemns has to be reversible without a release."""
    monkeypatch.setenv("MLBE_DOUBLES_MAX_BUY_ODDS", "100000")
    assert price_ceiling_allows(650, Config().doubles_max_buy_odds)[0]
