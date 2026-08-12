"""Five batter markets come off the price-only list, each behind its own rule.

Pricing a market and betting it were deliberately separated (#100): the props
were bought from the odds API only to replace the phantom -110 the audit had
been grading them against, and a hard pass stopped that data decision becoming
a betting decision. The capture has now paid for itself, so the markets are
promoted -- but each keeps the screen its graded record earns, and ``batter_r``
stays shut because no slice of it has ever paid.
"""

from __future__ import annotations

import math

from mlb_engine.config import Config
from mlb_engine.data.oddsapi import PRICE_ONLY_MARKETS
from mlb_engine.features.market_gates import price_band_allows, prob_floor_allows

REOPENED = ("batter_1b", "batter_2b", "batter_hr", "batter_rbi", "batter_hrr")


# --- what is open and what is not ------------------------------------------
def test_the_measured_markets_are_bettable_again() -> None:
    for market in REOPENED:
        assert market not in PRICE_ONLY_MARKETS


def test_runs_stays_shut_because_no_rule_reopens_it() -> None:
    """-41.4 units, and every probability band and price bucket of it loses."""
    assert "batter_r" in PRICE_ONLY_MARKETS


def test_the_untested_markets_are_left_alone() -> None:
    assert {"batter_tb", "pitcher_er"} <= PRICE_ONLY_MARKETS


# --- singles: a price floor, not a model claim ------------------------------
def test_a_plus_money_single_is_bought_and_a_juiced_one_is_not() -> None:
    floor = Config().singles_min_buy_odds
    assert price_band_allows(180, floor, math.inf)[0]
    assert price_band_allows(100, floor, math.inf)[0]
    keep, reason = price_band_allows(-130, floor, math.inf)
    assert not keep
    assert "-130" in reason


def test_an_open_ended_floor_names_only_its_lower_bound() -> None:
    _, reason = price_band_allows(-130, 100.0, math.inf, "singles-price-floor")
    assert "+100 or better" in reason
    assert "inf" not in reason


# --- RBI: a conviction floor, blind to the payout ---------------------------
def test_a_cheap_rbi_ticket_is_refused_however_long_the_price() -> None:
    floor = Config().rbi_min_buy_prob
    assert prob_floor_allows(0.44, floor)[0]
    keep, reason = prob_floor_allows(0.28, floor)
    assert not keep
    assert "0.280" in reason


def test_a_zero_floor_disables_the_gate() -> None:
    assert prob_floor_allows(0.05, 0.0)[0]


def test_missing_inputs_never_create_a_betting_decision() -> None:
    assert prob_floor_allows(None, 0.40)[0]
    assert price_band_allows(None, 100.0, math.inf)[0]
