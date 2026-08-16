"""Two-sided devigging of market prices.

The fair probability is what the buy screen measures the model against, so a
one-sided implied probability quietly loads half the book's hold onto the bar:
~3.4 points on a player prop. These tests pin the arithmetic and the pairing.
"""

from __future__ import annotations

from mlb_engine.data.oddsapi import OddsAPIClient, _Event, _opposite_prices
from mlb_engine.market.ev import MarketQuote, evaluate


def _event() -> _Event:
    return _Event("evt1", "NYY", "BOS", "new york yankees", "boston red sox")


def test_no_vig_prob_normalises_the_pair() -> None:
    over = MarketQuote(book="draftkings", american=-196.0, opposite_american=150.0)
    raw = 196 / 296  # 0.662 -- the price with the vig still in it
    assert abs(raw - 0.6622) < 1e-3
    assert abs(over.no_vig_prob - 0.6231) < 1e-3
    assert over.no_vig_prob < raw
    assert over.devigged


def test_unpaired_quote_keeps_the_raw_implied_probability() -> None:
    q = MarketQuote(book="circa", american=-110.0)
    assert abs(q.no_vig_prob - 110 / 210) < 1e-9
    assert not q.devigged


def test_evaluate_reports_devig_coverage() -> None:
    quotes = [
        MarketQuote(book="draftkings", american=-150.0, opposite_american=130.0),
        MarketQuote(book="circa", american=-140.0),
    ]
    res = evaluate(0.60, quotes)
    # Circa carries weight 2.0, so one devigged book out of three weight units.
    assert abs(res.devig_coverage - 1 / 3) < 1e-9
    assert res.edge > 0.60 - (150 / 250)  # edge is bigger once the vig is off


def test_consensus_ignores_a_quote_that_still_carries_its_vig() -> None:
    """One undevigged book pushed *both* sides of a market above their true
    probability, which is how a moneyline could lose CLV whichever side we bet.
    """
    paired = MarketQuote(book="draftkings", american=-150.0, opposite_american=130.0)
    raw = MarketQuote(book="circa", american=-140.0)
    assert evaluate(0.60, [paired, raw]).fair_prob == paired.no_vig_prob
    # Line shopping is untouched: the raw quote can still be the price we take.
    assert evaluate(0.60, [paired, raw]).best_quote.book == "circa"


def test_both_sides_of_a_devigged_market_sum_to_one() -> None:
    home = [
        MarketQuote(book="draftkings", american=-150.0, opposite_american=130.0),
        MarketQuote(book="circa", american=-145.0, opposite_american=133.0),
    ]
    away = [
        MarketQuote(book="draftkings", american=130.0, opposite_american=-150.0),
        MarketQuote(book="circa", american=133.0, opposite_american=-145.0),
    ]
    total = evaluate(0.5, home).fair_prob + evaluate(0.5, away).fair_prob
    assert abs(total - 1.0) < 1e-9


def test_consensus_falls_back_when_no_quote_can_be_devigged() -> None:
    quotes = [
        MarketQuote(book="draftkings", american=120.0),
        MarketQuote(book="circa", american=-110.0),
    ]
    res = evaluate(0.5, quotes)
    weighted = (1.0 * quotes[0].no_vig_prob + 2.0 * quotes[1].no_vig_prob) / 3.0
    assert abs(res.fair_prob - weighted) < 1e-9
    assert res.devig_coverage == 0.0


def test_props_pair_over_with_its_own_under() -> None:
    client = OddsAPIClient("k")
    quotes: dict = {}
    client._parse_props(
        {
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "batter_hits",
                            "outcomes": [
                                {"name": "Over", "description": "Aaron Judge",
                                 "point": 0.5, "price": -196},
                                {"name": "Under", "description": "Aaron Judge",
                                 "point": 0.5, "price": 150},
                                {"name": "Over", "description": "Aaron Judge",
                                 "point": 1.5, "price": 220},
                                {"name": "Under", "description": "Aaron Judge",
                                 "point": 1.5, "price": -280},
                            ],
                        }
                    ],
                }
            ]
        },
        _event(),
        quotes,
    )
    by_sel = {k[2]: v[0] for k, v in quotes.items()}
    # Both sides of both lines are priced, and each pairs against its own
    # opposite rather than against the other line.
    assert sorted(by_sel) == [
        "Aaron Judge H o0.5", "Aaron Judge H o1.5",
        "Aaron Judge H u0.5", "Aaron Judge H u1.5",
    ]
    assert by_sel["Aaron Judge H o0.5"].american == -196.0
    assert by_sel["Aaron Judge H o0.5"].opposite_american == 150.0
    assert by_sel["Aaron Judge H u0.5"].american == 150.0
    assert by_sel["Aaron Judge H u0.5"].opposite_american == -196.0
    assert by_sel["Aaron Judge H o1.5"].opposite_american == -280.0
    assert by_sel["Aaron Judge H u1.5"].opposite_american == 220.0
    # The two sides of one line agree on a single fair probability.
    o, u = by_sel["Aaron Judge H o0.5"], by_sel["Aaron Judge H u0.5"]
    assert abs(o.no_vig_prob + u.no_vig_prob - 1.0) < 1e-9


def test_run_line_pairs_across_different_points() -> None:
    """The two sides of a spread carry -1.5 and +1.5, so grouping on the point
    alone would never pair them."""
    client = OddsAPIClient("k")
    quotes: dict = {}
    client._parse_game(
        {
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "New York Yankees", "point": -1.5, "price": 120},
                                {"name": "Boston Red Sox", "point": 1.5, "price": -140},
                            ],
                        }
                    ],
                }
            ]
        },
        _event(),
        quotes,
        f5=False,
    )
    quote = next(iter(quotes.values()))[0]
    assert quote.devigged
    assert 0 < quote.no_vig_prob < 1


def test_three_way_group_is_left_alone() -> None:
    outcomes = [
        {"name": "Over", "description": "A", "point": 0.5, "price": -110},
        {"name": "Under", "description": "A", "point": 0.5, "price": -110},
        {"name": "Push", "description": "A", "point": 0.5, "price": 500},
    ]
    assert _opposite_prices(outcomes) == {}


def test_missing_price_does_not_form_a_pair() -> None:
    outcomes = [
        {"name": "Over", "description": "A", "point": 0.5, "price": -110},
        {"name": "Under", "description": "A", "point": 0.5, "price": None},
    ]
    assert _opposite_prices(outcomes) == {}
