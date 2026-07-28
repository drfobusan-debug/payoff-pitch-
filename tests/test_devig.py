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
    by_line = {k[2]: v[0] for k, v in quotes.items()}
    assert len(by_line) == 2
    half, one_half = (by_line[k] for k in sorted(by_line))
    assert half.opposite_american == 150.0
    assert one_half.opposite_american == -280.0


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
