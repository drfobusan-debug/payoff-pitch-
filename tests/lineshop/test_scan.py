import pytest

from lineshop import scan as sc
from lineshop.feed import Game, Quote, parse, restrict


def game(quotes: dict[tuple[str, str], list[Quote]]) -> Game:
    return Game(
        sport="nfl",
        game_id="g1",
        home="Baltimore Ravens",
        away="Los Angeles Chargers",
        commence="2026-09-10T00:20:00Z",
        quotes=quotes,
    )


def spread_game() -> Game:
    return game(
        {
            ("spreads", "Baltimore Ravens"): [
                Quote("DraftKings", -110, -3.0),
                Quote("FanDuel", -115, -2.5),
                Quote("BetMGM", -108, -3.5),
            ],
            ("spreads", "Los Angeles Chargers"): [
                Quote("DraftKings", -110, 3.0),
                Quote("FanDuel", -105, 2.5),
                Quote("BetMGM", -112, 3.5),
            ],
        }
    )


def test_best_offer_takes_the_number_before_the_price():
    g = spread_game()
    home = sc.best_offer(g, "spreads", "Baltimore Ravens")
    away = sc.best_offer(g, "spreads", "Los Angeles Chargers")
    assert (home.point, home.books) == (-2.5, ("FanDuel",))
    assert (away.point, away.books) == (3.5, ("BetMGM",))


def test_over_wants_the_lowest_line_and_under_the_highest():
    g = game(
        {
            ("totals", "Over"): [Quote("DraftKings", -110, 44.5), Quote("FanDuel", -110, 43.5)],
            ("totals", "Under"): [Quote("DraftKings", -110, 44.5), Quote("FanDuel", -110, 45.5)],
        }
    )
    assert sc.best_offer(g, "totals", "Over").point == 43.5
    assert sc.best_offer(g, "totals", "Under").point == 45.5


def test_sides_are_expressed_against_the_home_margin_axis():
    g = spread_game()
    # Home -3.5 needs the home team to win by 4 or more; the away side of the
    # same number wins below it, and a moneyline is the same question at zero.
    assert sc._threshold(g, "spreads", g.home, -3.5) == (sc.ABOVE, 3.5)
    assert sc._threshold(g, "spreads", g.away, 3.5) == (sc.BELOW, 3.5)
    assert sc._threshold(g, "h2h", g.home, None) == (sc.ABOVE, 0.0)
    assert sc._threshold(g, "totals", "Under", 44.5) == (sc.BELOW, 44.5)


def test_a_push_counts_half_so_the_hook_is_worth_something():
    g = spread_game()
    on_the_number = sc.side_probability("nfl", g, "spreads", g.away, 3.0, line=3.0)
    over_the_hook = sc.side_probability("nfl", g, "spreads", g.away, 3.5, line=3.0)
    # +3.5 wins the games that push at +3, so it must be worth about half that
    # push again -- and 3 is the biggest push in the sport.
    assert over_the_hook.p - on_the_number.p > 0.04


def test_crossing_nets_the_number_against_the_price():
    g = spread_game()
    found = {c.side: c for c in sc.crossings("nfl", g)}
    away = found["Los Angeles Chargers"]
    assert away.best.point == 3.5
    assert away.consensus_point == 3.0
    assert away.edge == pytest.approx(away.prob_gain - away.price_cost)
    assert away.prob_gain > 0
    assert 3 in away.keys


def test_a_worse_price_can_eat_the_extra_half_point():
    g = game(
        {
            ("spreads", "Baltimore Ravens"): [Quote("DraftKings", -110, -3.0)],
            ("spreads", "Los Angeles Chargers"): [
                Quote("DraftKings", -110, 3.0),
                Quote("BetMGM", -160, 3.5),
            ],
        }
    )
    # +3.5 buys ~4.7% of win probability; -160 over -110 costs ~9% of it.
    assert not [c for c in sc.crossings("nfl", g) if c.best.american == -160]


def test_middle_ev_pays_the_window_and_loses_only_one_leg_outside_it():
    g = game(
        {
            ("totals", "Over"): [Quote("DraftKings", -110, 41.5)],
            ("totals", "Under"): [Quote("FanDuel", -110, 44.5)],
        }
    )
    (middle,) = sc.middles("nfl", g)
    assert middle.low.point == 41.5 and middle.high.point == 44.5
    hit = middle.p_middle
    # One unit each side: the window pays both, everything else pays one and
    # loses one, so EV per unit staked is that arithmetic and nothing else.
    expected = (hit * (0.909 + 0.909) + (1 - hit) * (0.909 - 1)) / 2
    assert middle.ev == pytest.approx(expected, abs=0.002)
    assert hit > 0.05


def test_numbers_that_do_not_overlap_are_not_a_middle():
    g = game(
        {
            ("totals", "Over"): [Quote("DraftKings", -110, 44.5)],
            ("totals", "Under"): [Quote("FanDuel", -110, 43.5)],
        }
    )
    assert sc.middles("nfl", g) == []


def test_a_push_middle_is_priced_off_the_push_not_the_window():
    # -3 with +4 has no integer between the numbers, but a 3-point home win
    # pushes one leg while the other cashes, which is most of what 3 is worth.
    g = game(
        {
            ("spreads", "Baltimore Ravens"): [Quote("DraftKings", -110, -3.0)],
            ("spreads", "Los Angeles Chargers"): [Quote("FanDuel", -110, 4.0)],
        }
    )
    (middle,) = sc.middles("nfl", g)
    assert middle.p_middle == pytest.approx(0.0, abs=1e-9)
    assert middle.p_push > 0.10
    assert middle.ev > 0


def test_a_scan_only_shops_the_books_the_operator_holds():
    g = spread_game()
    (scanned,) = sc.scan("nfl", [g], books=("DraftKings", "FanDuel"))
    # BetMGM has the best number on both sides; without an account there it is
    # not an edge, and the consensus still comes off the whole board.
    assert scanned.best[("spreads", "Los Angeles Chargers")].point == 3.0
    assert scanned.consensus_spread == -3.0


def test_a_board_nobody_has_priced_is_not_a_disagreement():
    g = game({("spreads", "Baltimore Ravens"): [Quote("DraftKings", -110, -3.0)]})
    assert sc.scan("nfl", [g]) == []


def test_parse_keeps_every_rung_and_every_book():
    payload = [
        {
            "id": "abc",
            "home_team": "Baltimore Ravens",
            "away_team": "Los Angeles Chargers",
            "commence_time": "2026-09-10T00:20:00Z",
            "bookmakers": [
                {
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Baltimore Ravens", "price": -110, "point": -3.0},
                                {"name": "Los Angeles Chargers", "price": -110, "point": 3.0},
                            ],
                        },
                        {
                            "key": "h2h",
                            "outcomes": [{"name": "Baltimore Ravens", "price": -160}],
                        },
                    ],
                }
            ],
        }
    ]
    (parsed,) = parse("nfl", payload)
    assert parsed.matchup == "Los Angeles Chargers @ Baltimore Ravens"
    assert parsed.get("spreads", "Baltimore Ravens")[0].point == -3.0
    assert parsed.get("h2h", "Baltimore Ravens")[0].american == -160
    assert restrict([parsed], ("betmgm",)) == []
