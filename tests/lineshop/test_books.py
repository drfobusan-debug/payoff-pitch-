from lineshop import books as bk
from lineshop.feed import Game, Quote

BOOKS = ("DraftKings", "FanDuel", "BetMGM", "BetRivers", "Caesars", "Bovada")


def board(best_home: str, best_away: str, games: int = 3) -> list[Game]:
    """A slate where two named books hold the best number on every side."""
    out = []
    for i in range(games):
        quotes: dict[tuple[str, str], list[Quote]] = {
            ("spreads", "Home"): [],
            ("spreads", "Away"): [],
        }
        for book in BOOKS:
            home_pt = -2.5 if book == best_home else -3.0
            away_pt = 3.5 if book == best_away else 3.0
            quotes[("spreads", "Home")].append(Quote(book, -110, home_pt))
            quotes[("spreads", "Away")].append(Quote(book, -110, away_pt))
        out.append(Game("nfl", f"g{i}", "Home", "Away", "2026-09-10T00:20:00Z", quotes))
    return out


def test_coverage_is_a_property_of_the_set_not_the_books_in_it():
    report = bk.rank("nfl", board("DraftKings", "FanDuel"), sets_of=2)
    assert report.coverage[("DraftKings", "FanDuel")] == 1.0
    # Two books that are never best cover nothing, however respectable.
    assert report.coverage[("BetMGM", "BetRivers")] == 0.0


def test_a_redundant_pair_is_worth_no_more_than_one_of_them():
    # DraftKings and FanDuel post the identical best number on both sides, so
    # the second account adds nothing the first did not already have.
    games = board("DraftKings", "DraftKings")
    for game in games:
        for side, point in (("Home", -2.5), ("Away", 3.5)):
            quotes = game.quotes[("spreads", side)]
            game.quotes[("spreads", side)] = [
                Quote("FanDuel", -110, point) if q.book == "FanDuel" else q for q in quotes
            ]
    report = bk.rank("nfl", games, sets_of=2, fixed=("DraftKings",))
    assert report.coverage[("DraftKings", "FanDuel")] == 1.0
    assert report.coverage[("DraftKings",)] == report.coverage[("DraftKings", "FanDuel")]


def test_a_lonely_quote_is_not_a_best_price():
    thin = Game(
        "nfl",
        "thin",
        "Home",
        "Away",
        "2026-09-10T00:20:00Z",
        {("spreads", "Home"): [Quote("Bovada", -110, -1.0)]},
    )
    report = bk.rank("nfl", [thin])
    assert report.sides == 0
    assert report.scores == ()


def test_hold_pairs_the_two_sides_of_a_spread():
    games = board("DraftKings", "FanDuel", games=1)
    scores = {s.book: s for s in bk.rank("nfl", games).scores}
    # -110 both ways is a 4.76% overround, and it only shows up if the -3 and
    # +3 quotes are recognised as the same market.
    assert scores["Caesars"].hold > 0.04


def test_the_fixed_set_is_kept_and_filled_out():
    report = bk.rank("nfl", board("DraftKings", "FanDuel"), sets_of=3, fixed=("BetMGM",))
    with_mgm = [c for c in report.coverage if "BetMGM" in c and len(c) == 3]
    assert with_mgm
    assert max(report.coverage[c] for c in with_mgm) == 1.0
