"""Ratings to points, the situational deltas, and the market's veto."""

from __future__ import annotations

from nfl_engine.features import adjustments as adj
from nfl_engine.features.ratings import RatingBook, TeamRating
from nfl_engine.models import expectation as exp_mod
from nfl_engine.models.drives import LG_DRIVES


def _book(**edits: TeamRating) -> RatingBook:
    teams = {"HOME": TeamRating(team="HOME"), "AWAY": TeamRating(team="AWAY")}
    teams.update(edits)
    return RatingBook(
        teams=teams,
        league={"drives": 11.0, "epa": 0.0},
        home_edge={"epa": 0.03},
        games_used=1000,
    )


def test_two_average_teams_get_the_home_edge_and_the_league_total():
    book = _book()
    view = exp_mod.forecast(book, "HOME", "AWAY")
    assert abs(view.margin() - exp_mod.HOME_EDGE) < 1e-6
    assert abs(view.total() - exp_mod.TOTAL_BASE) < 1e-6
    assert view.home_points > view.away_points


def test_a_better_offence_moves_the_margin_and_the_total():
    good = TeamRating(team="HOME", off_epa=0.10)
    flat = exp_mod.forecast(_book(), "HOME", "AWAY")
    strong = exp_mod.forecast(_book(HOME=good), "HOME", "AWAY")
    assert strong.margin() > flat.margin()
    assert strong.total() > flat.total()


def test_a_better_defence_moves_the_margin_but_lowers_the_total():
    stingy = TeamRating(team="HOME", def_epa=-0.10)
    flat = exp_mod.forecast(_book(), "HOME", "AWAY")
    wall = exp_mod.forecast(_book(HOME=stingy), "HOME", "AWAY")
    assert wall.margin() > flat.margin()
    assert wall.total() < flat.total()


def test_the_market_is_the_mean_and_the_rating_is_reported_beside_it():
    book = _book(HOME=TeamRating(team="HOME", off_epa=0.10))
    view = exp_mod.forecast(book, "HOME", "AWAY", market_margin=1.0, market_total=41.0)
    assert abs(view.margin() - 1.0) < 1e-6
    assert abs(view.total() - 41.0) < 1e-6
    assert view.rating_margin > 1.0
    assert view.rating_edge_margin() == view.rating_margin - 1.0
    assert view.rating_edge_total() == view.rating_total - 41.0


def test_a_blend_weight_below_one_moves_toward_the_rating():
    book = _book(HOME=TeamRating(team="HOME", off_epa=0.10))
    blended = exp_mod.forecast(
        book, "HOME", "AWAY", market_margin=1.0, market_total=41.0, market_weight=0.5
    )
    assert blended.margin() > 1.0
    assert blended.margin() < blended.rating_margin


def test_no_market_falls_back_to_the_rating():
    view = exp_mod.forecast(_book(), "HOME", "AWAY", market_margin=None)
    assert view.rating_edge_margin() is None
    assert abs(view.margin() - view.rating_margin) < 1e-6


def test_a_thin_book_defers_to_the_market_whatever_the_weight():
    """Under 400 games of history the rating does not get a vote."""
    thin = RatingBook(
        teams={"HOME": TeamRating(team="HOME", off_epa=0.30)},
        league={"drives": 11.0},
        games_used=10,
    )
    view = exp_mod.forecast(
        thin, "HOME", "AWAY", market_margin=-2.0, market_total=44.0, market_weight=0.0
    )
    assert abs(view.margin() - -2.0) < 1e-6


def test_drive_counts_come_from_the_league_when_a_team_is_unknown():
    empty = RatingBook()
    home, away = exp_mod.drive_counts(empty, "HOME", "AWAY")
    assert home == away == LG_DRIVES


def test_expected_game_carries_the_drive_counts_into_the_simulator():
    book = _book(HOME=TeamRating(team="HOME", off_drives=0.5))
    game = exp_mod.forecast(book, "HOME", "AWAY").expected_game()
    assert abs(game.home_drives - 11.5) < 1e-6
    assert abs(game.away_drives - 11.0) < 1e-6
    assert abs(game.margin() - exp_mod.HOME_EDGE) < 1e-6


def test_wind_takes_points_off_the_total_only_outdoors():
    calm = adj.Situation(roof="outdoors", wind_mph=adj.WIND_MEAN_MPH)
    gale = adj.Situation(roof="outdoors", wind_mph=25.0)
    dome = adj.Situation(roof="dome", wind_mph=25.0)
    assert abs(adj.wind_total_delta(calm)) < 1e-9
    assert adj.wind_total_delta(gale) < -2.0
    assert adj.wind_total_delta(dome) == 0.0
    assert adj.wind_total_delta(adj.Situation(roof="closed", wind_mph=25.0)) == 0.0


def test_a_calm_day_is_worth_points_the_other_way():
    """The market's total already contains an average breeze."""
    assert adj.wind_total_delta(adj.Situation(roof="outdoors", wind_mph=0.0)) > 0.0


def test_the_wind_term_is_capped():
    absurd = adj.Situation(roof="outdoors", wind_mph=200.0)
    assert adj.wind_total_delta(absurd) == adj.WIND_MAX_TOTAL_POINTS


def test_wind_does_not_pick_a_side():
    delta = adj.adjust(adj.Situation(roof="outdoors", wind_mph=25.0))
    assert delta.total_points < 0.0
    assert delta.margin_points == 0.0


def test_missing_wind_is_not_treated_as_calm():
    assert adj.wind_total_delta(adj.Situation(roof="outdoors", wind_mph=None)) == 0.0


def test_divisional_games_come_down_relative_to_non_divisional():
    div = adj.adjust(adj.Situation(roof="dome", div_game=True))
    non = adj.adjust(adj.Situation(roof="dome", div_game=False))
    assert div.total_points < non.total_points
    assert abs((non.total_points - div.total_points) - abs(adj.DIV_GAME_TOTAL_POINTS)) < 1e-9


def test_rest_travel_and_cold_are_reported_but_never_priced():
    """They measured at |t| < 1.9 against the closing line, so they get no points."""
    heavy = adj.Situation(
        roof="outdoors",
        wind_mph=adj.WIND_MEAN_MPH,
        temp_f=20.0,
        home_rest=4,
        away_rest=13,
        neutral_site=True,
        div_game=False,
    )
    delta = adj.adjust(heavy)
    priced = delta.total_points - adj.div_total_delta(heavy)
    assert abs(priced) < 1e-9
    assert delta.margin_points == 0.0
    notes = adj.unpriced_notes(heavy)
    assert any("short week" in note for note in notes)
    assert any("bye" in note for note in notes)
    assert any("neutral" in note for note in notes)
    assert any("cold" in note for note in notes)


def test_the_situational_block_reaches_the_forecast():
    windy = adj.Situation(roof="outdoors", wind_mph=25.0, div_game=True)
    view = exp_mod.forecast(
        _book(), "HOME", "AWAY", situation=windy, market_margin=0.0, market_total=44.0
    )
    assert view.total() < 44.0
    assert abs(view.margin()) < 1e-6
    assert "wind" in view.adjustment.describe()
