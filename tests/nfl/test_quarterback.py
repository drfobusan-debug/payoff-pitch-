"""Who is starting: the book, the charge, and the two things it must not do."""

from __future__ import annotations

import pandas as pd

from nfl_engine.features import quarterback as qb_mod
from nfl_engine.features.ratings import RatingBook, TeamRating
from nfl_engine.models import expectation as exp_mod


def _games() -> pd.DataFrame:
    """One team's two seasons: STARTER all of 2023, then hurt in week 6 of 2024."""
    rows = []
    for week in range(1, 6):
        rows.append(
            {
                "season": 2023,
                "week": week,
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_qb_id": "STARTER",
                "away_qb_id": "OTHER",
            }
        )
    for week in range(1, 7):
        qb = "STARTER" if week < 6 else "BACKUP"
        rows.append(
            {
                "season": 2024,
                "week": week,
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_qb_id": qb,
                "away_qb_id": "OTHER",
            }
        )
    return pd.DataFrame(rows)


def _book() -> RatingBook:
    return RatingBook(
        teams={"HOME": TeamRating(team="HOME"), "AWAY": TeamRating(team="AWAY")},
        league={"drives": 11.0, "epa": 0.0},
        games_used=1000,
    )


def test_the_incumbent_and_the_backup_are_told_apart():
    book = qb_mod.build(_games())
    assert book.status(2024, 6, "HOME", "STARTER") == qb_mod.INCUMBENT
    assert book.status(2024, 6, "HOME", "BACKUP") == qb_mod.FILL_IN


def test_last_seasons_starter_counts_as_the_incumbent_in_week_one():
    book = qb_mod.build(_games())
    assert book.status(2024, 1, "HOME", "STARTER") == qb_mod.INCUMBENT


def test_an_unnamed_or_unknown_quarterback_charges_nothing():
    book = qb_mod.build(_games())
    assert book.status(2024, 6, "HOME", None) == qb_mod.UNKNOWN
    assert book.status(2024, 6, "NEWTEAM", "ROOKIE") == qb_mod.UNKNOWN
    delta, notes = qb_mod.margin_delta(
        book,
        season=2024,
        week=6,
        home="HOME",
        away="AWAY",
        home_qb=None,
        away_qb=None,
    )
    assert delta == 0.0
    assert notes == ()


def test_the_incumbent_is_read_from_prior_weeks_only():
    """Week 6 must not know that BACKUP started week 6, or the test is circular."""
    book = qb_mod.build(_games())
    assert book.incumbent[(2024, 6, "HOME")] == "STARTER"
    assert (2024, 1, "HOME") not in book.incumbent


def test_a_home_fill_in_costs_the_home_side_and_an_away_fill_in_pays_it():
    book = qb_mod.build(_games())
    home_out, notes = qb_mod.margin_delta(
        book,
        season=2024,
        week=6,
        home="HOME",
        away="AWAY",
        home_qb="BACKUP",
        away_qb="OTHER",
    )
    assert home_out == qb_mod.FILL_IN_MARGIN_POINTS
    assert home_out < 0.0
    assert notes and "HOME" in notes[0]
    away_out, _ = qb_mod.margin_delta(
        book,
        season=2024,
        week=6,
        home="AWAY",
        away="HOME",
        home_qb="OTHER",
        away_qb="BACKUP",
    )
    assert away_out == -qb_mod.FILL_IN_MARGIN_POINTS


def test_backups_on_both_sides_cancel():
    games = _games()
    games.loc[games.week == 6, "away_qb_id"] = "OTHER"
    book = qb_mod.build(games)
    delta, notes = qb_mod.margin_delta(
        book,
        season=2024,
        week=6,
        home="HOME",
        away="AWAY",
        home_qb="BACKUP",
        away_qb="OTHER_BACKUP",
    )
    assert delta == 0.0
    assert len(notes) == 2


def test_september_charges_nothing_because_the_bias_reverses_there():
    """Weeks 1-4 measured +1.14 (t +1.22) the *other* way; see the module docstring."""
    book = qb_mod.build(_games())
    delta, notes = qb_mod.margin_delta(
        book,
        season=2024,
        week=1,
        home="HOME",
        away="AWAY",
        home_qb="ROOKIE",
        away_qb="OTHER",
    )
    assert delta == 0.0
    assert notes == ()


def test_the_charge_moves_the_rating_and_not_the_market():
    """With the market at weight 1.0 the correction must not touch a price."""
    flat = exp_mod.forecast(_book(), "HOME", "AWAY", market_margin=3.0, market_total=44.0)
    hurt = exp_mod.forecast(
        _book(),
        "HOME",
        "AWAY",
        market_margin=3.0,
        market_total=44.0,
        qb_margin_points=-3.0,
    )
    assert hurt.margin() == flat.margin()
    assert hurt.rating_margin == flat.rating_margin - 3.0
    assert hurt.rating_edge_margin() == flat.rating_edge_margin() - 3.0


def test_the_charge_is_the_mean_when_no_market_has_posted():
    flat = exp_mod.forecast(_book(), "HOME", "AWAY")
    hurt = exp_mod.forecast(_book(), "HOME", "AWAY", qb_margin_points=-3.0)
    assert abs(hurt.margin() - (flat.margin() - 3.0)) < 1e-6
    assert abs(hurt.total() - flat.total()) < 1e-6


def test_an_empty_or_columnless_schedule_gives_an_empty_book():
    assert qb_mod.build(pd.DataFrame()).prior == {}
    assert qb_mod.build(pd.DataFrame({"season": [2024]})).incumbent == {}
