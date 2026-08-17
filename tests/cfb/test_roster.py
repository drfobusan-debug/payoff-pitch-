"""Roster continuity: the transfer join, the shares it builds, and where it fires.

The term exists only on the ratings-only margin. Against a closing spread the
same signal goes 51.11% ATS, so a test pins that a game with a market number does
not receive it.
"""

from __future__ import annotations

import pytest

from cfb_engine.config import Config
from cfb_engine.data.roster import (
    FITTED_MAX_PTS,
    FITTED_PTS_PER_UNIT,
    RosterBook,
    build_incoming_shares,
    normalize_name,
    parse_production,
)


def _player(name: str, team: str, ppa: float) -> dict[str, object]:
    return {"name": name, "team": team, "totalPPA": {"all": ppa}}


def _move(
    first: str, last: str, origin: str, destination: str, date: str = "2025-01-10"
) -> dict[str, object]:
    return {
        "firstName": first,
        "lastName": last,
        "origin": origin,
        "destination": destination,
        "transferDate": f"{date}T00:00:00.000Z",
    }


def test_production_book_sums_a_team_and_keys_each_player():
    book = parse_production([_player("Will Rogers", "Texas", 60.0), _player("A B", "Texas", 40.0)])
    assert book.total_for("Texas") == 100.0
    assert book.players[("will rogers", "texas")] == 60.0


def test_names_fold_to_the_form_both_feeds_agree_on():
    assert normalize_name("Ja'Marr  Chase-Smith") == "jamarr chase smith"
    assert normalize_name("José Núñez Jr.") == "jose nunez jr"


def test_an_arrival_is_credited_the_production_he_actually_posted():
    prior = parse_production(
        [_player("Sam Leavitt", "Rice", 30.0), _player("Other Guy", "Texas", 70.0)]
    )
    shares = build_incoming_shares([_move("Sam", "Leavitt", "Rice", "Texas")], prior, 2025)
    # 30 PPA bought against Texas's own 70 PPA of prior output.
    assert shares["texas"] == 30.0 / 70.0


def test_a_transfer_who_never_played_is_credited_nothing_not_missing():
    """``/ppa/players/season`` lists only real usage, so absence means zero."""
    prior = parse_production([_player("Other Guy", "Texas", 70.0)])
    assert build_incoming_shares([_move("Walk", "On", "Rice", "Texas")], prior, 2025) == {}


def test_in_season_arrivals_are_dropped():
    prior = parse_production(
        [_player("Sam Leavitt", "Rice", 30.0), _player("Other Guy", "Texas", 70.0)]
    )
    late = [_move("Sam", "Leavitt", "Rice", "Texas", date="2025-09-01")]
    assert build_incoming_shares(late, prior, 2025) == {}


def test_one_arrival_cannot_outweigh_a_whole_prior_season():
    """An elite producer joining a weak roster is a denominator artefact."""
    prior = parse_production(
        [_player("Sam Leavitt", "Georgia", 200.0), _player("Other Guy", "Kent State", 20.0)]
    )
    shares = build_incoming_shares(
        [_move("Sam", "Leavitt", "Georgia", "Kent State")], prior, 2025
    )
    assert shares["kent state"] == 1.0


def test_the_gap_counts_kept_and_bought_production():
    book = RosterBook(retained={"texas": 0.40, "rice": 0.60}, bought={"texas": 0.35})
    assert book.share("Texas") == 0.75
    assert book.share("Rice") == 0.60
    assert book.gap("Texas", "Rice") == 0.75 - 0.60


def test_an_unknown_team_is_not_imputed_to_league_average():
    book = RosterBook(retained={"texas": 0.40}, bought={})
    assert book.share("Some FCS School") is None
    assert book.gap("Texas", "Some FCS School") is None
    assert book.margin_delta("Texas", "Some FCS School", FITTED_PTS_PER_UNIT, 8.0) == 0.0


def test_margin_delta_signs_and_caps_at_the_swept_bound():
    book = RosterBook(retained={"texas": 0.90, "rice": 0.10}, bought={"texas": 0.90})
    forward = book.margin_delta("Texas", "Rice", FITTED_PTS_PER_UNIT, FITTED_MAX_PTS)
    assert forward == FITTED_MAX_PTS  # 1.70 gap x 6.5 = 11.1 pts, clipped to 8
    assert book.margin_delta("Rice", "Texas", FITTED_PTS_PER_UNIT, FITTED_MAX_PTS) == -FITTED_MAX_PTS
    small = RosterBook(retained={"texas": 0.50, "rice": 0.40}, bought={})
    assert small.margin_delta("Texas", "Rice", FITTED_PTS_PER_UNIT, FITTED_MAX_PTS) == pytest.approx(
        0.65
    )


def test_a_zero_coefficient_disables_the_term():
    book = RosterBook(retained={"texas": 0.90, "rice": 0.10}, bought={})
    assert book.margin_delta("Texas", "Rice", 0.0, FITTED_MAX_PTS) == 0.0


def _pipeline(tmp_path, monkeypatch, calls: list[int]):
    from datetime import date

    from cfb_engine.data.cfbd import CFBDClient, RatingBook, TeamRating
    from cfb_engine.pipeline import Pipeline

    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))

    class _Fake(CFBDClient):
        def fetch_roster_book(self, season: int) -> RosterBook:
            calls.append(season)
            return RosterBook(retained={"iowa": 0.80, "nebraska": 0.20}, bought={})

    pipe = Pipeline(Config(), cfbd=_Fake(None))
    ratings = RatingBook(
        ratings={
            "iowa": TeamRating("Iowa", 28.0, 24.0),
            "nebraska": TeamRating("Nebraska", 26.0, 26.0),
        },
        league_avg=27.5,
    )
    return pipe, ratings, date(2026, 9, 5)


def _game():
    from datetime import date

    from cfb_engine.schemas import Game, TeamGameInfo

    return Game(
        game_id="1",
        game_date=date(2026, 9, 5),
        home=TeamGameInfo(name="Iowa", abbrev="IOWA", is_home=True),
        away=TeamGameInfo(name="Nebraska", abbrev="NEB", is_home=False),
    )


def _odds(*, spread: bool):
    from cfb_engine.market.board import GameOdds
    from cfb_engine.market.ev import MarketQuote

    odds = GameOdds(matchup="NEB @ IOWA")
    quote = MarketQuote(book="b", american=-110, opposite_american=-110)
    odds.add_ml("IOWA", MarketQuote(book="b", american=-160, opposite_american=+140))
    odds.add_ml("NEB", MarketQuote(book="b", american=+140, opposite_american=-160))
    if spread:
        odds.add_spread(-6.5, "IOWA", quote)
    return odds


def _priced(pipe, ratings, *, spread: bool):
    from cfb_engine.features.context import ContextBook
    from cfb_engine.models.montecarlo import MonteCarlo

    recs = pipe._price_game(
        _game(),
        _odds(spread=spread),
        ratings,
        ContextBook(),
        MonteCarlo(pipe.cfg.model),
        None,
        None,
        None,
        None,
        None,
        season=2026,
    )
    return [r for rec in recs for r in rec.reasons]


def test_a_game_with_a_market_number_never_receives_the_roster_term(tmp_path, monkeypatch):
    """It is 51.11% ATS against a closing spread, so the price wins the argument."""
    calls: list[int] = []
    pipe, ratings, _ = _pipeline(tmp_path, monkeypatch, calls)
    assert not any("buys more production" in r for r in _priced(pipe, ratings, spread=True))
    assert calls == []  # and the two extra CFBD calls are never even made


def test_a_line_less_game_is_priced_on_the_roster_gap(tmp_path, monkeypatch):
    calls: list[int] = []
    pipe, ratings, _ = _pipeline(tmp_path, monkeypatch, calls)
    reasons = _priced(pipe, ratings, spread=False)
    assert any("buys more production" in r for r in reasons)
    # 0.60 gap x 6.5 = 3.9 points to the side that kept and bought more.
    assert any("IOWA returns and buys more production (+3.9" in r for r in reasons)
    _priced(pipe, ratings, spread=False)
    assert calls == [2026]  # built once, then memoised


def test_shipped_defaults_are_the_fitted_ones():
    cfg = Config()
    assert cfg.roster_pts == FITTED_PTS_PER_UNIT
    assert cfg.roster_max_pts == FITTED_MAX_PTS
    # The market-facing version of the same idea stays off: 51.96% ATS.
    assert cfg.returning_pts == 0.0
