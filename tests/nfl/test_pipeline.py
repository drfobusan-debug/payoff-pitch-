"""Board to screened bets: anchoring, and what happens when the anchor is missing."""

from __future__ import annotations

from datetime import date as Date

import pytest

from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.models.drives import DriveSim
from nfl_engine.pipeline import anchor, price_slate, slate_buys
from nfl_engine.schemas import Game, TeamGameInfo

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"


def game() -> Game:
    return Game(
        game_id="1",
        season=2025,
        week=3,
        game_date=Date(2025, 9, 21),
        home=TeamGameInfo(name="Kansas City Chiefs", abbrev=HOME, is_home=True),
        away=TeamGameInfo(name="Buffalo Bills", abbrev=AWAY, is_home=False),
    )


def full_board(*, outlier: float | None = None) -> GameOdds:
    odds = GameOdds(matchup=MATCHUP)
    for index in range(3):
        book = f"book{index}"
        odds.add_ml(HOME, MarketQuote(book=book, american=-150, opposite_american=130))
        odds.add_ml(AWAY, MarketQuote(book=book, american=130, opposite_american=-150))
        odds.add_spread(
            -3.0, HOME, MarketQuote(book=book, american=-110, opposite_american=-110)
        )
        odds.add_spread(
            -3.0, AWAY, MarketQuote(book=book, american=-110, opposite_american=-110)
        )
        odds.add_total(47.5, True, MarketQuote(book=book, american=-110, opposite_american=-110))
        odds.add_total(47.5, False, MarketQuote(book=book, american=-110, opposite_american=-110))
    if outlier is not None:
        odds.add_total(
            47.5, True, MarketQuote(book="outlier", american=outlier, opposite_american=-160)
        )
    return odds


def sim() -> DriveSim:
    return DriveSim(n_sims=4000, seed=3)


def test_the_anchor_is_the_consensus_margin_not_the_home_handicap() -> None:
    margin, total = anchor(full_board())
    assert margin == pytest.approx(3.0)  # home favoured by 3
    assert total == pytest.approx(47.5)


def test_a_board_at_consensus_produces_no_bets() -> None:
    pricings = price_slate([game()], {MATCHUP: full_board()}, sim=sim())
    assert len(pricings) == 1
    assert pricings[0].distribution is not None
    assert slate_buys(pricings) == []


def test_the_simulated_mean_tracks_the_market_it_was_anchored_to() -> None:
    pricing = price_slate([game()], {MATCHUP: full_board()}, sim=sim())[0]
    assert pricing.distribution is not None
    assert pricing.distribution.mean_margin() == pytest.approx(3.0, abs=1.0)
    assert pricing.distribution.mean_total() == pytest.approx(47.5, abs=1.5)


def test_a_book_off_the_market_survives_the_screens() -> None:
    pricings = price_slate([game()], {MATCHUP: full_board(outlier=140)}, sim=sim())
    bets = slate_buys(pricings)
    assert [bet.book for bet in bets] == ["outlier"]
    assert bets[0].market == "total"


def test_a_game_with_no_board_is_reported_not_priced() -> None:
    pricing = price_slate([game()], {}, sim=sim())[0]
    assert pricing.notes == ("no_board",)
    assert pricing.bets == []


def test_a_board_with_no_two_way_anchor_cannot_produce_a_bet() -> None:
    """Without a market mean the rating is the only forecast, and phase 3 says a
    bet cannot rest on it -- so every row is priced for the record and vetoed."""
    odds = GameOdds(matchup=MATCHUP)
    for index in range(3):
        odds.add_ml(
            HOME, MarketQuote(book=f"book{index}", american=-150, opposite_american=130)
        )
    pricing = price_slate([game()], {MATCHUP: odds}, sim=sim())[0]
    assert pricing.notes == ("no_market_anchor",)
    assert pricing.bets
    assert all("no_market_anchor" in bet.screens for bet in pricing.bets)
    assert slate_buys([pricing]) == []
