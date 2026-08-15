"""The dry run: paper-only rows, append-once pricing, and a replayed week."""

from __future__ import annotations

from datetime import date as Date

import numpy as np
import pytest

from nfl_engine.audit.ledger import (
    LIVE,
    PAPER,
    apply_close,
    entry_from_bet,
    grade,
    load_ledger,
    merge_ledger,
    position_key,
    update_ledger,
)
from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.market.ev import price_game
from nfl_engine.models.distribution import ScoreDistribution
from nfl_engine.pipeline import price_slate
from nfl_engine.replay import CLOSE_BOOK, ReplayWeek
from nfl_engine.schemas import Game, TeamGameInfo

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"
TAKEN = "2026-09-10T18:00:00Z"


def distribution() -> ScoreDistribution:
    margins = np.array([-14, -7, -3, -3, 1, 3, 3, 6, 7, 14, 17, 21])
    return ScoreDistribution(home=np.full(len(margins), 24), away=24 - margins)


def board(home_ml: float = -150) -> GameOdds:
    odds = GameOdds(matchup=MATCHUP)
    for book in ("dk", "fd"):
        odds.add_ml(HOME, MarketQuote(book, home_ml, 130))
        odds.add_ml(AWAY, MarketQuote(book, 130, home_ml))
        odds.add_spread(-3.0, HOME, MarketQuote(book, -110, -110))
        odds.add_spread(-3.0, AWAY, MarketQuote(book, -110, -110))
        odds.add_total(47.5, True, MarketQuote(book, -110, -110))
        odds.add_total(47.5, False, MarketQuote(book, -110, -110))
    return odds


def game() -> Game:
    return Game(
        game_id="g1",
        season=2026,
        week=1,
        game_date=Date(2026, 9, 13),
        home=TeamGameInfo(name=HOME, abbrev=HOME, is_home=True),
        away=TeamGameInfo(name=AWAY, abbrev=AWAY, is_home=False),
    )


def entries(captured_at: str = TAKEN):
    bets = price_game(board(), distribution(), home=HOME, away=AWAY)
    return [
        entry_from_bet(bet, season=2026, week=1, date="2026-09-13", captured_at=captured_at)
        for bet in bets
    ]


def test_every_row_is_paper_and_carries_the_capture_time():
    rows = entries()
    assert rows
    assert {row.mode for row in rows} == {PAPER}
    assert {row.captured_at for row in rows} == {TAKEN}


def test_nothing_in_the_engine_writes_a_live_row():
    """The mode column exists to make a paper record unquotable as a real one."""
    assert LIVE == "live"
    assert all(row.mode == PAPER for row in entries())


def test_pricing_twice_adds_no_positions(tmp_path):
    path = tmp_path / "ledger.csv"
    first = merge_ledger(path, entries())
    assert len(first) == len(entries())
    again = merge_ledger(path, entries())
    assert again == []
    assert len(load_ledger(path)) == len(first)


def test_a_moved_price_does_not_become_a_second_position(tmp_path):
    path = tmp_path / "ledger.csv"
    merge_ledger(path, entries())
    moved = entries(captured_at="2026-09-12T18:00:00Z")
    for row in moved:
        if row.market == "moneyline" and row.side == HOME:
            row.odds = -175
    assert merge_ledger(path, moved) == []
    held = [r for r in load_ledger(path) if r.market == "moneyline" and r.side == HOME]
    # The price of record is the first one seen, not the latest.
    assert {row.odds for row in held} == {-150.0}


def test_a_new_rung_is_a_new_position(tmp_path):
    path = tmp_path / "ledger.csv"
    merge_ledger(path, entries())
    extra = board()
    extra.add_spread(-2.5, HOME, MarketQuote("dk", -120, 100))
    extra.add_spread(-2.5, AWAY, MarketQuote("dk", 100, -120))
    bets = price_game(extra, distribution(), home=HOME, away=AWAY)
    rows = [entry_from_bet(b, season=2026, week=1, date="2026-09-13") for b in bets]
    added = merge_ledger(path, rows)
    assert {(row.market, row.side, row.line) for row in added} == {
        ("spread", HOME, -2.5),
        ("spread", AWAY, 2.5),
    }


def test_grading_a_week_does_not_lose_the_capture_time(tmp_path):
    path = tmp_path / "ledger.csv"
    merge_ledger(path, entries())
    rows = load_ledger(path)
    for row in rows:
        grade(row, 27, 20, home=HOME)
    update_ledger(path, rows)
    graded = load_ledger(path)
    assert {row.captured_at for row in graded} == {TAKEN}
    assert all(row.result for row in graded)
    assert all(row.mode == PAPER for row in graded)


def test_position_key_separates_book_and_rung():
    rows = entries()
    assert len({position_key(row) for row in rows}) == len(rows)


def test_clv_is_zero_when_the_price_taken_is_the_close():
    """A replay stamps its own board as the close, so CLV must come out at zero.

    If it did not, the CLV arithmetic would be comparing two different kinds of
    number -- which is exactly the bug phase 5 shipped a test for.
    """
    row = next(r for r in entries() if r.market == "moneyline" and r.side == HOME)
    apply_close(row, row.odds, row.opposite_odds)
    assert row.clv == pytest.approx(0.0, abs=1e-9)


def test_replay_week_prices_and_grades_without_a_network(tmp_path):
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(HOME, MarketQuote(CLOSE_BOOK, -150, 130))
    odds.add_ml(AWAY, MarketQuote(CLOSE_BOOK, 130, -150))
    odds.add_spread(-3.0, HOME, MarketQuote(CLOSE_BOOK, -110, -110))
    odds.add_spread(-3.0, AWAY, MarketQuote(CLOSE_BOOK, -110, -110))
    week = ReplayWeek(
        season=2026,
        week=1,
        first_day=Date(2026, 9, 13),
        games=[game()],
        board={MATCHUP: odds},
        finals={MATCHUP: (27, 20)},
    )
    pricings = price_slate(week.games, week.board)
    rows = [
        entry_from_bet(bet, season=2026, week=1, date="2026-09-13", captured_at=TAKEN)
        for pricing in pricings
        for bet in pricing.bets
    ]
    assert rows
    # One book on the board means the consensus is the price taken: no execution
    # edge exists and nothing may be bought.
    assert not [row for row in rows if row.tier != "Pass"]
    for row in rows:
        final = week.finals[row.matchup]
        grade(row, final[0], final[1], home=HOME)
    settled = {(row.market, row.side): row.result for row in rows}
    assert settled[("moneyline", HOME)] == "win"
    assert settled[("moneyline", AWAY)] == "loss"
    assert settled[("spread", HOME)] == "win"  # won by 7, laid 3
    assert settled[("spread", AWAY)] == "loss"
    assert len(merge_ledger(tmp_path / "l.csv", rows)) == len(rows)


def test_replay_quote_rows_are_tagged_as_nflverse():
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(HOME, MarketQuote(CLOSE_BOOK, -150, 130))
    week = ReplayWeek(
        season=2026,
        week=1,
        first_day=Date(2026, 9, 13),
        games=[game()],
        board={MATCHUP: odds},
        finals={},
    )
    rows = week.quote_rows(TAKEN)
    assert [row.source for row in rows] == ["nflverse_close"]
    assert rows[0].game_date == "2026-09-13"
