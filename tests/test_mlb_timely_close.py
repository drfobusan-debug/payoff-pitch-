"""The MLB per-game close and execution pricing.

CLV is scored only on a close taken inside 45 minutes of first pitch: the
fixed-time 12:50 / 18:40 snapshots were hours early for most games. And a buy
is struck at an execution book (DK/MGM) and measured against everyone else's
consensus, not at the best number on the whole board.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlb_engine.audit import clv, tape
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.config import Config, EVThresholds
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote, evaluate
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

D = date(2026, 8, 15)
START = datetime(2026, 8, 15, 23, 5, tzinfo=timezone.utc)


def _q(minutes_before: float | None, matchup: str = "LAD @ MIL", start: datetime = START):
    seen = "" if minutes_before is None else (start - timedelta(minutes=minutes_before)).isoformat()
    return clv.ClosingQuote(matchup, "game_total", "Over 8.5", -110.0, 0.52, seen, start.isoformat())


def test_a_close_counts_only_inside_45_minutes_of_first_pitch() -> None:
    assert clv.timely_close(_q(30))
    assert clv.timely_close(_q(45))
    assert not clv.timely_close(_q(46))
    assert not clv.timely_close(_q(-5))  # after first pitch
    assert not clv.timely_close(_q(None))  # unstamped: a pre-per-game close


def test_a_stale_close_is_quarantined_not_scored() -> None:
    def entry() -> LedgerEntry:
        return LedgerEntry(date=D.isoformat(), matchup="LAD @ MIL", category="game",
                           market="game_total", selection="Over 8.5", line=None, book="dk",
                           odds=-110.0, tier="pass", model_prob=0.5, ev=0.0, result="win",
                           pnl=0.0, fair_prob=0.5)
    stale, fresh, unstamped = entry(), entry(), entry()
    for e, q in ((stale, _q(180)), (fresh, _q(30)), (unstamped, _q(None))):
        clv.attach_clv([e], {q.key: q})
    assert stale.clv_quarantine == clv.STALE_CLOSE and stale.clv is None
    assert unstamped.clv_quarantine == clv.STALE_CLOSE
    assert fresh.clv_quarantine == clv.CLEAN and fresh.clv == pytest.approx(0.02)


def test_the_tape_close_ignores_captures_before_the_window() -> None:
    caps = [tape.QuoteCapture("e1", START, "LAD @ MIL", "game_total", "Over 8.5", "dk", -110.0, -110.0)]
    early = tape.to_rows(caps, slate_date=D, source=tape.CLOSE, now=START - timedelta(hours=3))
    assert clv.close_from_tape(early).reason == clv.STALE_CLOSE
    late = tape.to_rows(caps, slate_date=D, source=tape.RUN, now=START - timedelta(minutes=20))
    tc = clv.close_from_tape(early + late)
    assert tc.reason is None and tc.quote is not None


def test_a_game_is_due_once_inside_the_window() -> None:
    now = START - timedelta(minutes=40)
    games = [("LAD @ MIL", START.isoformat()), ("NYY @ BOS", (START + timedelta(hours=2)).isoformat())]
    assert clv.close_due(games, {}, now, 45) == [games[0]]
    closed = _q(40)
    assert clv.close_due(games, {closed.key: closed}, now, 45) == []
    # An early snapshot of the same game does not count as its close.
    early = _q(180)
    assert clv.close_due(games, {early.key: early}, now, 45) == [games[0]]


def test_game_two_of_a_doubleheader_is_due_after_game_one_closed() -> None:
    g2 = START + timedelta(hours=4)
    game_one = _q(30)
    due = clv.close_due([("LAD @ MIL", START.isoformat()), ("LAD @ MIL", g2.isoformat())],
                        {game_one.key: game_one}, g2 - timedelta(minutes=30), 45)
    assert due == [("LAD @ MIL", g2.isoformat())]


def test_stamps_survive_a_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "closing.json"
    clv.save_closing(path, [_q(30)])
    assert clv.load_closing(path)[_q(30).key] == _q(30)


def test_closing_quotes_stamp_capture_and_first_pitch() -> None:
    quotes = {("LAD @ MIL", "game_total", "Over 8.5"): [MarketQuote("fanduel", -110.0, opposite_american=-110.0)]}
    (q,) = clv.closing_quotes(quotes, captured_at="t0", first_pitch={"LAD @ MIL": "t1"})
    assert (q.captured_at, q.first_pitch_utc) == ("t0", "t1")


EXEC = frozenset({"draftkings", "betmgm"})


def test_the_bet_is_struck_at_an_execution_book_against_the_others() -> None:
    quotes = [
        MarketQuote("draftkings", -105.0, opposite_american=-115.0),
        MarketQuote("fanduel", 110.0, opposite_american=-130.0),  # best on the board
        MarketQuote("caesars", -120.0, opposite_american=100.0),
    ]
    res = evaluate(0.55, quotes, EXEC)
    assert res.best_quote.book == "draftkings" and res.executable
    others = evaluate(0.55, quotes[1:])
    assert res.fair_prob == pytest.approx(others.fair_prob)
    assert evaluate(0.55, quotes).best_quote.book == "fanduel"


def test_no_execution_book_is_not_executable() -> None:
    quotes = [MarketQuote("fanduel", 110.0, opposite_american=-130.0)]
    assert not evaluate(0.55, quotes, EXEC).executable
    # Only execution books on the side: their own consensus is all there is.
    only = [MarketQuote("draftkings", -110.0, opposite_american=-110.0)]
    res = evaluate(0.55, only, EXEC)
    assert res.executable and res.fair_prob == pytest.approx(0.5)


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _rec(book: str):
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config(ev=EVThresholds(min_prob=0.0, max_ev=1.0))
    p._calibrator = _Identity()
    p._shrink = None
    p._splits = {}
    p._ml_gate = MLSharpGate.from_env()
    p._pen_gate = MLPenGate.from_env()
    p._lineup_gate = LineupLockGate.from_env()
    p._lineup_lock = None
    p._drift_gate = DriftGate.from_env()
    p._open_board = {}
    sel = "Shota Imanaga Strikeouts o5.5"
    quotes = {("CWS @ CHC", "pitcher_k", sel): [MarketQuote(book, -110.0, opposite_american=-110.0)]}
    return p._mk(SimpleNamespace(game_date="2026-08-15", game_pk=1), "CWS @ CHC", "pitcher",
                 "pitcher_k", sel, 0.70, line=5.5, player_id=1, stat="K", side="over",
                 quotes=quotes)


def test_a_buy_no_execution_book_quotes_is_refused(monkeypatch, legacy_anchor) -> None:
    monkeypatch.setenv("MLBE_EXEC_BOOKS", "draftkings,betmgm")
    assert _rec("draftkings").tier is not Tier.PASS
    off_book = _rec("fanduel")
    assert off_book.tier is Tier.PASS and off_book.pass_gate == "no_execution_book"
    assert off_book.market_american == -110.0  # still priced, so probation can grade it
