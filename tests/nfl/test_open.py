"""The opening stamp: the week's first archived board, and the drift since it.

The finding being acted on: across CFB's first 43 buys, the ones the market kept
moving toward after the bet went 17-6 and the ones it moved away from 7-11 --
but every card read ``drift 0.0`` because the run that bet also wrote the first
board. Here the open is whatever ``nfl-engine open`` archived Tuesday morning or
the night before, and the price run reads it back rather than its own board.
"""

from __future__ import annotations

import argparse

import pytest

from nfl_engine import cli
from nfl_engine.audit.ledger import (
    LedgerEntry,
    apply_open,
    line_move_toward,
    load_ledger,
    prob_per_point,
    save_ledger,
)
from nfl_engine.data import capture
from nfl_engine.market.board import GameOdds, MarketQuote

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"
OPENED = "2026-09-08T12:05:00Z"
BET = "2026-09-10T12:05:00Z"


def board(*, home_spread: float, total: float, home_ml: float = -150) -> dict[str, GameOdds]:
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(HOME, MarketQuote("dk", home_ml, -home_ml + 20))
    odds.add_ml(AWAY, MarketQuote("dk", -home_ml + 20, home_ml))
    odds.add_spread(home_spread, HOME, MarketQuote("dk", -110, -110))
    odds.add_spread(home_spread, AWAY, MarketQuote("dk", -110, -110))
    odds.add_total(total, True, MarketQuote("dk", -110, -110))
    odds.add_total(total, False, MarketQuote("dk", -110, -110))
    return {MATCHUP: odds}


def row(**overrides: object) -> LedgerEntry:
    entry = LedgerEntry(
        season=2026,
        week=1,
        date="2026-09-13",
        matchup=MATCHUP,
        market="spread",
        side=HOME,
        line=-7.5,
        book="dk",
        odds=-110.0,
        opposite_odds=-110.0,
        tier="Strong buy",
        model_prob=0.55,
        fair_prob=0.5,
        ev_model=0.05,
        ev_fair=0.0,
        paired_books=1,
        captured_at=BET,
        kickoff_utc="2026-09-13T17:00:00Z",
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def test_line_moves_are_read_on_the_side_taken() -> None:
    assert line_move_toward("spread", HOME, -6.5, -7.5) == 1.0  # favourite got dearer
    assert line_move_toward("spread", AWAY, 6.5, 7.5) == -1.0  # dog got a better number
    assert line_move_toward("total", "over", 47.5, 49.5) == 2.0
    assert line_move_toward("total", "under", 47.5, 49.5) == -2.0
    assert line_move_toward("moneyline", HOME, 0.0, 0.0) == 0.0


def test_drift_sums_the_handicap_and_the_price() -> None:
    entry = apply_open(row(), -110, -110, -6.5, captured_at=OPENED, margin_sd=13.2)
    assert entry.open_line == -6.5
    assert entry.open_odds == -110
    assert entry.open_prob == pytest.approx(0.5)
    assert entry.drift == pytest.approx(prob_per_point(13.2), abs=1e-6)
    assert entry.open_captured_at == OPENED


def test_an_unmoved_board_reads_zero_drift_not_missing() -> None:
    entry = apply_open(row(), -110, -110, -7.5, captured_at=BET)
    assert entry.drift == 0.0
    assert entry.open_captured_at == BET


def test_a_moneyline_drift_is_the_price_alone() -> None:
    entry = row(market="moneyline", line=None, odds=-160.0, opposite_odds=140.0)
    apply_open(entry, -140, 120, None)
    assert entry.open_line is None
    assert entry.drift is not None and entry.drift > 0  # -140 -> -160: dearer


def test_the_opening_board_is_the_earliest_snapshot_and_never_moves(tmp_path) -> None:
    first = capture.rows_from_board(
        board(home_spread=-6.5, total=47.5), season=2026, week=1, captured_at=OPENED
    )
    later = capture.rows_from_board(
        board(home_spread=-7.5, total=49.5), season=2026, week=1, captured_at=BET
    )
    assert capture.write_snapshot(first, season=2026, week=1, root=tmp_path) is not None
    assert capture.write_snapshot(later, season=2026, week=1, root=tmp_path) is not None
    opened, taken = capture.opening_board(2026, 1, root=tmp_path)
    assert taken == OPENED
    assert opened[MATCHUP].main_spread() == -6.5
    assert opened[MATCHUP].main_total() == 47.5


def test_the_opening_quote_uses_the_open_main_line_not_the_bet_line() -> None:
    opened = board(home_spread=-6.5, total=47.5)
    assert cli._opening_quote(opened, row()) == (-110, -110, -6.5)
    assert cli._opening_quote(opened, row(side=AWAY, line=7.5)) == (-110, -110, 6.5)
    assert cli._opening_quote(opened, row(market="total", side="under", line=49.5)) == (
        -110,
        -110,
        47.5,
    )
    assert cli._opening_quote(opened, row(market="moneyline", line=None)) == (-150, 170, None)
    assert cli._opening_quote({}, row()) is None


def test_price_stamps_the_open_from_the_archive_and_the_csv_keeps_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    opened = capture.rows_from_board(
        board(home_spread=-6.5, total=47.5), season=2026, week=1, captured_at=OPENED
    )
    capture.write_snapshot(opened, season=2026, week=1, root=tmp_path)
    monkeypatch.setattr(cli.capture, "capture_dir", lambda root=None: tmp_path / "captures")

    entries = [row(), row(market="total", side="over", line=49.5)]
    note = cli._stamp_open(entries, 2026, 1)
    assert note.startswith(f"opening board {OPENED}: 2 of 2 rows stamped")
    assert "own board" not in note
    assert entries[0].open_line == -6.5 and entries[0].drift is not None
    assert entries[0].drift > 0  # KC -6.5 -> -7.5: the market came to us first
    assert entries[1].open_line == 47.5
    assert entries[1].drift == pytest.approx(2.0 * prob_per_point(13.4), abs=1e-4)

    path = tmp_path / "ledger.csv"
    save_ledger(path, entries)
    back = load_ledger(path)
    assert back[0].open_line == -6.5
    assert back[0].open_captured_at == OPENED
    assert back[0].drift == pytest.approx(entries[0].drift or 0.0)


def test_pricing_off_the_first_board_of_the_week_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    own = capture.rows_from_board(
        board(home_spread=-7.5, total=49.5), season=2026, week=1, captured_at=BET
    )
    capture.write_snapshot(own, season=2026, week=1, root=tmp_path)
    monkeypatch.setattr(cli.capture, "capture_dir", lambda root=None: tmp_path / "captures")
    entries = [row()]
    note = cli._stamp_open(entries, 2026, 1)
    assert "nothing archived earlier" in note
    assert entries[0].drift == 0.0


def test_nothing_archived_leaves_the_open_empty(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(cli.capture, "capture_dir", lambda root=None: tmp_path / "captures")
    entries = [row()]
    assert "none archived" in cli._stamp_open(entries, 2026, 1)
    assert entries[0].open_odds is None and entries[0].drift is None


def test_open_command_archives_and_prices_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(cli.capture, "capture_dir", lambda root=None: tmp_path / "captures")
    calls: list[tuple[int, str]] = []

    def fake_fetch(days, *, kind=capture.GAME_KIND, archive=True):
        calls.append((days, kind))
        rows = capture.rows_from_board(
            board(home_spread=-6.5, total=47.5), season=2026, week=1, captured_at=OPENED
        )
        capture.write_snapshot(rows, season=2026, week=1, kind=kind)
        return cli.Fetched(2026, 1, OPENED, [], {})

    monkeypatch.setattr(cli, "_fetch", fake_fetch)
    monkeypatch.setattr(cli, "merge_ledger", lambda *a, **k: pytest.fail("open must not price"))
    assert cli.cmd_open(argparse.Namespace(days=8)) == 0
    assert calls == [(8, capture.GAME_KIND)]
    assert capture.opening_board(2026, 1)[1] == OPENED


def test_the_parser_knows_open(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_open", lambda args: seen.append(args) or 0)
    assert cli.main(["open"]) == 0
    assert seen and seen[0].days == 8
