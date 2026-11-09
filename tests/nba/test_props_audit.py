"""Prop reads in the audit: main line per player-game, Under band, watched players, coverage."""

from __future__ import annotations

import pytest

from nba_engine.audit import ledger, props
from nba_engine.audit.settle import pnl


def _row(
    day: str,
    player: str,
    market: str,
    side: str,
    line: float,
    fair: float,
    outcome: str,
    *,
    book: str = "draftkings",
    event: str = "ev1",
) -> ledger.LedgerRow:
    r = ledger.LedgerRow(
        day,
        f"{day}T23:00:00Z",
        event,
        "",
        "DET @ ORL",
        market,
        "incl_ot",
        side,
        player,
        line,
        book,
        -110.0 if book else None,
        -110.0 if book else None,
        1 if book else 0,
        "",
        5,
        5,
        fair,
        None,
        None,
        None,
        "Pass",
        "no_model" if book else "no_model;no_exec_book",
        f"{day}T20:00:00Z",
        "t",
        "",
        outcome=outcome,
    )
    r.pnl = pnl(outcome, -110.0) if outcome else None
    return r


def _pair(day: str, player: str, market: str, line: float, p_over: float, over_won: bool, **kw):
    o, u = ("win", "loss") if over_won else ("loss", "win")
    return [
        _row(day, player, market, "over", line, p_over, o, **kw),
        _row(day, player, market, "under", line, 1 - p_over, u, **kw),
    ]


def test_main_line_is_the_pair_nearest_a_coin_flip() -> None:
    rows = _pair("2026-11-02", "Cade Cunningham", "pl_reb", 4.5, 0.62, True)
    rows += _pair("2026-11-02", "Cade Cunningham", "pl_reb", 5.5, 0.49, False)
    rows += _pair("2026-11-02", "Cade Cunningham", "pl_reb", 6.5, 0.36, False)
    (m,) = props.main_lines(rows)
    assert m.line == 5.5 and m.under is not None and m.under.outcome == "win"


def test_rows_without_an_execution_book_price_are_not_read() -> None:
    rows = _pair("2026-11-02", "Caleb Wilson", "pl_pra", 26.5, 0.5, True, book="")
    assert props.main_lines(rows) == []
    cov = {m: (n, p) for m, n, p, _, _ in props.coverage(rows)}
    assert cov["pl_pra"] == (2, 0)


def test_under_band_is_one_bet_per_player_game_clustered_by_game() -> None:
    rows = _pair("2026-11-01", "A One", "pl_pts", 20.5, 0.5, False, event="g1")
    rows += _pair("2026-11-01", "B Two", "pl_ast", 5.5, 0.5, False, event="g1")
    rows += _pair("2026-11-02", "A One", "pl_pts", 21.5, 0.5, True, event="g2")
    under = props.under_bands(rows)[0]
    assert under.n == 3 and len(under.games) == 2
    assert under.units == pytest.approx(2 * 100 / 110 - 1)


def test_watched_players_read_their_own_market_only() -> None:
    rows = _pair("2026-10-30", "Tristan da Silva", "pl_3pm", 2.5, 0.5, True)
    rows += _pair("2026-11-01", "Tristan da Silva", "pl_3pm", 2.5, 0.5, False)
    rows += _pair("2026-11-01", "Tristan da Silva", "pl_pts", 11.5, 0.5, True)
    (w,) = props.watched(rows, [("Tristan Da Silva", "pl_3pm")])
    assert len(w.mains) == 2 and w.side("over").record == "1-1"
    text = "\n".join(props.lines(rows, [("Tristan da Silva", "pl_3pm")]))
    assert "Tristan da Silva 3PM: 2 games" in text and "11-01 2.5 Under" in text


def test_a_watched_player_with_no_line_says_so() -> None:
    text = "\n".join(props.lines([], [("Caleb Wilson", "pl_pra")]))
    assert "Caleb Wilson PRA: no DK/BetMGM main line graded yet" in text


def test_watchlist_env_replaces_the_default() -> None:
    assert props.watchlist("") == props.WATCH
    assert props.watchlist("Jalen Duren:pl_reb, Bad:pl_xyz") == (("Jalen Duren", "pl_reb"),)
    assert ("Cade Cunningham", "pl_reb") in props.WATCH
