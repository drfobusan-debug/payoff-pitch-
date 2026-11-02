"""Buys price only at DraftKings/BetMGM; every book still feeds the consensus."""

from __future__ import annotations

from dataclasses import replace

from nhl_engine.audit.ledger import LedgerRow, backfill_ppp
from nhl_engine.config import GateParams
from nhl_engine.data.capture import QuoteRow
from nhl_engine.market.board import selections
from nhl_engine.market.pricing import stamp
from nhl_engine.models.markets import Prob
from nhl_engine.pipeline import _per_book_rules


def _q(side: str, american: float, opp: float, book: str, market: str = "game_ml") -> QuoteRow:
    return QuoteRow(
        "2026-10-01T15:00:00Z",
        "2026-10-01",
        "A @ H",
        "ev1",
        market,
        side,
        "",
        None,
        book,
        american,
        opp,
    )


def test_best_price_is_an_execution_books_quote_but_consensus_uses_every_book():
    rows = [
        _q("H", -120, 100, "draftkings"),
        _q("A", 100, -120, "draftkings"),
        _q("H", -105, -115, "bovada"),  # longer, but not a book a buy can be placed at
        _q("A", -115, -105, "bovada"),
    ]
    h = next(s for s in selections(rows) if s.side == "H")
    assert h.best_book == "draftkings" and h.best_american == -120 and h.executable
    assert h.books == 2 and h.two_sided == 2


def test_selection_no_execution_book_quotes_is_refused_no_exec_book():
    rows = [_q("H", -105, -115, "bovada"), _q("A", -115, -105, "bovada")]
    h = next(s for s in selections(rows) if s.side == "H")
    assert not h.executable and h.best_book == "bovada"
    p = stamp(
        h,
        Prob(0.60, 0.0),
        ot_rule="incl_ot_so",
        thr=GateParams(),
        quote_age_minutes=5,
        goalie_confirmed=True,
        settlement_gate=None,
    )
    assert "no_exec_book" in p.gates and not p.is_buy


def test_betmgm_counts_and_novig_does_not():
    rows = [
        _q("H", -110, -110, "betmgm"),
        _q("A", -110, -110, "betmgm"),
        _q("H", 100, -120, "novig"),
        _q("A", -120, 100, "novig"),
    ]
    h = next(s for s in selections(rows) if s.side == "H")
    assert h.best_book == "betmgm" and h.best_american == -110 and h.executable


def test_per_book_team_totals_mark_non_execution_books():
    rows = [
        _q("over", -110, -110, "betmgm", market="team_total"),
        _q("over", -105, -115, "fanduel", market="team_total"),
    ]
    split = {s.best_book: s.executable for s in _per_book_rules(selections(rows))}
    assert split == {"betmgm": True, "fanduel": False}


def test_per_book_team_totals_keep_each_books_own_quote_time():
    old = _q("over", -105, -115, "draftkings", market="team_total")
    new = _q("over", -110, -110, "betmgm", market="team_total")
    old = replace(old, captured_at="2026-10-01T12:00:00Z")
    at = {s.best_book: s.best_captured_at for s in _per_book_rules(selections([old, new]))}
    assert at == {"draftkings": "2026-10-01T12:00:00Z", "betmgm": "2026-10-01T15:00:00Z"}


def test_backfill_ppp_fills_only_ungraded_ppp_rows():
    def row(market: str, outcome: str | None) -> LedgerRow:
        r = LedgerRow(
            slate_date="2026-10-05",
            matchup="A @ H",
            home="H",
            away="A",
            market=market,
            side="over",
            entity="P",
            line=0.5,
            ot_rule="incl_ot",
            book="draftkings",
            american=150.0,
            books=1,
            consensus=0.4,
            model_prob=0.4,
            push_prob=0.0,
            edge=0.0,
            ev=0.0,
            tier="Pass",
        )
        r.outcome = outcome
        return r

    class Res:
        is_final = True

    rows = [row("sk_ppp", None), row("sk_ppp", "loss"), row("sk_sog", None)]
    import nhl_engine.audit.ledger as led

    orig = led.settle_ppp
    led.settle_ppp = lambda **_: "win"  # type: ignore[assignment]
    try:
        n = backfill_ppp(rows, {"A @ H": Res()}, lambda *_: 1)  # type: ignore[dict-item]
    finally:
        led.settle_ppp = orig
    assert n == 1
    assert [r.outcome for r in rows] == ["win", "loss", None]
    assert rows[0].pnl == 1.5
