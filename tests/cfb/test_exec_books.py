"""Buys price only at the execution books; every book still feeds the fair price."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfb_engine.audit import snapshot
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import EXEC_BOOKS, MarketQuote, evaluate
from cfb_engine.market.tiers import Tier
from tests.cfb.test_priceband_pipeline import _ctx, _pipeline


def test_execution_books_are_draftkings_and_betmgm() -> None:
    assert EXEC_BOOKS == ("draftkings", "betmgm")


def test_a_longer_price_elsewhere_is_ignored_but_still_feeds_the_fair_price() -> None:
    quotes = [
        MarketQuote("draftkings", -110, -110),
        MarketQuote("betmgm", -105, -115),
        MarketQuote("lowvig", +105, -125),
    ]
    res = evaluate(0.55, quotes)
    assert res.executable
    assert (res.best_quote.book, res.best_quote.american) == ("betmgm", -105)
    exec_only = evaluate(0.55, quotes[:2])
    assert res.fair_prob != exec_only.fair_prob


def test_a_side_no_execution_book_quotes_is_priced_but_not_executable() -> None:
    res = evaluate(0.55, [MarketQuote("lowvig", +105, -125), MarketQuote("bovada", -110, -110)])
    assert not res.executable
    assert res.best_quote.book == "lowvig"


def test_every_book_counts_when_no_execution_books_are_given() -> None:
    quotes = [MarketQuote("draftkings", -110, -110), MarketQuote("lowvig", +105, -125)]
    res = evaluate(0.55, quotes, exec_books=None)
    assert res.executable and res.best_quote.book == "lowvig"


def test_the_board_snapshot_keeps_the_best_price_anywhere() -> None:
    quote = snapshot._quote(
        [MarketQuote("draftkings", -110, -110), MarketQuote("lowvig", +105, -125)], None
    )
    assert quote is not None and quote.american == 105


def _board(book: str) -> GameOdds:
    odds = GameOdds(matchup="UGA @ ALA")
    odds.add_ml("ALA", MarketQuote(book, -150.0, opposite_american=130.0))
    odds.add_ml("UGA", MarketQuote(book, 130.0, opposite_american=-150.0))
    return odds


def test_a_buy_without_an_execution_book_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _pipeline(tmp_path, monkeypatch)
    bought = next(
        r for r in pipe._price_ml(_ctx(0.64), _board("draftkings")) if r.selection.startswith("ALA")
    )
    assert bought.tier != Tier.PASS and bought.pass_gate is None
    refused = next(
        r for r in pipe._price_ml(_ctx(0.64), _board("lowvig")) if r.selection.startswith("ALA")
    )
    assert refused.tier == Tier.PASS
    assert refused.pass_gate == "no_exec_book"
    assert refused.book == "lowvig"
