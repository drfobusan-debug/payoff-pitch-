"""Graded-only markets: priced and graded, never bought."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.probation import (
    LIFT,
    CandidateScreen,
    candidate_probation,
    market_probation,
    screen_probation,
)
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.features.adjustments import Adjustment
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import MatchupSignal
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.gradedonly import GradedOnly, gate, is_gate
from cfb_engine.market.tiers import Tier
from cfb_engine.models.montecarlo import GameSimResult
from cfb_engine.pipeline import Pipeline, _GameCtx
from cfb_engine.schemas import Game, TeamGameInfo

DAY = date(2026, 9, 5)
GATE = gate("game_total")


def _ctx() -> _GameCtx:
    """Totals that land under 55.5 in 70% of draws."""
    totals = np.array([40.0] * 700 + [70.0] * 300)
    sim = GameSimResult(margins=np.full(1000, 3.0), totals=totals)
    game = Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name="Tulane", abbrev="TUL", is_home=True),
        away=TeamGameInfo(name="Memphis", abbrev="MEM", is_home=False),
    )
    return _GameCtx(game, sim, Adjustment(), MatchupSignal())


def _board() -> GameOdds:
    odds = GameOdds(matchup="MEM @ TUL")
    odds.add_total(55.5, True, MarketQuote("draftkings", -110.0, opposite_american=-110.0))
    odds.add_total(55.5, False, MarketQuote("draftkings", -110.0, opposite_american=-110.0))
    return odds


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_MARKET_ANCHOR", "0")
    monkeypatch.setenv("CFBE_SHRINK_TAILS", "0")
    monkeypatch.setenv("CFBE_MAX_EDGE", "1")
    return Pipeline(Config(), cfbd=CFBDClient(None))


def _under(pipe: Pipeline):  # type: ignore[no-untyped-def]
    return next(r for r in pipe._price_total(_ctx(), _board()) if r.side == "under")


def test_a_totals_buy_ships_as_a_graded_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CFBE_GRADED_ONLY_MARKETS", raising=False)
    rec = _under(_pipeline(tmp_path, monkeypatch))
    assert rec.tier == Tier.PASS
    assert rec.pass_gate == GATE
    assert any("graded only, not a bet" in r for r in rec.reasons)


def test_an_empty_list_buys_totals_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFBE_GRADED_ONLY_MARKETS", "")
    rec = _under(_pipeline(tmp_path, monkeypatch))
    assert rec.tier != Tier.PASS
    assert rec.pass_gate is None


def test_other_markets_are_untouched() -> None:
    assert GradedOnly().verdict("game_ml", "Moderate") == (True, "", None)
    assert GradedOnly().verdict("game_ats", "Strong") == (True, "", None)
    assert is_gate(GATE) and not is_gate("price_too_long") and not is_gate(None)


def _entry(result: str, day: int, *, tier: str, pass_gate: str | None) -> LedgerEntry:
    return LedgerEntry(
        date=f"2026-09-{day:02d}",
        matchup="Memphis @ Tulane",
        category="Total",
        market="game_total",
        selection="Under 45.5",
        line=45.5,
        book="b",
        odds=100.0,
        under_odds=-110,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl=1.0 if result == "win" else -1.0,
        pass_gate=pass_gate,
    )


def _rows(pattern: str) -> list[LedgerEntry]:
    return [
        _entry("win" if ch == "w" else "loss", 1 + i % 28, tier=Tier.PASS.value, pass_gate=GATE)
        for i, ch in enumerate(pattern)
    ]


def test_would_be_buys_grade_as_a_screen_and_feed_candidates() -> None:
    rows = _rows("w" * 30 + "l" * 5 + "w" * 30 + "l" * 5)
    (screen,) = screen_probation(rows, min_n=50)
    assert (screen.name, screen.status) == (GATE, LIFT)
    assert market_probation(rows, min_n=50) == []
    candidate = CandidateScreen("low_total", lambda e: e.market == "game_total", "test")
    (verdict,) = candidate_probation(rows, (candidate,), min_n=50)
    assert verdict.n == 70
