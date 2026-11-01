"""The totals-environment veto refuses a buy and names the screen that did it."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.probation import _would_buy, screen_probation
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.features.adjustments import Adjustment
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import (
    LOW_SCORING_GATE,
    LOW_SCORING_REASON,
    SHOOTOUT_GATE,
    MatchupSignal,
    attributed_gate,
)
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.tiers import Tier
from cfb_engine.models.montecarlo import GameSimResult
from cfb_engine.pipeline import Pipeline, _GameCtx
from cfb_engine.schemas import Game, TeamGameInfo

DAY = date(2026, 10, 17)
LINE = 55.0


def _ctx(over_share: float, signal: MatchupSignal) -> _GameCtx:
    n_over = int(round(over_share * 1000))
    totals = np.array([70.0] * n_over + [40.0] * (1000 - n_over))
    sim = GameSimResult(margins=np.zeros(1000), totals=totals)
    game = Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name="Memphis", abbrev="MEM", is_home=True),
        away=TeamGameInfo(name="Tulane", abbrev="TUL", is_home=False),
    )
    return _GameCtx(game, sim, Adjustment(), signal)


def _board() -> GameOdds:
    odds = GameOdds(matchup="TUL @ MEM")
    quote = MarketQuote("b", -110, opposite_american=-110)
    odds.add_total(LINE, True, quote)
    odds.add_total(LINE, False, quote)
    return odds


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_MARKET_ANCHOR", "0")
    monkeypatch.setenv("CFBE_SHRINK_TAILS", "0")
    return Pipeline(Config(), cfbd=CFBDClient(None))


def _side(pipe: Pipeline, ctx: _GameCtx, side: str):
    recs = pipe._price_total(ctx, _board())
    return next(r for r in recs if r.side == side)


SHOOTOUT = MatchupSignal(ppa_edge=0.1, combined_ppa_env=0.2, combined_explosive_env=0.4)
LOW = MatchupSignal(ppa_edge=0.1, combined_ppa_env=-0.2, pace_env=-4.0)


def test_a_re_armed_shootout_under_is_refused_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_VETO_SHOOTOUT", "1")
    rec = _side(_pipeline(tmp_path, monkeypatch), _ctx(0.44, SHOOTOUT), "under")
    assert rec.tier == Tier.PASS
    assert rec.pass_gate == SHOOTOUT_GATE
    assert any(r.startswith("veto: shootout environment") for r in rec.reasons)


def test_with_the_shootout_veto_off_the_under_is_a_graded_only_buy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _side(_pipeline(tmp_path, monkeypatch), _ctx(0.44, SHOOTOUT), "under")
    assert rec.tier == Tier.PASS
    assert rec.pass_gate == "graded_only_game_total"


def test_a_low_scoring_over_keeps_its_own_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _side(_pipeline(tmp_path, monkeypatch), _ctx(0.56, LOW), "over")
    assert rec.tier == Tier.PASS
    assert rec.pass_gate == LOW_SCORING_GATE


def test_a_saved_veto_reason_recovers_its_gate_and_an_existing_gate_wins() -> None:
    reason = f"veto: {LOW_SCORING_REASON}"
    assert attributed_gate(None, ["edge 3.1%", reason]) == LOW_SCORING_GATE
    assert attributed_gate("graded_only_game_total", [reason]) == "graded_only_game_total"
    assert attributed_gate(None, ["veto: turnover-luck regression (+1.20/gm)"]) is None


def test_low_scoring_refusals_are_their_own_screen_and_not_would_be_buys() -> None:
    rows = [
        LedgerEntry(
            date=f"2026-10-{1 + i % 28:02d}",
            matchup=f"A{i} @ H{i}",
            category="Totals",
            market="game_total",
            selection="Over 45.5",
            line=45.5,
            book="b",
            odds=-110.0,
            tier=Tier.PASS.value,
            model_prob=0.56,
            ev=0.05,
            result="win" if i % 3 else "loss",
            pnl=0.9091 if i % 3 else -1.0,
            pass_gate=LOW_SCORING_GATE,
        )
        for i in range(12)
    ]
    (verdict,) = screen_probation(rows, min_n=10)
    assert verdict.name == LOW_SCORING_GATE and verdict.n == 12
    assert all(not _would_buy(e) for e in rows)
