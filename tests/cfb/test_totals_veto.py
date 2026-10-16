"""The totals-environment veto refuses a buy and names the screen that did it."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.features.adjustments import Adjustment
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import LOW_SCORING_GATE, SHOOTOUT_GATE, MatchupSignal
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


def test_a_shootout_under_is_refused_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _side(_pipeline(tmp_path, monkeypatch), _ctx(0.44, SHOOTOUT), "under")
    assert rec.tier == Tier.PASS
    assert rec.pass_gate == SHOOTOUT_GATE
    assert any(r.startswith("veto: shootout environment") for r in rec.reasons)


def test_turning_the_shootout_flag_off_lets_the_under_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_VETO_SHOOTOUT", "0")
    rec = _side(_pipeline(tmp_path, monkeypatch), _ctx(0.44, SHOOTOUT), "under")
    assert rec.tier != Tier.PASS
    assert rec.pass_gate is None


def test_a_low_scoring_over_keeps_its_own_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_VETO_SHOOTOUT", "0")
    rec = _side(_pipeline(tmp_path, monkeypatch), _ctx(0.56, LOW), "over")
    assert rec.tier == Tier.PASS
    assert rec.pass_gate == LOW_SCORING_GATE
