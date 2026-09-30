"""The moneyline probability: market-implied SD, and no one-sided shrink."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.features.adjustments import Adjustment
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import MatchupSignal
from cfb_engine.market.ev import MarketQuote
from cfb_engine.models.montecarlo import GameSimResult, market_win_prob
from cfb_engine.pipeline import Pipeline, _GameCtx
from cfb_engine.schemas import Game, TeamGameInfo

DAY = date(2026, 9, 5)


def _ctx(exp_margin: float) -> _GameCtx:
    rng = np.random.default_rng(1)
    margins = rng.normal(exp_margin, 16.0, 20000)
    sim = GameSimResult(margins=margins, totals=np.full(20000, 52.0))
    game = Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    return _GameCtx(game, sim, Adjustment(), MatchupSignal())


def _board(home_american: float, away_american: float) -> GameOdds:
    odds = GameOdds(matchup="UGA @ ALA")
    odds.add_ml("ALA", MarketQuote("b", home_american, opposite_american=away_american))
    odds.add_ml("UGA", MarketQuote("b", away_american, opposite_american=home_american))
    return odds


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_MARKET_ANCHOR", "0")
    return Pipeline(Config(), cfbd=CFBDClient(None))


def test_ml_sides_sum_to_one_and_favourite_is_not_shrunk(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    assert pipe.shrink is not None  # the tail shrink is on for the other markets
    recs = pipe._price_ml(_ctx(10.0), _board(-380.0, 300.0))
    by = {r.selection.split()[0]: r for r in recs}
    assert abs(by["ALA"].model_prob + by["UGA"].model_prob - 1.0) < 1e-9
    assert by["ALA"].model_prob == pytest.approx(by["ALA"].raw_prob)


def test_ml_favourite_priced_on_market_curve_not_flat_sixteen(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    ctx = _ctx(10.0)
    recs = pipe._price_ml(ctx, _board(-380.0, 300.0))
    fav = next(r for r in recs if r.selection.startswith("ALA"))
    assert fav.raw_prob == pytest.approx(market_win_prob(ctx.sim.exp_margin, pipe.cfg.model))
    # ~.77 on the market curve versus ~.73 from the 16-pt normal the sim drew.
    assert fav.raw_prob > ctx.sim.home_win_prob() + 0.03


def test_ml_flat_sim_curve_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBE_ML_MARKET_SD", "0")
    pipe = _pipeline(tmp_path, monkeypatch)
    ctx = _ctx(10.0)
    recs = pipe._price_ml(ctx, _board(-380.0, 300.0))
    fav = next(r for r in recs if r.selection.startswith("ALA"))
    assert fav.raw_prob == pytest.approx(ctx.sim.home_win_prob())
