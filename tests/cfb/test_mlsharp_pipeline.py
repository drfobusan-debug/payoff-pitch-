"""The sharp-money gate inside the pipeline: moneyline only, and attributed."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.data.vsin_splits import Split
from cfb_engine.features.adjustments import Adjustment
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import MatchupSignal
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.mlsharp import GATE
from cfb_engine.market.tiers import Tier
from cfb_engine.models.montecarlo import GameSimResult
from cfb_engine.pipeline import Pipeline, _GameCtx
from cfb_engine.recommendations import Recommendation
from cfb_engine.schemas import Game, TeamGameInfo

DAY = date(2026, 9, 5)
MATCHUP = "UGA @ ALA"


def _ctx(home_win_prob: float) -> _GameCtx:
    wins = int(round(home_win_prob * 1000))
    margins = np.array([7.0] * wins + [-7.0] * (1000 - wins))
    sim = GameSimResult(margins=margins, totals=np.full(1000, 52.0))
    game = Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    return _GameCtx(game, sim, Adjustment(), MatchupSignal())


def _ml_board(home_american: float, away_american: float) -> GameOdds:
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml("ALA", MarketQuote("b", home_american, opposite_american=away_american))
    odds.add_ml("UGA", MarketQuote("b", away_american, opposite_american=home_american))
    return odds


def _ats_board(home_point: float, american: float) -> GameOdds:
    odds = GameOdds(matchup=MATCHUP)
    for side in ("ALA", "UGA"):
        odds.add_spread(
            home_point, side, MarketQuote("b", american, opposite_american=american)
        )
    return odds


def _pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, splits: dict[tuple[str, str, str], Split]
) -> Pipeline:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_MARKET_ANCHOR", "0")
    monkeypatch.setenv("CFBE_SHRINK_TAILS", "0")  # isolate the gate from tail shrink
    pipe = Pipeline(Config(), cfbd=CFBDClient(None))
    pipe.splits = splits
    return pipe


def _favourite(pipe: Pipeline, home_win_prob: float = 0.68) -> Recommendation:
    """Alabama at -180 with the model above the price: a buy the price band
    cannot touch, so anything that refuses it is this gate."""
    recs = pipe._price_ml(_ctx(home_win_prob), _ml_board(-180.0, 150.0))
    return next(r for r in recs if r.selection.startswith("ALA"))


def test_a_buy_the_public_holds_more_heavily_than_the_money_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = {(MATCHUP, "game_ml", "ALA"): Split(60.0, 78.0, "circa")}
    rec = _favourite(_pipeline(tmp_path, monkeypatch, splits))
    assert (rec.tier, rec.pass_gate) == (Tier.PASS, GATE)
    assert rec.sharp_div == pytest.approx(-18.0)
    assert any("handle-minus-tickets -18 at circa" in r for r in rec.reasons)


def test_a_buy_the_money_confirms_ships_with_its_divergence_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits = {(MATCHUP, "game_ml", "ALA"): Split(78.0, 60.0, "circa")}
    rec = _favourite(_pipeline(tmp_path, monkeypatch, splits))
    assert (rec.tier, rec.pass_gate) != (Tier.PASS, GATE)
    assert rec.tier != Tier.PASS
    assert rec.sharp_div == pytest.approx(18.0)


def test_a_game_vsin_posts_no_split_for_is_priced_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Most of the CFB board has no public split; that cannot become a veto."""
    rec = _favourite(_pipeline(tmp_path, monkeypatch, {}))
    assert rec.tier != Tier.PASS
    assert rec.pass_gate is None
    assert rec.sharp_div is None


def test_the_spread_records_the_split_but_is_never_refused_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MLB inversion was measured on moneylines alone, so only they act on it."""
    splits = {(MATCHUP, "game_ats", "ALA"): Split(20.0, 80.0, "circa")}
    pipe = _pipeline(tmp_path, monkeypatch, splits)
    rec = next(
        r
        for r in pipe._price_ats(_ctx(0.68), _ats_board(-3.0, -110.0))
        if r.selection.startswith("ALA")
    )
    # 60 points of public money against this side, and no screen owns the row.
    assert rec.sharp_div == pytest.approx(-60.0)
    assert rec.pass_gate is None
    assert not any("handle-minus-tickets" in r for r in rec.reasons)


def test_a_split_taken_at_vsins_number_still_resolves_at_the_engines_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VSiN posts one number per game; the engine shops another. The money is on
    the team either way, so the split is keyed to the side, not the handicap."""
    splits = {(MATCHUP, "game_ats", "ALA"): Split(70.0, 50.0, "circa")}
    pipe = _pipeline(tmp_path, monkeypatch, splits)
    rec = next(
        r
        for r in pipe._price_ats(_ctx(0.68), _ats_board(-2.5, -110.0))
        if r.selection.startswith("ALA")
    )
    assert rec.sharp_div == pytest.approx(20.0)


def test_switching_the_gate_off_leaves_the_row_bought_and_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CFBE_ML_SHARP_GATE", "0")
    splits = {(MATCHUP, "game_ml", "ALA"): Split(60.0, 78.0, "circa")}
    rec = _favourite(_pipeline(tmp_path, monkeypatch, splits))
    assert rec.tier != Tier.PASS
    assert rec.pass_gate is None
    assert rec.sharp_div == pytest.approx(-18.0)
    assert any("gate off, measuring" in r for r in rec.reasons)


def test_a_row_another_screen_already_refused_is_not_reattributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attribution has to stay honest, or probation grades this gate on bets it
    never touched: the +260 dog belongs to the price band."""
    splits = {(MATCHUP, "game_ml", "UGA"): Split(20.0, 80.0, "circa")}
    pipe = _pipeline(tmp_path, monkeypatch, splits)
    dog = next(
        r
        for r in pipe._price_ml(_ctx(0.68), _ml_board(-300.0, 260.0))
        if r.selection.startswith("UGA")
    )
    assert dog.tier == Tier.PASS
    assert dog.pass_gate == "price_too_long"
