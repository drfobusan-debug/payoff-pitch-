"""Early-season total pull toward the market for games with a top-ranked defense."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient, RatingBook, TeamRating
from cfb_engine.features.context import ContextBook
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote
from cfb_engine.models.montecarlo import MonteCarlo
from cfb_engine.pipeline import Pipeline
from cfb_engine.schemas import Game, TeamGameInfo

DAY = date(2027, 9, 18)
LA = 26.0
BOOK = RatingBook(
    ratings={
        "iowa": TeamRating("Iowa", 24.0, 14.0),
        "nebraska": TeamRating("Nebraska", 30.0, 28.0),
        "rice": TeamRating("Rice", 22.0, 32.0),
    },
    league_avg=LA,
)


def _pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, week: int = 3) -> Pipeline:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    pipe = Pipeline(Config(), cfbd=CFBDClient(None))
    monkeypatch.setattr(pipe, "_slate_week", lambda season, slate_date: week)
    return pipe


def _game(home: str, home_ab: str, away: str, away_ab: str) -> Game:
    return Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name=home, abbrev=home_ab, is_home=True),
        away=TeamGameInfo(name=away, abbrev=away_ab, is_home=False),
    )


def _odds(total: float = 50.0, home_spread: float = -3.0) -> GameOdds:
    odds = GameOdds(matchup="x")
    q = MarketQuote(book="b", american=-110, opposite_american=-110)
    odds.add_spread(home_spread, "H", q)
    odds.add_total(total, True, q)
    odds.add_total(total, False, q)
    return odds


def _rating_total(game: Game) -> float:
    h, a = BOOK.get(game.home.name), BOOK.get(game.away.name)
    assert h is not None and a is not None
    return (h.offense * a.defense + a.offense * h.defense) / LA


def test_only_the_top_ranked_defenses_on_early_weeks_from_2027(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBE_TOPDEF_RANK", "1")
    pipe = _pipeline(tmp_path, monkeypatch)
    assert pipe._early_top_defenses(BOOK, 2027, DAY) == frozenset({"iowa"})
    assert pipe._early_top_defenses(BOOK, 2026, DAY) == frozenset()
    assert pipe._early_top_defenses(None, 2027, DAY) == frozenset()


def test_the_pull_ends_after_week_five(tmp_path, monkeypatch):
    assert _pipeline(tmp_path, monkeypatch, week=5)._early_top_defenses(BOOK, 2027, DAY)
    assert (
        _pipeline(tmp_path, monkeypatch, week=6)._early_top_defenses(BOOK, 2027, DAY) == frozenset()
    )


def test_a_top_defense_game_blends_the_total_not_the_margin(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path, monkeypatch)
    game = _game("Iowa", "IOWA", "Nebraska", "NEB")
    base = pipe._means(game, _odds(), BOOK, 2.4)
    pulled = pipe._means(game, _odds(), BOOK, 2.4, total_blend=0.55)
    assert base is not None and pulled is not None
    rt = _rating_total(game)
    assert base.exp_total == pytest.approx(0.65 * rt + 0.35 * 50.0)
    assert pulled.exp_total == pytest.approx(0.45 * rt + 0.55 * 50.0)
    assert pulled.exp_margin == pytest.approx(base.exp_margin)


def _price(pipe: Pipeline, game: Game) -> tuple[float, list[str]]:
    recs = pipe._price_game(
        game, _odds(), BOOK, ContextBook(), MonteCarlo(pipe.cfg.model), season=2027
    )
    totals = [r for r in recs if r.market == "game_total"]
    assert totals
    return totals[0].exp_total, totals[0].reasons


def test_price_game_applies_and_names_the_pull(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBE_TOPDEF_RANK", "1")
    pipe = _pipeline(tmp_path, monkeypatch)
    pipe._top_defenses = pipe._early_top_defenses(BOOK, 2027, DAY)
    top = _game("Iowa", "IOWA", "Nebraska", "NEB")
    exp_total, reasons = _price(pipe, top)
    assert exp_total == pytest.approx(0.45 * _rating_total(top) + 0.55 * 50.0, abs=0.3)
    assert "Top-1 defense (IOWA): total 55% to market" in reasons

    plain = _game("Rice", "RICE", "Nebraska", "NEB")
    exp_total, reasons = _price(pipe, plain)
    assert exp_total == pytest.approx(0.65 * _rating_total(plain) + 0.35 * 50.0, abs=0.3)
    assert not any(r.startswith("Top-") for r in reasons)
