"""The graded-only "strongest buy" flag: a near-market shadow price beside the engine's."""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

from cfb_engine.audit.ledger import entries_from_graded, load_ledger, update_ledger
from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.features.adjustments import Adjustment, compute_adjustment
from cfb_engine.features.context import GameContext
from cfb_engine.market import strongest
from cfb_engine.market.board import GameOdds
from cfb_engine.market.confidence import MatchupSignal
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.tiers import Tier
from cfb_engine.models.montecarlo import GameSimResult
from cfb_engine.output.card import _game_best_block, _slate_best_block
from cfb_engine.output.excel import COLUMNS, write_workbook
from cfb_engine.pipeline import Pipeline, _GameCtx
from cfb_engine.recommendations import Recommendation, load_json, save_json
from cfb_engine.schemas import Game, TeamGameInfo

DAY = date(2026, 11, 14)


def _sim(home_win_prob: float) -> GameSimResult:
    wins = int(round(home_win_prob * 1000))
    margins = np.array([7.0] * wins + [-7.0] * (1000 - wins))
    return GameSimResult(margins=margins, totals=np.full(1000, 52.0))


def _ctx(home_win_prob: float, shadow_prob: float | None) -> _GameCtx:
    game = Game(
        game_id="1",
        game_date=DAY,
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    shadow = None if shadow_prob is None else _sim(shadow_prob)
    return _GameCtx(game, _sim(home_win_prob), Adjustment(), MatchupSignal(), shadow)


def _board() -> GameOdds:
    odds = GameOdds(matchup="UGA @ ALA")
    odds.add_ml("ALA", MarketQuote("draftkings", -110.0, opposite_american=-110.0))
    odds.add_ml("UGA", MarketQuote("draftkings", -110.0, opposite_american=-110.0))
    return odds


@pytest.fixture
def pipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_MARKET_ANCHOR", "0")
    monkeypatch.setenv("CFBE_SHRINK_TAILS", "0")
    monkeypatch.setenv("CFBE_ML_MARKET_SD", "0")
    return Pipeline(Config(), cfbd=CFBDClient(None))


def _home(recs: list[Recommendation]) -> Recommendation:
    return next(r for r in recs if r.selection.startswith("ALA"))


def test_a_buy_that_still_clears_at_the_shadow_price_is_strongest(pipe: Pipeline) -> None:
    rec = _home(pipe._price_ml(_ctx(0.56, 0.555), _board()))
    assert rec.tier != Tier.PASS
    assert rec.strongest and rec.shadow_prob == pytest.approx(0.555)


def test_a_buy_the_shadow_price_reads_as_the_market_is_not(pipe: Pipeline) -> None:
    rec = _home(pipe._price_ml(_ctx(0.56, 0.505), _board()))
    assert rec.tier != Tier.PASS
    assert not rec.strongest


def test_a_pass_is_never_strongest_even_when_the_shadow_would_buy(pipe: Pipeline) -> None:
    rec = _home(pipe._price_ml(_ctx(0.505, 0.56), _board()))
    assert rec.tier == Tier.PASS and not rec.strongest


def test_a_pass_the_shadow_would_buy_is_a_watch_row(pipe: Pipeline) -> None:
    rec = _home(pipe._price_ml(_ctx(0.505, 0.56), _board()))
    assert rec.shadow_buy and rec.shadow_watch
    assert rec.as_row()[strongest.LABEL] == strongest.WATCH_CELL
    plain = _home(pipe._price_ml(_ctx(0.505, 0.505), _board()))
    assert not plain.shadow_buy and not plain.shadow_watch
    assert plain.as_row()[strongest.LABEL] == ""


def test_a_shadow_buy_still_needs_an_executable_book(pipe: Pipeline) -> None:
    odds = GameOdds(matchup="UGA @ ALA")
    odds.add_ml("ALA", MarketQuote("lowvig", -110.0, opposite_american=-110.0))
    odds.add_ml("UGA", MarketQuote("lowvig", -110.0, opposite_american=-110.0))
    rec = _home(pipe._price_ml(_ctx(0.505, 0.56), odds))
    assert rec.tier == Tier.PASS and not rec.shadow_buy


@pytest.mark.parametrize("shadow", [None, 0.50, 0.555])
def test_the_shadow_never_moves_the_engines_tier_price_or_ev(
    pipe: Pipeline, shadow: float | None
) -> None:
    base = _home(pipe._price_ml(_ctx(0.56, None), _board()))
    rec = _home(pipe._price_ml(_ctx(0.56, shadow), _board()))
    bare = dataclasses.replace(rec, shadow_prob=None, strongest=False, shadow_buy=False)
    assert bare == base


def test_without_situational_drops_rest_bye_and_travel_but_keeps_the_neutral_site() -> None:
    params = Config().features
    ctx = GameContext(neutral_site=True, rest_home=14, rest_away=6, travel_away_miles=2400.0)
    full = compute_adjustment(ctx, params, 2.5, "ALA", "UGA")
    bare = compute_adjustment(ctx, strongest.without_situational(params), 2.5, "ALA", "UGA")
    neutral_only = compute_adjustment(GameContext(neutral_site=True), params, 2.5, "ALA", "UGA")
    assert full.margin_delta != bare.margin_delta
    assert bare.margin_delta == pytest.approx(neutral_only.margin_delta)


def _rec(selection: str, tier: Tier, *, is_strongest: bool) -> Recommendation:
    return Recommendation(
        game_date=DAY,
        game_id=selection,
        matchup=f"AWAY @ {selection}",
        market="game_ml",
        selection=f"{selection} ML",
        line=None,
        model_prob=0.56,
        market_american=-110,
        ev=0.07,
        edge=0.036,
        fair_prob=0.524,
        tier=tier,
        home_abbrev=selection,
        away_abbrev="AWAY",
        team_side="home",
        side="win",
        shadow_prob=0.55 if is_strongest else 0.51,
        strongest=is_strongest,
        shadow_buy=is_strongest,
    )


def _watch(selection: str) -> Recommendation:
    rec = _rec(selection, Tier.PASS, is_strongest=False)
    return dataclasses.replace(rec, model_prob=0.53, shadow_prob=0.56, shadow_buy=True)


def test_the_card_lists_watch_rows_in_their_own_box() -> None:
    from cfb_engine.output.card import _shadow_watch_block

    recs = [_rec("IND", Tier.STRONG, is_strongest=True), _watch("TEM")]
    block = _shadow_watch_block(recs)
    assert strongest.WATCH_LABEL in block and "TEM ML" in block and "IND ML" not in block
    quiet = _shadow_watch_block([_rec("IND", Tier.STRONG, is_strongest=True)])
    assert "No pass clears" in quiet
    old = [dataclasses.replace(_watch("TEM"), shadow_prob=None)]
    assert _shadow_watch_block(old) == ""


def test_the_card_prints_strongest_buys_in_bold_caps_with_a_note() -> None:
    recs = [
        _rec("IND", Tier.STRONG, is_strongest=True),
        _rec("OSU", Tier.STRONG, is_strongest=False),
    ]
    for block in (_slate_best_block(recs), _game_best_block(recs)):
        assert block.count("<li class='strongest'>") == 1
        assert "<li><b>OSU ML" in block or "OSU ML" in block.split("<li class='strongest'>")[0]
        assert strongest.CARD_NOTE in block
    plain = _slate_best_block([_rec("OSU", Tier.STRONG, is_strongest=False)])
    assert "strongest" not in plain


def test_the_workbook_and_ledger_carry_the_column(tmp_path: Path) -> None:
    recs = [
        _rec("IND", Tier.STRONG, is_strongest=True),
        _rec("OSU", Tier.MODERATE, is_strongest=False),
    ]
    assert strongest.LABEL in COLUMNS
    wb = load_workbook(write_workbook(recs, tmp_path / "card.xlsx", DAY))
    ws = wb["All"]
    header = [c.value for c in ws[1]]
    col = header.index(strongest.LABEL)
    cells = {r[3]: r[col] for r in ws.iter_rows(min_row=2, values_only=True) if r[3]}
    assert cells["IND ML"] == strongest.CELL and cells["OSU ML"] in (None, "")

    path = tmp_path / "ledger.csv"
    update_ledger(path, entries_from_graded([(r, "win") for r in recs], DAY), DAY)
    assert [e.strongest for e in load_ledger(path)] == [True, False]
    assert [e.shadow_buy for e in load_ledger(path)] == [True, False]

    watch = tmp_path / "watch.xlsx"
    ws = load_workbook(write_workbook([_watch("TEM")], watch, DAY))["All"]
    header = [c.value for c in ws[1]]
    row = next(ws.iter_rows(min_row=2, values_only=True))
    assert row[header.index(strongest.LABEL)] == strongest.WATCH_CELL


def test_old_cards_and_ledger_rows_load_without_the_flag(tmp_path: Path) -> None:
    rec = _rec("IND", Tier.STRONG, is_strongest=False)
    old = dataclasses.replace(rec, shadow_prob=None)
    path = tmp_path / "p.json"
    save_json([old], path)
    raw = json.loads(path.read_text())
    for row in raw:
        del row["shadow_prob"], row["strongest"], row["shadow_buy"]
    path.write_text(json.dumps(raw))
    (loaded,) = load_json(path)
    assert loaded.strongest is False and loaded.shadow_prob is None

    ledger = tmp_path / "ledger.csv"
    update_ledger(ledger, entries_from_graded([(old, "loss")], DAY), DAY)
    assert load_ledger(ledger)[0].strongest is None
    assert load_ledger(ledger)[0].shadow_buy is None
