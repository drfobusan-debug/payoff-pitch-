"""Tests for the four selection guards the 27-slate ledger review produced.

Buys over that ledger went 39.8% for -12.6% ROI while the model's own ranking
held up, so every guard here tightens *selection* only -- none of them touches a
simulated or calibrated probability, and each is independently switchable:

  * ``MLBE_MAX_BUY_ODDS`` -- price ceiling (plus-money buys went 28.5%)
  * ``MLBE_NO_BUY_<MARKET>`` -- markets the ledger disqualified outright
  * ``MLBE_MARKET_ANCHOR[_<MARKET>]`` -- toll for disagreeing with the price
  * ``MLBE_CLV_GATE`` -- pre-bet closing line value, off the opening board
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from mlb_engine.audit.clv import (
    ClosingQuote,
    board_path,
    closing_quotes,
    load_closing,
    merge_board,
    quote_key,
    save_closing,
)
from mlb_engine.config import Config, EVThresholds
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import EVResult, MarketQuote, evaluate
from mlb_engine.market.tiers import Tier, classify
from mlb_engine.pipeline import Pipeline

MATCHUP = "MIA @ ATL"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _res(edge: float, american: float = -110.0) -> EVResult:
    q = MarketQuote(book="bk", american=american)
    fair = 0.5
    prob = fair + edge
    return EVResult(
        model_prob=prob,
        best_quote=q,
        decimal=1.91,
        ev=prob * 0.91 - (1.0 - prob),
        fair_prob=fair,
        edge=edge,
        sharp_divergence=None,
    )


def _pipeline(cfg: Config | None = None) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg or Config()
    p._calibrator = _Identity()
    p._shrink = None
    p._splits = {}
    p._ml_gate = MLSharpGate.from_env()
    p._pen_gate = MLPenGate.from_env()
    p._lineup_gate = LineupLockGate.from_env()
    p._lineup_lock = None
    p._drift_gate = DriftGate.from_env()
    p._open_board = {}
    return p


def _rec(
    p: Pipeline,
    market: str,
    model_prob: float,
    american: float = -110.0,
    selection: str = "MIA ML",
):
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        (MATCHUP, market, selection): [
            MarketQuote(book="dk", american=american, opposite_american=-110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "game", market, selection, model_prob,
        team_side="away", side="win", quotes=quotes,
    )


# ---- price ceiling ---------------------------------------------------------
def test_plus_money_is_never_bought() -> None:
    thr = EVThresholds()
    tier, reasons = classify(_res(0.06, american=150.0), thr)
    assert tier is Tier.PASS
    assert any("longer than" in r for r in reasons)


def test_the_same_edge_at_a_short_price_still_buys() -> None:
    tier, _ = classify(_res(0.06, american=-130.0), EVThresholds())
    assert tier is Tier.STRONG


def test_ceiling_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MAX_BUY_ODDS", "100000")
    thr = EVThresholds().for_market("game_ml")
    assert classify(_res(0.06, american=150.0), thr)[0] is Tier.STRONG


def test_ceiling_takes_a_per_market_override(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MAX_BUY_ODDS_GAME_TOTAL", "100000")
    base = EVThresholds()
    assert base.for_market("game_total").max_buy_odds == 100000
    assert base.for_market("game_ml").max_buy_odds == base.max_buy_odds


# ---- disqualified markets --------------------------------------------------
def test_losing_batter_markets_are_disqualified() -> None:
    base = EVThresholds()
    assert base.for_market("batter_h").no_buy
    assert base.for_market("batter_hr").no_buy
    # Doubles are the one batter market the ledger has in profit, and the
    # game markets are graded on their own record.
    assert not base.for_market("batter_2b").no_buy
    assert not base.for_market("game_ml").no_buy


def test_a_disqualified_market_passes_at_any_edge() -> None:
    thr = EVThresholds().for_market("batter_h")
    tier, reasons = classify(_res(0.07), thr)
    assert tier is Tier.PASS
    assert any("disqualified" in r for r in reasons)


def test_disqualification_is_reversible(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_NO_BUY_BATTER_H", "0")
    assert not EVThresholds().for_market("batter_h").no_buy


def test_a_disqualified_market_is_still_priced_and_graded() -> None:
    """Shadow bets: the pass keeps the price, EV and edge for the ledger."""
    p = _pipeline()
    rec = _rec(p, "batter_h", 0.62, selection="Some Batter o0.5 H")
    assert rec.tier is Tier.PASS
    assert rec.market_american == -110.0
    assert rec.ev is not None and rec.edge is not None
    assert rec.model_prob == 0.62


# ---- market anchoring ------------------------------------------------------
def test_anchor_defaults_to_a_toll_everywhere_but_totals() -> None:
    cfg = Config()
    assert cfg.anchor_for("game_ml") == 0.5
    assert cfg.anchor_for("batter_h") == 0.5
    # Totals are the only market where the model out-forecasts the price, and
    # the only profitable buy bucket, so they are not taxed.
    assert cfg.anchor_for("game_total") == 0.0
    assert cfg.anchor_for("f5_total") == 0.0


def test_anchor_takes_a_per_market_override(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MARKET_ANCHOR_GAME_TOTAL", "0.3")
    assert Config().anchor_for("game_total") == 0.3


def test_a_raised_global_anchor_leaves_totals_alone(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MARKET_ANCHOR", "0.8")
    cfg = Config()
    assert cfg.anchor_for("game_ml") == 0.8
    assert cfg.anchor_for("game_total") == 0.0


def test_anchoring_shrinks_the_bet_probability_not_the_model() -> None:
    p = _pipeline()
    rec = _rec(p, "game_ml", 0.60)
    assert rec.model_prob == 0.60  # what the audit grades the model on
    assert rec.bet_prob is not None and abs(rec.bet_prob - 0.55) < 1e-6
    assert rec.edge is not None and abs(rec.edge - 0.05) < 1e-6


def test_anchor_off_bets_the_model_itself(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_MARKET_ANCHOR", "0")
    p = _pipeline()
    rec = _rec(p, "game_ml", 0.56)
    assert rec.bet_prob == 0.56


# ---- pre-bet CLV -----------------------------------------------------------
def test_drift_gate_is_neutral_without_an_opening_board() -> None:
    keep, reason = DriftGate().allows(None, 0.55)
    assert keep and reason == ""


def test_drift_gate_vetoes_a_side_the_market_left() -> None:
    keep, reason = DriftGate().allows(0.55, 0.52)
    assert not keep
    assert "moved" in reason


def test_drift_gate_keeps_a_side_the_market_came_to() -> None:
    keep, reason = DriftGate().allows(0.50, 0.54)
    assert keep
    assert "clv: OK" in reason


def test_drift_within_tolerance_is_kept() -> None:
    assert DriftGate(min_drift=0.02).allows(0.52, 0.51)[0]


def test_drift_gate_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_CLV_GATE", "0")
    gate = DriftGate.from_env()
    assert not gate.enabled
    assert gate.allows(0.60, 0.40) == (True, "")


def test_drift_tolerance_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_CLV_DRIFT", "0.05")
    assert DriftGate.from_env().allows(0.55, 0.52)[0]


def test_drift_gate_vetoes_a_buy_in_the_pipeline() -> None:
    p = _pipeline()
    p._open_board = {quote_key(MATCHUP, "game_ml", "MIA ML"): 0.56}
    rec = _rec(p, "game_ml", 0.60)
    assert rec.tier is Tier.PASS
    assert any("clv: PASS" in r for r in rec.reasons)


# ---- the opening board -----------------------------------------------------
def test_board_keeps_the_first_price_seen() -> None:
    opened = ClosingQuote(MATCHUP, "game_ml", "MIA ML", -110.0, 0.50)
    later = ClosingQuote(MATCHUP, "game_ml", "MIA ML", -140.0, 0.57)
    other = ClosingQuote(MATCHUP, "game_ml", "ATL ML", 120.0, 0.43)
    board = merge_board({opened.key: opened}, [later, other])
    assert {q.key: q.no_vig_prob for q in board}[opened.key] == 0.50
    assert len(board) == 2


def test_first_run_captures_the_board_and_is_therefore_neutral(tmp_path) -> None:
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    p = _pipeline(cfg)
    quotes = {
        (MATCHUP, "game_ml", "MIA ML"): [
            MarketQuote(book="dk", american=-110.0, opposite_american=-110.0)
        ]
    }
    board = p._record_board(date(2026, 8, 8), quotes)
    key = quote_key(MATCHUP, "game_ml", "MIA ML")
    # The board it just captured is the board it compares against, so nothing
    # can read as drift on a slate's first run.
    assert DriftGate().allows(board[key], evaluate(0.5, quotes[(MATCHUP, "game_ml", "MIA ML")]).fair_prob)[0]
    assert board_path(cfg.audit_dir, date(2026, 8, 8)).exists()


def test_a_later_run_does_not_overwrite_the_open(tmp_path) -> None:
    cfg = Config(data_dir=tmp_path)
    cfg.ensure_dirs()
    p = _pipeline(cfg)
    slate = date(2026, 8, 8)
    key = (MATCHUP, "game_ml", "MIA ML")
    open_quotes = {key: [MarketQuote(book="dk", american=-110.0, opposite_american=-110.0)]}
    p._record_board(slate, open_quotes)
    moved = {key: [MarketQuote(book="dk", american=140.0, opposite_american=-160.0)]}
    board = p._record_board(slate, moved)
    assert board[quote_key(*key)] == 0.5


def test_board_round_trips_through_the_snapshot_format(tmp_path) -> None:
    path = tmp_path / "board.json"
    q = ClosingQuote(MATCHUP, "game_ml", "MIA ML", -110.0, 0.5)
    save_closing(path, [q])
    assert load_closing(path)[q.key] == q


def test_closing_quotes_and_the_board_agree_on_keys() -> None:
    quotes = {
        (MATCHUP, "game_ml", "MIA ML"): [
            MarketQuote(book="dk", american=-110.0, opposite_american=-110.0)
        ]
    }
    assert [q.key for q in closing_quotes(quotes)] == [
        quote_key(MATCHUP, "game_ml", "MIA ML")
    ]
