"""Tests for the clock gate and the late re-pricing pass that answers it.

Over the ledger's 915 graded buys carrying a first-pitch stamp, the ones priced
inside three hours returned +5.7% (n=369) against -14.9% (n=546) priced earlier.
So a buy is now refused for being early -- which is only honest if a later pass
re-prices the game and folds it back into the same card, which is what
``mlb-engine run --within-hours`` does.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from mlb_engine.cli import _merge_late_pass
from mlb_engine.config import Config
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline
from mlb_engine.recommendations import Recommendation, load_json, save_json

MATCHUP = "MIA @ ATL"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline(gate: LineupLockGate | None = None) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p._calibrator = _Identity()
    p._shrink = None
    p._splits = {}
    p._ml_gate = MLSharpGate.from_env()
    p._pen_gate = MLPenGate.from_env()
    p._lineup_gate = gate or LineupLockGate()
    p._lineup_lock = None
    p._drift_gate = DriftGate.from_env()
    p._open_board = {}
    return p


def _rec(p: Pipeline, model_prob: float = 0.65) -> Recommendation:
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        # A price the market itself calls a favourite, which is the only place a
        # buy can clear the conviction floor and stay under the edge cap.
        (MATCHUP, "game_ml", "MIA ML"): [
            MarketQuote(book="dk", american=-110.0, opposite_american=130.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "game", "game_ml", "MIA ML", model_prob,
        team_side="away", side="win", quotes=quotes,
    )


# ---- the gate --------------------------------------------------------------
def test_the_clock_gate_ships_on() -> None:
    assert LineupLockGate.from_env().clock is True


def test_an_early_price_is_refused_in_every_market() -> None:
    gate = LineupLockGate()
    keep, reason = gate.clock_allows(gate.read(projected=False, hours=7.5))
    assert keep is False
    assert "7.5h out" in reason and "-14.9%" in reason


def test_a_price_near_lock_is_kept() -> None:
    gate = LineupLockGate()
    keep, reason = gate.clock_allows(gate.read(projected=False, hours=1.2))
    assert keep is True
    assert reason == ""


def test_a_slate_with_no_start_times_is_never_refused_by_the_clock() -> None:
    """Every backtest and any slate the feed hasn't stamped lands here."""
    gate = LineupLockGate()
    assert gate.clock_allows(gate.read(projected=True, hours=None))[0] is True
    assert gate.clock_allows(None)[0] is True


def test_the_clock_gate_is_switchable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_LINEUP_CLOCK_GATE", "0")
    gate = LineupLockGate.from_env()
    assert gate.clock_allows(gate.read(projected=False, hours=9.0))[0] is True


def test_the_window_moves_with_the_staleness_setting(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_LINEUP_STALE_HOURS", "6")
    gate = LineupLockGate.from_env()
    assert gate.clock_allows(gate.read(projected=False, hours=5.0))[0] is True
    assert gate.clock_allows(gate.read(projected=False, hours=6.5))[0] is False


# ---- the gate inside the pipeline -----------------------------------------
def test_an_early_buy_is_passed_and_attributed_to_the_clock() -> None:
    p = _pipeline()
    p._lineup_lock = p._lineup_gate.read(projected=True, hours=8.0)
    rec = _rec(p)
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "lineup_clock"
    # Priced, graded and auditable: the ledger measures the refusal.
    assert rec.market_american == -110.0 and rec.ev is not None
    assert rec.hours_to_first_pitch is None  # stamped per game, not per row


def test_the_same_buy_survives_inside_the_window() -> None:
    p = _pipeline()
    p._lineup_lock = p._lineup_gate.read(projected=False, hours=1.0)
    assert _rec(p).tier is not Tier.PASS


# ---- which games a late pass prices ---------------------------------------
def _stamp(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def test_a_late_pass_prices_only_the_games_it_can_still_bet() -> None:
    assert Pipeline._starts_within(_stamp(2.0), 3.0) is True
    assert Pipeline._starts_within(_stamp(4.0), 3.0) is False
    # Underway: its pre-match board is gone, so a price for it prices nothing.
    assert Pipeline._starts_within(_stamp(-0.5), 3.0) is False
    # No start time: priced, and left to the gate that can read it.
    assert Pipeline._starts_within(None, 3.0) is True


# ---- folding the pass into the card --------------------------------------
def _pred(game_pk: int, selection: str) -> Recommendation:
    return Recommendation(
        game_date=date(2026, 8, 8),
        game_pk=game_pk,
        matchup=MATCHUP,
        category="game",
        market="game_ml",
        selection=selection,
        model_prob=0.55,
    )


def test_the_late_pass_replaces_its_games_and_keeps_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "predictions_2026-08-08.json"
    save_json([_pred(1, "morning 1"), _pred(2, "morning 2")], path)
    merged = _merge_late_pass([_pred(2, "late 2")], path)
    assert [(r.game_pk, r.selection) for r in merged] == [
        (1, "morning 1"),
        (2, "late 2"),
    ]


def test_the_late_pass_stands_alone_without_an_earlier_card(tmp_path: Path) -> None:
    recs = [_pred(2, "late 2")]
    assert _merge_late_pass(recs, tmp_path / "missing.json") == recs


def test_an_unreadable_card_does_not_cost_the_late_pass_its_bets(tmp_path: Path) -> None:
    path = tmp_path / "predictions_2026-08-08.json"
    path.write_text("{not json")
    recs = [_pred(2, "late 2")]
    assert _merge_late_pass(recs, path) == recs


def test_the_merged_card_round_trips_for_the_audit(tmp_path: Path) -> None:
    """What the pass writes is what tomorrow's audit grades."""
    path = tmp_path / "predictions_2026-08-08.json"
    save_json([_pred(1, "morning 1")], path)
    merged = _merge_late_pass([_pred(2, "late 2")], path)
    save_json(merged, path)
    assert [r.selection for r in load_json(path)] == ["morning 1", "late 2"]
