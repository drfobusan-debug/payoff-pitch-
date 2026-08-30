"""Tests for the batter-prop provenance cap: a hitter who may not bat is smaller.

Graded buys carrying a provenance stamp run posted -1.6% (n=391) against
projected -9.9% (n=774), an 8.3pp gap at ~1.4 SE that the newer half of the
sample cannot reproduce (posted -14.7% vs projected -15.9%). So a projected
lineup caps a batter prop at Moderate rather than refusing it, and leaves pitcher
props -- profitable under both provenances -- alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

MATCHUP = "MIA @ ATL"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline() -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p._calibrator = _Identity()
    p._shrink = None
    p._splits = {}
    p._ml_gate = MLSharpGate.from_env()
    p._pen_gate = MLPenGate.from_env()
    p._lineup_gate = LineupLockGate()
    p._lineup_lock = None
    p._drift_gate = DriftGate.from_env()
    p._open_board = {}
    return p


def _rec(p: Pipeline, market: str, selection: str, model_prob: float = 0.61):
    """A buy threading every batter screen: -130 at model .61 is Strong.

    Above the conviction floor and the EV floor, under the edge ceiling and the
    .62 batter probability ceiling, so the provenance cap is the only thing that
    can move the tier.
    """
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        (MATCHUP, market, selection): [
            MarketQuote(book="dk", american=-130.0, opposite_american=110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", market, selection, model_prob,
        team_side="away", side="over", quotes=quotes,
    )


# ---- the cap itself --------------------------------------------------------
def test_the_provenance_cap_ships_on() -> None:
    assert LineupLockGate.from_env().provenance_cap is True


def test_a_projected_batter_prop_is_capped() -> None:
    gate = LineupLockGate()
    cap, reason = gate.caps_at_moderate(
        gate.read(projected=True, hours=1.0), "batter_2b"
    )
    assert cap is True
    assert "projected lineup" in reason and "-9.9%" in reason


def test_a_posted_batter_prop_is_left_alone() -> None:
    gate = LineupLockGate()
    lock = gate.read(projected=False, hours=1.0)
    assert gate.caps_at_moderate(lock, "batter_2b") == (False, "")


def test_a_pitcher_prop_is_never_capped_on_a_projected_lineup() -> None:
    """Both provenances profit there, and a starter is named days ahead."""
    gate = LineupLockGate()
    lock = gate.read(projected=True, hours=1.0)
    assert gate.caps_at_moderate(lock, "pitcher_k")[0] is False
    assert gate.caps_at_moderate(lock, "game_ml")[0] is False


def test_an_unrecorded_provenance_is_not_a_projected_one() -> None:
    """Every backtest lands here: the cap refuses a measured status, not a gap."""
    assert LineupLockGate().caps_at_moderate(None, "batter_2b")[0] is False


def test_the_cap_is_switchable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_LINEUP_PROVENANCE_CAP", "0")
    gate = LineupLockGate.from_env()
    lock = gate.read(projected=True, hours=1.0)
    assert gate.caps_at_moderate(lock, "batter_2b")[0] is False


# ---- the cap inside the pipeline ------------------------------------------
def test_a_strong_batter_buy_on_a_projected_lineup_becomes_moderate() -> None:
    p = _pipeline()
    p._lineup_lock = p._lineup_gate.read(projected=False, hours=1.0)
    posted = _rec(p, "batter_2b", "Some Batter o0.5 2B")
    assert posted.tier is Tier.STRONG

    p._lineup_lock = p._lineup_gate.read(projected=True, hours=1.0)
    projected = _rec(p, "batter_2b", "Some Batter o0.5 2B")
    assert projected.tier is Tier.MODERATE
    # The reason travels with the row, and _attach_context stamps the status the
    # ledger splits on, so the cap is gradeable against the buys it shrank.
    assert any("MODERATE cap" in r for r in projected.reasons)
