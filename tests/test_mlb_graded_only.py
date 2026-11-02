"""MLB ships graded-only, and the 2027 refusals claim their rows first.

Every market starts priced and graded but not bought; a buy the model wanted is
stamped ``graded_only_<market>``. The named screens that run before it -- the
+200 longshot refusal and the elite-arm batter-over refusal -- keep the rows
they refuse, so probation grades each screen on its own refusals.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlb_engine.config import Config, EVThresholds
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market import gradedonly
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

MATCHUP = "CWS @ CHC"
LEVELS_OFF = EVThresholds(min_prob=0.0, max_ev=1.0)


@pytest.fixture(autouse=True)
def _under_the_legacy_anchor(legacy_anchor: None) -> None:
    """Screens are exercised by flipping a buy; see tests/conftest.py."""


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline() -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config(ev=LEVELS_OFF)
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


def _pitcher_k(p: Pipeline, american: float = -110.0, opposite: float = -110.0):
    selection = "Shota Imanaga Strikeouts o5.5"
    game = SimpleNamespace(game_date="2026-08-15", game_pk=1)
    quotes = {
        (MATCHUP, "pitcher_k", selection): [
            MarketQuote(book="dk", american=american, opposite_american=opposite)
        ]
    }
    return p._mk(
        game, MATCHUP, "pitcher", "pitcher_k", selection, 0.70,
        line=5.5, player_id=1, stat="K", side="over", quotes=quotes,
    )


def _batter_2b(p: Pipeline, siera: float | None):
    selection = "Some Batter o0.5 2B"
    game = SimpleNamespace(game_date="2026-08-15", game_pk=1)
    quotes = {
        (MATCHUP, "batter_2b", selection): [
            MarketQuote(book="dk", american=-110.0, opposite_american=-110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", "batter_2b", selection, 0.60,
        team_side="away", side="over", quotes=quotes, opp_starter_siera=siera,
    )


def test_every_market_ships_graded_only(monkeypatch) -> None:
    monkeypatch.delenv("MLBE_GRADED_ONLY_MARKETS", raising=False)
    g = gradedonly.GradedOnly.from_env()
    assert g.covers("game_ml") and g.covers("batter_hr") and g.covers("pitcher_k")


def test_an_empty_list_turns_graded_only_off(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_GRADED_ONLY_MARKETS", "")
    assert not gradedonly.GradedOnly.from_env().covers("game_ml")


def test_a_named_list_covers_only_those_markets(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_GRADED_ONLY_MARKETS", "game_ml, batter_hr")
    g = gradedonly.GradedOnly.from_env()
    assert g.covers("batter_hr") and not g.covers("pitcher_k")


def test_a_would_be_buy_is_kept_as_a_graded_only_pass(monkeypatch) -> None:
    buy = _pitcher_k(_pipeline())
    assert buy.tier is not Tier.PASS
    monkeypatch.setenv("MLBE_GRADED_ONLY_MARKETS", "*")
    rec = _pitcher_k(_pipeline())
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "graded_only_pitcher_k"
    assert gradedonly.is_gate(rec.pass_gate)
    assert f"would have been {buy.tier.value}" in rec.reasons[0]


def test_a_longshot_buy_is_refused_before_graded_only(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_GRADED_ONLY_MARKETS", "*")
    rec = _pitcher_k(_pipeline(), american=210.0, opposite=-260.0)
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "longshot_price"


def test_a_price_short_of_the_longshot_line_is_kept() -> None:
    rec = _pitcher_k(_pipeline(), american=190.0, opposite=-230.0)
    assert rec.pass_gate != "longshot_price"


def test_a_batter_over_against_an_elite_arm_is_refused() -> None:
    soft = _batter_2b(_pipeline(), siera=4.9)
    assert soft.tier is not Tier.PASS
    elite = _batter_2b(_pipeline(), siera=3.6)
    assert elite.tier is Tier.PASS
    assert elite.pass_gate == "batter_over_elite_arm"


def test_an_unknown_starter_is_not_an_elite_one() -> None:
    assert _batter_2b(_pipeline(), siera=None).tier is not Tier.PASS


def test_the_elite_arm_refusal_is_switchable(monkeypatch) -> None:
    monkeypatch.setenv("MLBE_ELITE_ARM_SIERA", "0")
    assert _batter_2b(_pipeline(), siera=3.6).tier is not Tier.PASS
