"""The three level screens the whole-ledger audit produced, and their order.

Over the 1,619 real-priced graded buys that carry a devigged fair price the book
went 46.8% for -7.2% per unit, and it lost at every claimed-probability band by
more than the rows it passed at the same band: the engine was buying its own
disagreement with the price. Three screens answer that, and they are one rule:

  * ``MLBE_MARKET_ANCHOR`` -- bet a probability pulled 30% toward the devigged
    price, without moving the model's own number
  * ``MLBE_MIN_PROB`` -- that anchored probability has to reach 0.58
  * ``MLBE_MAX_EV`` -- and its claimed EV must not exceed 0.25

Together they take those 1,619 buys to 487 at 61.2% for +1.0% per unit. Each is
switchable per market, because the evidence behind them is a month long.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config, EVThresholds
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import EVResult, MarketQuote
from mlb_engine.market.tiers import Tier, classify, price_screen
from mlb_engine.pipeline import Pipeline

MATCHUP = "MIA @ ATL"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _res(prob: float, ev: float, edge: float = 0.05) -> EVResult:
    return EVResult(
        model_prob=prob,
        best_quote=MarketQuote(book="bk", american=-110.0),
        decimal=1.91,
        ev=ev,
        fair_prob=prob - edge,
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
    american: float = -140.0,
    opposite: float = 130.0,
    selection: str = "MIA ML",
):
    """One priced selection through the real chain. -140/+130 devigs to 0.5729."""
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        (MATCHUP, market, selection): [
            MarketQuote(book="dk", american=american, opposite_american=opposite)
        ]
    }
    return p._mk(
        game, MATCHUP, "game", market, selection, model_prob,
        team_side="home", side="win", quotes=quotes,
    )


# ---- (a) the blend --------------------------------------------------------
def test_the_bet_is_the_blend_and_the_model_keeps_its_own_number() -> None:
    """Anchoring is a selection change, so calibration and audit are untouched.

    0.65 pulled 30% toward a 0.5729 fair price is 0.6269, and it is that number
    the EV, the edge and the floor are read on -- while ``model_prob`` stays the
    model's, which is what the PPV/NPV audit and the calibration refit grade.
    """
    rec = _rec(_pipeline(), "game_ml", 0.65)
    assert rec.model_prob == 0.65
    assert rec.raw_prob == 0.65
    assert rec.fair_prob is not None and abs(rec.fair_prob - 0.5729) < 1e-3
    assert rec.bet_prob is not None and abs(rec.bet_prob - 0.6269) < 1e-3
    # Edge and EV are priced off the blend, not off the model.
    assert rec.edge is not None and abs(rec.edge - 0.054) < 1e-3
    assert rec.ev is not None and abs(rec.ev - 0.0747) < 1e-3
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)


def test_totals_are_still_bet_on_the_model_itself() -> None:
    """The one family where the model out-forecast the price keeps its own number."""
    rec = _rec(_pipeline(), "game_total", 0.65, selection="Over 8.5")
    assert rec.bet_prob == 0.65
    assert Config().anchor_for("game_total") == 0.0


# ---- (b) the conviction floor ---------------------------------------------
def test_a_low_conviction_buy_is_refused_and_named() -> None:
    """A positive-EV row under the floor is passed, with a gradeable gate name.

    -110 against a +100 other side devigs to 0.5116, so 0.60 anchors to 0.5735:
    a 6-point edge at a payable price, and a level the audit says not to buy.
    """
    rec = _rec(_pipeline(), "game_ml", 0.60, american=-110.0, opposite=100.0)
    assert rec.ev is not None and rec.ev > 0
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "prob_floor"
    assert any("bet prob" in r for r in rec.reasons)
    # A pass is still a shadow bet: the price and the numbers are kept.
    assert rec.market_american == -110.0
    assert rec.edge is not None


def test_the_floor_reads_the_blend_not_the_model() -> None:
    """A model above the floor whose blend is not is refused -- that is the point.

    0.60 clears 0.58 on its own, but pulled 30% toward a 0.5116 fair price it is
    0.5735: a selection the market does not join. Those are the rows that lost.
    """
    rec = _rec(_pipeline(), "game_ml", 0.60, american=-110.0, opposite=100.0)
    assert rec.model_prob == 0.60 > EVThresholds().min_prob
    assert rec.bet_prob is not None and rec.bet_prob < EVThresholds().min_prob
    assert rec.pass_gate == "prob_floor"


def test_the_floor_is_per_market_and_reversible(monkeypatch) -> None:
    base = EVThresholds()
    assert base.min_prob == 0.58
    assert base.for_market("batter_hr").min_prob == 0.58
    monkeypatch.setenv("MLBE_MIN_PROB_BATTER_HR", "0")
    assert EVThresholds().for_market("batter_hr").min_prob == 0.0
    monkeypatch.setenv("MLBE_MIN_PROB", "0")
    assert EVThresholds().for_market("game_ml").min_prob == 0.0


def test_the_floor_off_lets_the_same_row_through() -> None:
    """So the screen is what refused it, not the price or the edge."""
    p = _pipeline(Config(ev=EVThresholds(min_prob=0.0)))
    rec = _rec(p, "game_ml", 0.60, american=-110.0, opposite=100.0)
    assert rec.tier in (Tier.STRONG, Tier.MODERATE)


# ---- (c) the EV ceiling ---------------------------------------------------
def test_an_outsized_claimed_ev_is_refused_and_named() -> None:
    thr = EVThresholds()
    tier, reasons = classify(_res(0.62, ev=0.40), thr)
    assert tier is Tier.PASS
    assert any("EV +0.400 > 0.25" in r for r in reasons)
    assert price_screen(_res(0.62, ev=0.40), thr) == (
        "ev_ceiling", "EV +0.400 > 0.25 -> pass"
    )


def test_the_ceiling_leaves_an_ordinary_ev_alone() -> None:
    assert classify(_res(0.62, ev=0.10), EVThresholds())[0] is Tier.STRONG


def test_the_ceiling_is_per_market_and_reversible(monkeypatch) -> None:
    assert EVThresholds().for_market("batter_hr").max_ev == 0.25
    monkeypatch.setenv("MLBE_MAX_EV_BATTER_HR", "1")
    assert EVThresholds().for_market("batter_hr").max_ev == 1.0
    assert classify(_res(0.62, ev=0.40), EVThresholds(max_ev=1.0))[0] is Tier.STRONG


# ---- order, so probation grades the right screen --------------------------
def test_the_relative_screens_still_run_first() -> None:
    """A row that fails EV, edge or price is named for that, not for its level.

    The two level screens are last in ``price_screen`` on purpose: the audit
    grades a refusal by its gate, so a negative-EV row must keep ``ev_floor``
    even though its probability is also under the conviction floor.
    """
    thr = EVThresholds()
    assert price_screen(_res(0.40, ev=-0.05), thr)[0] == "ev_floor"
    assert price_screen(_res(0.40, ev=0.05, edge=0.005), thr)[0] == "thin_edge"
    assert price_screen(_res(0.40, ev=0.05, edge=0.20), thr)[0] == "edge_ceiling"
    # And with the relative screens satisfied, the level screens are reached in
    # their own order: the floor before the ceiling.
    assert price_screen(_res(0.50, ev=0.30), thr)[0] == "prob_floor"
    assert price_screen(_res(0.62, ev=0.30), thr)[0] == "ev_ceiling"
