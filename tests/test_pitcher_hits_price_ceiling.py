"""Hits-allowed overs are refused at plus money.

Over 60 graded ``pitcher_h`` over-buys the record splits on the price rather
than on the pitcher: even money or shorter went 52.9% for -0.5% ROI (n=35),
plus money 26.7% for -40.7% (n=25). The long price is nearly always the 5.5
line, which needs six hits -- bad contact *and* a start long enough to allow
it, two things the simulator draws independently -- so past even money the
model is selling a joint event at the price of one of its halves.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config, EVThresholds
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.market_gates import price_ceiling_allows
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

REFUSE_AT = -100.0
MATCHUP = "CWS @ CHC"
OVER = "Shota Imanaga Hits o5.5"
UNDER = "Shota Imanaga Hits u5.5"

# These rows are a 50% model against a plus-money price, which the shipped
# conviction floor and EV ceiling refuse on their own. The screen under test here
# is the price ceiling, so the level screens are lifted while it is exercised.
LEVELS_OFF = EVThresholds(min_prob=0.0, max_ev=1.0)


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _hits_rec(
    american: float,
    *,
    side: str = "over",
    model_prob: float = 0.50,
    gate_reason: str | None = None,
):
    """A priced ``pitcher_h`` selection, run through the real selection chain."""
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
    game = SimpleNamespace(game_date="2026-08-15", game_pk=1)
    selection = OVER if side == "over" else UNDER
    quotes = {
        (MATCHUP, "pitcher_h", selection): [
            MarketQuote(book="dk", american=american, opposite_american=-110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "pitcher", "pitcher_h", selection, model_prob,
        line=5.5, player_id=1, stat="H", side=side, quotes=quotes,
        gate_reason=gate_reason,
    )


def test_a_plus_money_hits_over_is_refused() -> None:
    keep, reason = price_ceiling_allows(135, REFUSE_AT, "pitcher-hits-price-ceiling")
    assert not keep
    assert "+135" in reason


def test_the_ceiling_refuses_from_even_money_out() -> None:
    """'-100 or longer, don't' -- the band that held is shorter than even money.

    No graded buy sat exactly on -100, so the boundary states the rule rather
    than fitting a cutoff: the 35 buys shorter than it went -0.5%, the 25 at or
    beyond it -40.7%.
    """
    assert not price_ceiling_allows(-100, REFUSE_AT)[0]
    assert price_ceiling_allows(-101, REFUSE_AT)[0]


def test_a_short_hits_over_is_still_buyable() -> None:
    assert price_ceiling_allows(-140, REFUSE_AT)[0]


def test_an_unpriced_selection_is_left_to_the_other_screens() -> None:
    keep, reason = price_ceiling_allows(None, REFUSE_AT)
    assert keep
    assert reason == ""


def test_the_shipped_default_is_the_measured_cutoff() -> None:
    assert Config().pitcher_hits_max_buy_odds == REFUSE_AT


def test_the_screen_can_be_lifted_from_the_environment(monkeypatch) -> None:
    """A screen the ledger condemns has to be reversible without a release."""
    monkeypatch.setenv("MLBE_PITCHER_HITS_MAX_BUY_ODDS", "100000")
    assert price_ceiling_allows(135, Config().pitcher_hits_max_buy_odds)[0]


def test_the_refusal_is_named_in_the_ledger() -> None:
    rec = _hits_rec(135.0)
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "pitcher_hits_price_ceiling"
    assert any("pitcher-hits-price-ceiling" in r for r in rec.reasons)


def test_the_under_is_untouched() -> None:
    """The screen is about paying a long price for a joint event, not the market.

    The two graded under-buys won, and an under at plus money is the opposite
    bet: a short start makes it, so there is nothing joint to overpay for.
    """
    rec = _hits_rec(135.0, side="under")
    assert rec.tier is not Tier.PASS
    assert rec.pass_gate is None


def test_a_row_another_screen_already_refused_keeps_its_own_gate() -> None:
    """The screen runs last so probation credits it only with its own refusals."""
    rec = _hits_rec(135.0, gate_reason="thin starter: 97 pitches")
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "contact_floor"
