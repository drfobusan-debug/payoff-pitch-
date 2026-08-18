"""Doubles overs are refused at +300 and longer.

The doubles book is calibrated everywhere except where it bets: across 6,656
graded ``batter_2b o0.5`` rows the model's .140 band hits 14.0% and its .188
band 17.7%, but its .258 band -- the one the buy list is drawn from -- hits
15.0% on 346 rows. The selection consequently adds nothing, with bought rows at
14.3% (n=70) against passed rows' 14.2% (n=6,586), so the screen is a price
ceiling rather than a band: no price pocket survived, and at these lengths a
fraction of a point of probability error is a fifth of the stake.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.market_gates import price_ceiling_allows
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

REFUSE_AT = 300.0
MATCHUP = "MIA @ ATL"
SELECTION = "Jazz Chisholm 2B o0.5"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _doubles_rec(american: float, *, gate_reason: str | None = None):
    """A priced ``batter_2b`` over, run through the real selection chain."""
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
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
    quotes = {
        (MATCHUP, "batter_2b", SELECTION): [
            MarketQuote(book="dk", american=american, opposite_american=-110.0)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", "batter_2b", SELECTION, 0.30,
        line=0.5, player_id=1, stat="2B", side="over", quotes=quotes,
        gate_reason=gate_reason,
    )


def test_a_long_doubles_price_is_refused() -> None:
    keep, reason = price_ceiling_allows(455, REFUSE_AT, "doubles-price-ceiling")
    assert not keep
    assert "+455" in reason


def test_the_ceiling_is_exclusive() -> None:
    """'+300 or longer' is the rule, so +300 itself is refused."""
    assert not price_ceiling_allows(300, REFUSE_AT)[0]
    assert price_ceiling_allows(299, REFUSE_AT)[0]


def test_a_short_doubles_price_is_still_buyable() -> None:
    """The door is left open: a hitter the book prices near even is not this bet."""
    assert price_ceiling_allows(150, REFUSE_AT)[0]


def test_an_unpriced_selection_is_left_to_the_other_screens() -> None:
    keep, reason = price_ceiling_allows(None, REFUSE_AT)
    assert keep
    assert reason == ""


def test_the_shipped_default_is_the_measured_cutoff() -> None:
    assert Config().doubles_max_buy_odds == REFUSE_AT


def test_the_screen_can_be_lifted_from_the_environment(monkeypatch) -> None:
    """A screen the ledger condemns has to be reversible without a release."""
    monkeypatch.setenv("MLBE_DOUBLES_MAX_BUY_ODDS", "100000")
    assert price_ceiling_allows(650, Config().doubles_max_buy_odds)[0]


def test_the_refusal_is_named_in_the_ledger() -> None:
    rec = _doubles_rec(455.0)
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "doubles_price_ceiling"
    assert any("doubles-price-ceiling" in r for r in rec.reasons)


def test_a_row_another_screen_already_refused_keeps_its_own_gate() -> None:
    """The screen runs last so probation credits it only with its own refusals.

    A row the contact floor would have thrown out is not evidence about the
    price ceiling, and crediting it there inflates the refusals the ledger
    grades this screen on.
    """
    rec = _doubles_rec(455.0, gate_reason="contact floor: PASS")
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "contact_floor"
