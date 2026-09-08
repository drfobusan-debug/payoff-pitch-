"""The batter model's surest overs are its worst bets, so the top of the
conviction band is screened off rather than the bottom of it.

Every existing conviction screen is a floor: refuse the cheap ticket, keep the
confident one. The graded ledger says that ordering is inverted on batter props
-- p>=.62 buys are 14 points long and lose 16.5% of stake while the sub-.50
fades pay -- so this screen runs the other way. It is a screen and not a
recalibration: no probability moves, so the refit still grades on the same
basis and can retire it.

Both sides of the line now carry it. The fade half was held back at 40 rows and
graded as a candidate for 27 slates; at 544 graded buys it is -6.3% and negative
in both halves, so it ships with its own gate and its own knob.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlb_engine.config import Config
from mlb_engine.features.drift_gate import DriftGate
from mlb_engine.features.lineup_lock import LineupLockGate
from mlb_engine.features.market_gates import prob_ceiling_allows
from mlb_engine.features.ml_gate import MLPenGate, MLSharpGate
from mlb_engine.market.ev import MarketQuote
from mlb_engine.market.tiers import Tier
from mlb_engine.pipeline import Pipeline

MATCHUP = "MIA @ ATL"


class _Identity:
    def apply(self, market: str, prob: float) -> float:
        return prob


def _pipeline(cfg: Config | None = None) -> Pipeline:
    p = Pipeline.__new__(Pipeline)
    p.cfg = cfg or Config()
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


MARKET = "batter_2b"  # doubles: no no-buy, no singles profile, no RBI floor


def _confident(p: Pipeline, side: str):
    """A .70 batter buy, priced so the ceiling is the only thing refusing it."""
    return _prop(p, side, 0.70, american=-160.0, opposite=140.0)


def _modest(p: Pipeline, side: str):
    """The same side at .61 -- under the ceiling, over every floor."""
    return _prop(p, side, 0.61, american=-130.0, opposite=110.0)


def _prop(
    p: Pipeline,
    side: str,
    over_prob: float,
    *,
    american: float,
    opposite: float,
):
    """``over_prob`` is the over's probability; the fade's is its complement."""
    tag = "o0.5" if side == "over" else "u0.5"
    selection = f"Some Batter {tag} 2B"
    game = SimpleNamespace(game_date="2026-08-08", game_pk=1)
    quotes = {
        (MATCHUP, MARKET, selection): [
            MarketQuote(book="dk", american=american, opposite_american=opposite)
        ]
    }
    return p._mk(
        game, MATCHUP, "batter", MARKET, selection,
        over_prob if side == "over" else 1.0 - over_prob,
        team_side="away", side=side, quotes=quotes,
    )


def test_the_confident_batter_over_is_the_one_refused() -> None:
    ceiling = Config().batter_max_buy_prob
    assert ceiling == 0.62
    assert prob_ceiling_allows(0.55, ceiling)[0]
    assert prob_ceiling_allows(0.619, ceiling)[0]
    keep, reason = prob_ceiling_allows(0.70, ceiling)
    assert not keep
    assert "0.700" in reason


def test_the_ceiling_is_inclusive_where_the_floor_is_not() -> None:
    """.62 itself is inside the losing pocket, so it is refused -- the mirror
    of ``prob_floor_allows``, which admits a probability equal to its floor."""
    assert not prob_ceiling_allows(0.62, 0.62)[0]


def test_a_ceiling_of_one_disables_the_gate() -> None:
    assert prob_ceiling_allows(0.99, 1.0)[0]


def test_missing_inputs_never_create_a_betting_decision() -> None:
    assert prob_ceiling_allows(None, 0.62)[0]


def test_the_confident_batter_fade_is_refused_too() -> None:
    """The candidate reached 544 graded buys at -6.3%, so the fade now ships.

    It carries its own gate name rather than the over side's, because the two
    halves were shipped on separate evidence and have to be liftable separately.
    """
    p = _pipeline()
    rec = _confident(p, "under")
    assert rec.tier is Tier.PASS
    assert rec.pass_gate == "batter_under_prob_ceiling"
    assert _confident(p, "over").pass_gate == "batter_prob_ceiling"


def test_a_modest_batter_fade_still_buys() -> None:
    """Which buy tier is a separate question: this price devigs to .543, under
    ``strong_fair_prob``, so both sides are Moderate rather than Strong."""
    p = _pipeline()
    assert _modest(p, "under").tier is Tier.MODERATE
    assert _modest(p, "over").tier is Tier.MODERATE


def test_the_two_halves_of_the_ceiling_are_retired_separately() -> None:
    """Each side was shipped on its own record, so each lifts on its own."""
    fade_off = _pipeline(Config(batter_under_max_buy_prob=1.0))
    assert _confident(fade_off, "under").tier is Tier.STRONG
    assert _confident(fade_off, "over").pass_gate == "batter_prob_ceiling"

    over_off = _pipeline(Config(batter_max_buy_prob=1.0))
    assert _confident(over_off, "over").tier is Tier.STRONG
    assert _confident(over_off, "under").pass_gate == "batter_under_prob_ceiling"
