"""The batter model's surest overs are its worst bets, so the top of the
conviction band is screened off rather than the bottom of it.

Every existing conviction screen is a floor: refuse the cheap ticket, keep the
confident one. The graded ledger says that ordering is inverted on batter props
-- p>=.62 buys are 14 points long and lose 16.5% of stake while the sub-.50
fades pay -- so this screen runs the other way. It is a screen and not a
recalibration: no probability moves, so the refit still grades on the same
basis and can retire it.
"""

from __future__ import annotations

from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.audit.probation import CANDIDATE_SCREENS
from mlb_engine.config import Config
from mlb_engine.features.market_gates import prob_ceiling_allows


def _row(selection: str, model_prob: float, market: str = "batter_h") -> LedgerEntry:
    return LedgerEntry(
        date="2026-08-15",
        matchup="AWAY @ HOME",
        category="prop",
        market=market,
        selection=selection,
        line=1.5,
        book="dk",
        odds=-160.0,
        tier="Strong buy",
        model_prob=model_prob,
        ev=0.05,
        result="loss",
        pnl=-1.0,
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


def test_the_fade_half_is_graded_rather_than_assumed() -> None:
    """40 rows is the right sign and the wrong sample, so the under side ships
    as a candidate screen the audit re-grades every night, refusing the fades
    the live gate deliberately leaves alone."""
    (cand,) = [c for c in CANDIDATE_SCREENS if c.name == "batter_under_prob_ceiling_0.62"]
    assert cand.refuses(_row("Aaron Judge H u1.5", 0.70))
    assert not cand.refuses(_row("Aaron Judge H o1.5", 0.70))
    assert not cand.refuses(_row("Aaron Judge H u1.5", 0.55))
    assert not cand.refuses(_row("Someone Ks u5.5", 0.70, market="pitcher_k"))
