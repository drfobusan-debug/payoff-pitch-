"""A slate that buys nothing has to say which stage emptied it.

Zero buys reads identically whether the books never quoted the board, the board
was quoted and paid nothing, or the engine's own screens refused every row that
paid -- three different problems with three different fixes. The funnel counts
the stages so the card names the closing gate instead of implying an empty slate.
"""

from __future__ import annotations

from datetime import date as Date

from mlb_engine.audit import funnel as F
from mlb_engine.config import EVThresholds
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation


def _rec(
    market: str,
    *,
    priced: bool = True,
    gate: str | None = None,
    tier: Tier = Tier.PASS,
    ev: float = 0.05,
) -> Recommendation:
    return Recommendation(
        game_date=Date(2026, 9, 1),
        game_pk=1,
        matchup="AAA @ BBB",
        category="game",
        market=market,
        selection=f"{market} over",
        model_prob=0.6,
        market_american=-110.0 if priced else None,
        ev=ev,
        pass_gate=gate,
        tier=tier,
    )


# --- the stages are counted, and they are not the same stage ----------------
def test_an_unquoted_row_is_not_a_refusal() -> None:
    """Book coverage is not a screen: charging it to one blames the engine."""
    f = F.build([_rec("game_total", priced=False)])
    assert f.overall.candidates == 1
    assert f.overall.priced == 0
    assert f.overall.positive_ev == 0
    assert f.overall.gates[F.UNPRICED] == 1


def test_the_ev_floor_is_a_stage_and_not_a_screen() -> None:
    """A price that does not pay is the absence of a bet, not a bet refused."""
    f = F.build([_rec("game_total", gate="ev_floor", ev=-0.02)])
    assert f.overall.priced == 1
    assert f.overall.positive_ev == 0
    assert f.overall.cleared_price_screen == 0


def test_the_ev_stage_reads_the_row_and_not_the_gate_name() -> None:
    """Rendered under a floor the row was not priced under, the count follows
    the EV: a row paying +0.05 against a 0.10 floor has no bet in it, and one
    paying +0.05 against a floor of zero does, whatever the gate says."""
    row = [_rec("game_total", gate="ev_floor")]
    assert F.build(row, EVThresholds(min_ev=0.10)).overall.positive_ev == 0
    assert F.build(row, EVThresholds()).overall.positive_ev == 1


def test_a_price_screen_refusal_counts_as_positive_ev() -> None:
    f = F.build([_rec("game_total", gate="prob_floor")])
    assert f.overall.positive_ev == 1
    assert f.overall.cleared_price_screen == 0
    assert f.overall.closing_gate == "prob_floor"


def test_a_market_gate_refuses_a_row_that_already_cleared_the_price() -> None:
    """The distinction the 09-01 slate turned on: 239 rows cleared the price
    screen and every one died downstream."""
    f = F.build([_rec("batter_rbi", gate="batter_under_prob_ceiling")])
    assert f.overall.cleared_price_screen == 1
    assert f.overall.buys == 0
    assert f.overall.closing_gate == "batter_under_prob_ceiling"


def test_a_buy_is_counted_once_and_refused_by_nothing() -> None:
    f = F.build([_rec("game_total", tier=Tier.STRONG)])
    assert f.overall.buys == 1
    assert not f.overall.gates
    assert f.overall.closing_gate == ""


def test_the_markets_split_the_overall_row_for_row() -> None:
    recs = [
        _rec("game_total", gate="prob_floor"),
        _rec("game_total", gate="ev_floor"),
        _rec("f5_total", tier=Tier.MODERATE),
    ]
    f = F.build(recs)
    assert f.overall.candidates == 3
    assert sum(m.candidates for m in f.markets) == 3
    assert sum(m.buys for m in f.markets) == f.overall.buys == 1
    # Best-covered market first, so the table opens where the board is.
    assert [m.market for m in f.markets] == ["game_total", "f5_total"]


# --- the notes that make a zero-buy slate actionable ------------------------
def test_the_geometry_is_reported_when_the_thresholds_exclude_a_coinflip() -> None:
    """.58 floor against a .08 ceiling needs the market's own price at .50 --
    on a -110/-110 total the window is a single point wide."""
    note = F.geometry_note(EVThresholds(min_prob=0.58, max_edge=0.08))
    assert "0.50" in note
    # Both screens are strict, so the boundary is one point wide, not empty --
    # the note says so rather than claiming an impossibility.
    assert "exactly on the ceiling" in note
    assert "no row can clear" in F.geometry_note(EVThresholds(min_prob=0.60, max_edge=0.08))
    assert not F.geometry_note(EVThresholds(min_prob=0.55, max_edge=0.08))


def test_the_clock_names_itself_as_a_re_run_rather_than_a_verdict() -> None:
    f = F.build([_rec("game_total", gate="lineup_clock")])
    assert "--within-hours" in F.clock_note(f)
    assert F.clock_note(F.build([_rec("game_total", gate="prob_floor")])) == ""


def test_the_clock_advises_the_window_the_gate_is_actually_using(monkeypatch) -> None:
    """Sending the operator back inside a 3h window when the gate is set to 6
    would have them re-run outside it."""
    monkeypatch.setenv("MLBE_LINEUP_STALE_HOURS", "6")
    note = F.clock_note(F.build([_rec("game_total", gate="lineup_clock")]))
    assert "--within-hours 6" in note


def test_the_table_does_not_call_a_missing_price_a_gate() -> None:
    md = "\n".join(F.markdown(F.build([_rec("game_total", priced=False)])))
    assert "no book price" in md
