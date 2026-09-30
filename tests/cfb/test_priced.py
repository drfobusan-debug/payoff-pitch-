"""The money record: what the prices did, as distinct from what the model picked."""

from __future__ import annotations

from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.priced import (
    CONTRADICTION_ROI,
    contradictions,
    engine_priced_stat,
    priced_buys,
    priced_findings,
    priced_stats,
)
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier


def _entry(
    *,
    odds: float | None = -110,
    result: str = "win",
    market: str = "game_ml",
    tier: str = Tier.MODERATE.value,
    date: str = "2025-09-06",
    under_odds: float | None = -110,
    clv: float | None = None,
    clv_pts: float | None = None,
    drift: float | None = None,
    pass_gate: str | None = None,
) -> LedgerEntry:
    pnl = 0.0
    if result == "win" and odds is not None:
        pnl = round(american_to_decimal(odds) - 1.0, 4)
    elif result == "loss":
        pnl = -1.0
    return LedgerEntry(
        date=date,
        matchup="Alabama vs Georgia",
        category="Moneyline",
        market=market,
        selection="Georgia ML",
        line=None,
        book="pinnacle",
        odds=odds,
        under_odds=under_odds,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl=pnl,
        clv=clv,
        clv_pts=clv_pts,
        drift=drift,
        pass_gate=pass_gate,
    )


def test_only_the_rows_that_were_actually_bets_are_counted() -> None:
    """A pass has no stake, an unpriced row was graded at a price nobody offered,
    and a push returned nothing -- none of the three belongs in a P/L."""
    entries = [
        _entry(),
        _entry(tier=Tier.PASS.value),
        _entry(odds=None),  # unpriced row
        _entry(result="push"),
    ]
    kept = priced_buys(entries)
    assert len(kept) == 1
    assert kept[0].tier == Tier.MODERATE.value


def test_a_winning_record_at_a_short_price_is_still_a_loss() -> None:
    """The whole reason this table exists.

    Six -200 bets going 4-2 is a 66.7% win rate, which reads like an edge --
    against the 66.7% a -200 price demands it is exactly break-even, and one
    more loss makes it a loser while still winning "most" of its bets.
    """
    rows = [_entry(odds=-200, result="win")] * 4 + [_entry(odds=-200, result="loss")] * 3
    stat = engine_priced_stat(rows)
    assert stat.n == 7
    assert round(stat.win_rate, 3) == 0.571
    assert round(stat.breakeven, 3) == 0.667
    assert stat.shortfall < 0
    assert stat.roi < 0


def test_one_way_quotes_are_counted_apart_rather_than_dropped() -> None:
    """They were real bets with real units, but their edge was measured against
    a vigged number, so they cannot share a devigged column with the rest."""
    rows = [
        _entry(under_odds=None, result="loss"),
        _entry(under_odds=None, result="loss"),
        _entry(result="win"),
    ]
    stat = engine_priced_stat(rows)
    assert stat.n == 3
    assert stat.n_one_way == 2
    assert stat.two_sided == 1
    assert stat.units_one_way == -2.0
    prose = " ".join(priced_findings([stat, *priced_stats(rows)], min_n=1))
    # Counted once: the engine-wide row is the sum of the market rows, not a
    # market of its own.
    assert "2 of these bets were one-way quotes" in prose


def test_points_of_line_value_are_summarised_beside_probability_clv() -> None:
    """A football bet is shopped in points: -3 to -3.5 at the same -110 is a real
    loss of value that the probability column cannot see."""
    rows = [
        _entry(clv=0.01, clv_pts=0.5),
        _entry(clv=-0.01, clv_pts=-1.0),
        _entry(clv=None, clv_pts=None),
    ]
    stat = engine_priced_stat(rows)
    assert stat.n_clv == 2
    assert stat.beat_close == 1
    assert stat.n_clv_pts == 2
    assert stat.beat_number == 1
    assert round(stat.clv_pts, 3) == -0.25


def test_markets_are_reported_biggest_sample_first() -> None:
    rows = [_entry(market="game_ats") for _ in range(3)] + [_entry(market="game_ml")]
    stats = priced_stats(rows, lambda k: {"game_ats": "ATS"}.get(k, k))
    assert [s.key for s in stats] == ["game_ats", "game_ml"]
    assert stats[0].label == "ATS"


def test_the_engine_wide_row_is_not_read_as_one_more_market() -> None:
    """It is the sum of the market rows, so counting it doubles every total."""
    rows = [_entry(market="game_ats", result="loss") for _ in range(20)]
    stats = [engine_priced_stat(rows), *priced_stats(rows)]
    assert len(priced_findings(stats)) == 1


def test_a_thin_market_is_not_given_a_finding() -> None:
    """Ten coin flips is not a number about the market, so it gets no prose."""
    stats = priced_stats([_entry(result="loss") for _ in range(5)])
    assert priced_findings(stats) == []


def test_positive_ppv_lift_with_a_negative_return_is_named() -> None:
    """The side selection is right and the price already knows.

    Left unnamed, the lift column is an argument for betting more of exactly the
    market that costs the most.
    """
    losing = [_entry(odds=-200, result="loss")] * 8 + [_entry(odds=-200, result="win")] * 8
    stats = priced_stats(losing)
    named = contradictions(stats, {"game_ml": 0.05}, min_n=10)
    assert [s.key for s, _ in named] == ["game_ml"]
    assert stats[0].roi <= CONTRADICTION_ROI
    # No lift reported for the market means nothing to contradict.
    assert contradictions(stats, {}, min_n=10) == []
