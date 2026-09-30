from __future__ import annotations

from datetime import date

import pytest

from mlb_engine.audit import priced
from mlb_engine.audit.ledger import LedgerEntry
from mlb_engine.market.tiers import Tier
from mlb_engine.output import audit_insight as ai


def _entry(
    *,
    market: str = "batter_h",
    odds: float | None = -110,
    tier: str = Tier.STRONG.value,
    result: str = "win",
    pnl: float = 0.909,
    under_odds: float | None = -110,
    clv: float | None = None,
    source: str = "engine",
) -> LedgerEntry:
    return LedgerEntry(
        date=date(2026, 8, 18).isoformat(),
        matchup="AAA @ BBB",
        category="batter",
        market=market,
        selection="x",
        line=1.5,
        book="dk",
        odds=odds,
        tier=tier,
        model_prob=0.6,
        ev=0.05,
        result=result,
        pnl=pnl,
        under_odds=under_odds,
        clv=clv,
        source=source,
    )


def test_breakeven_comes_from_each_bets_own_price():
    # -250 needs 71.4%, and winning 3 of 4 at it still loses money.
    rows = [
        _entry(odds=-250, result="win", pnl=0.4),
        _entry(odds=-250, result="win", pnl=0.4),
        _entry(odds=-250, result="win", pnl=0.4),
        _entry(odds=-250, result="loss", pnl=-1.0),
    ]
    st = priced.engine_priced_stat(rows)
    assert st.n == 4
    assert st.win_rate == pytest.approx(0.75)
    assert st.breakeven == pytest.approx(0.7143, abs=1e-4)
    assert st.units == pytest.approx(0.2)
    assert st.roi == pytest.approx(0.05)
    assert st.shortfall > 0


def test_a_high_hit_rate_can_still_lose_units():
    rows = [_entry(odds=-400, result="win", pnl=0.25) for _ in range(6)] + [
        _entry(odds=-400, result="loss", pnl=-1.0) for _ in range(2)
    ]
    st = priced.engine_priced_stat(rows)
    assert st.win_rate == pytest.approx(0.75)  # would look excellent as PPV
    assert st.units < 0  # and it is a loss
    assert st.shortfall < 0


def test_passes_unpriced_pushes_and_outside_sources_are_excluded():
    rows = [
        _entry(),
        _entry(tier=Tier.PASS.value),
        _entry(odds=None),
        _entry(result="push", pnl=0.0),
        _entry(source="batx"),
    ]
    assert len(priced.priced_buys(rows)) == 1


def test_moderate_counts_as_a_buy():
    rows = [_entry(tier=Tier.MODERATE.value)]
    assert len(priced.priced_buys(rows)) == 1


def test_one_way_rows_are_kept_but_counted_apart():
    rows = [
        _entry(under_odds=None, result="loss", pnl=-1.0),
        _entry(under_odds=None, result="loss", pnl=-1.0),
        _entry(result="win", pnl=0.909),
    ]
    st = priced.engine_priced_stat(rows)
    assert (st.n, st.n_one_way, st.two_sided) == (3, 2, 1)
    assert st.units_one_way == pytest.approx(-2.0)


def test_clv_only_counts_rows_that_have_a_close():
    rows = [
        _entry(clv=0.02),
        _entry(clv=-0.01),
        _entry(clv=None),
    ]
    st = priced.engine_priced_stat(rows)
    assert st.n_clv == 2
    assert st.clv == pytest.approx(0.005)
    assert st.beat_close == 1
    assert st.clv_rate == pytest.approx(0.5)


def test_stats_are_grouped_and_labelled_per_market():
    rows = [_entry(market="batter_h") for _ in range(3)] + [
        _entry(market="batter_tb") for _ in range(2)
    ]
    stats = priced.priced_stats(rows, ai.market_label)
    assert [s.n for s in stats] == [3, 2]  # biggest sample first
    assert stats[0].label == ai.market_label("batter_h")


def test_empty_history_is_reported_not_crashed():
    st = priced.engine_priced_stat([])
    assert st.n == 0
    assert priced.priced_stats([]) == []
    assert priced.priced_findings([]) == []
    import math

    assert math.isnan(st.win_rate)


def test_positive_lift_with_negative_roi_is_flagged():
    rows = [_entry(market="batter_tb", odds=-250, result="win", pnl=0.4)] * 12 + [
        _entry(market="batter_tb", odds=-250, result="loss", pnl=-1.0)
    ] * 8
    stats = priced.priced_stats(rows, ai.market_label)
    clash = priced.contradictions(stats, {"batter_tb": 0.42})
    assert [s.key for s, _ in clash] == ["batter_tb"]
    # ...and is not flagged when the market is profitable
    good = [_entry(market="batter_tb", odds=125, result="win", pnl=1.25)] * 20
    ok = priced.contradictions(priced.priced_stats(good), {"batter_tb": 0.42})
    assert ok == []


def test_an_unmapped_market_key_is_prettified_not_printed_raw():
    assert ai.market_label("batter_h") == "Batter hits"
    assert ai.market_label("batter_hrrbi") == "Batter hrrbi"
    assert "_" not in ai.market_label("pitcher_props")


def test_the_one_way_sentence_counts_every_market_the_total_does():
    """The prose is read against the total row, so thin markets can't drop out."""
    rows = [_entry(market="batter_h", under_odds=None) for _ in range(20)] + [
        _entry(market="pitcher_outs", under_odds=None) for _ in range(2)
    ]
    stats = priced.priced_stats(rows, ai.market_label)
    total = priced.engine_priced_stat(rows)
    said = [f for f in priced.priced_findings(stats) if "one-way quotes" in f]
    assert said and f"{total.n_one_way} of these bets" in said[0]


def test_report_carries_the_priced_section_and_leads_with_units():
    df = ai.classify(ai.graded_to_frame(_graded_frame_rows(), date(2026, 8, 18)))
    rows = [_entry(odds=-250, result="win", pnl=0.4)] * 10 + [
        _entry(odds=-250, result="loss", pnl=-1.0)
    ] * 10
    html, narr = ai.build_report(date(2026, 8, 18), df, df, rows)
    assert "What the Prices Did" in html
    assert "None of that is a betting return" in html
    assert "classification" in html
    assert "batting average is not a return" in narr
    # and without a ledger it says so rather than inventing a P/L
    html2, narr2 = ai.build_report(date(2026, 8, 18), df, df)
    assert "the money column is empty" in html2


def _graded_frame_rows():
    from mlb_engine.recommendations import Recommendation

    out = []
    for i in range(12):
        for prob, res in ((0.7, "win"), (0.6, "loss"), (0.3, "win"), (0.2, "loss")):
            out.append(
                (
                    Recommendation(
                        game_date=date(2026, 8, 18),
                        game_pk=i,
                        matchup="AAA @ BBB",
                        category="batter",
                        market="batter_h",
                        selection="x",
                        model_prob=prob,
                    ),
                    res,
                )
            )
    return out
