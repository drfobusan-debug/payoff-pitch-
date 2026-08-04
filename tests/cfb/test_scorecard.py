"""PPV/NPV-by-market scorecard, scored against the price break-even."""

from __future__ import annotations

from datetime import date

from cfb_engine.audit.grade import LOSS, WIN
from cfb_engine.audit.scorecard import build_scorecard
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation


def _rec(market: str, tier: Tier, american: float) -> Recommendation:
    return Recommendation(
        game_date=date(2025, 9, 6),
        game_id="g",
        matchup="A@B",
        market=market,
        selection="x",
        model_prob=0.6,
        market_american=american,
        tier=tier,
    )


def test_scorecard_splits_by_market_and_scores_vs_breakeven():
    graded = [
        (_rec("game_ml", Tier.STRONG, -110), WIN),
        (_rec("game_ml", Tier.STRONG, -110), WIN),
        (_rec("game_ml", Tier.STRONG, -110), LOSS),
        (_rec("game_ats", Tier.MODERATE, -110), LOSS),
        (_rec("game_total", Tier.PASS, -110), WIN),
    ]
    rows = build_scorecard(graded, date(2025, 9, 6))
    by = {(r.market, r.tier): r for r in rows}

    ml_strong = by[("ML", Tier.STRONG.value)]
    assert ml_strong.n == 3 and ml_strong.wins == 2
    assert abs(ml_strong.ppv - 2 / 3) < 1e-4
    # -110 => break-even ~52.4%; a 66.7% PPV clears it.
    assert abs(ml_strong.breakeven - 0.5238) < 1e-3
    assert ml_strong.edge_vs_be > 0

    # Market split is present and independent of the "All" rows.
    assert ("ATS", Tier.MODERATE.value) in by
    assert ("Totals", Tier.PASS.value) in by
    assert ("All", "Buy (S+M)") in by
