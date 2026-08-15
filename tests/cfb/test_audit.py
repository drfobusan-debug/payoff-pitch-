"""Grading, ledger idempotency, and closing-line value."""

from __future__ import annotations

from datetime import date

import pytest

from cfb_engine.audit.clv import ClosingQuote, compute_clv, merge_closing
from cfb_engine.audit.grade import build_result_index, grade, result_for
from cfb_engine.audit.ledger import (
    LedgerEntry,
    entries_from_graded,
    load_ledger,
    price_bucket_metrics,
    update_ledger,
)
from cfb_engine.data.cfbd import GameResult
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation

DAY = date(2025, 11, 1)


def _ml(team_side: str, selection: str) -> Recommendation:
    return Recommendation(
        game_date=DAY,
        game_id="g1",
        matchup="Alabama vs Georgia",
        market="game_ml",
        selection=selection,
        model_prob=0.6,
        market_american=-120,
        opposite_american=100,
        ev=0.05,
        edge=0.04,
        fair_prob=0.55,
        tier=Tier.MODERATE,
        team_side=team_side,
        side="win",
        home_abbrev="Georgia",
        away_abbrev="Alabama",
    )


def _result() -> GameResult:
    return GameResult(home="Georgia", away="Alabama", home_points=27, away_points=20)


def test_ml_grading_win_and_loss():
    index = build_result_index([_result()])
    home_pick = _ml("home", "Georgia ML")
    away_pick = _ml("away", "Alabama ML")
    assert grade(home_pick, result_for(home_pick, index)) == "win"
    assert grade(away_pick, result_for(away_pick, index)) == "loss"


def test_ats_push():
    rec = _ml("away", "Alabama +7.0")
    rec.market = "game_ats"
    rec.line = 7.0
    # Alabama lost by exactly 7 -> ATS push.
    assert grade(rec, _result()) == "push"


def test_total_over_under():
    over = _ml("home", "Over 44.5")
    over.market = "game_total"
    over.line = 44.5
    over.side = "over"
    assert grade(over, _result()) == "win"  # 47 total
    under = _ml("home", "Under 44.5")
    under.market = "game_total"
    under.line = 44.5
    under.side = "under"
    assert grade(under, _result()) == "loss"


def test_ledger_is_idempotent(tmp_path):
    path = tmp_path / "ledger.csv"
    graded = [(_ml("home", "Georgia ML"), "win")]
    entries = entries_from_graded(graded, DAY)
    update_ledger(path, entries, DAY)
    update_ledger(path, entries, DAY)  # re-audit same day
    rows = load_ledger(path)
    assert len(rows) == 1
    assert rows[0].result == "win"
    assert rows[0].pnl > 0


def test_clv_positive_when_market_moves_to_us():
    closing = {"game_ml|Georgia ML": ClosingQuote(american=-150, no_vig_prob=0.60)}
    close_odds, close_prob, clv, clv_ev = compute_clv(
        "game_ml", "Georgia ML", bet_american=-120, bet_fair_prob=0.55, closing=closing
    )
    assert close_odds == -150
    assert clv is not None and clv > 0  # 0.60 - 0.55
    assert clv_ev is not None


def test_clv_missing_selection_is_none():
    assert compute_clv("game_ml", "Nobody ML", -120, 0.5, {}) == (None, None, None, None)


def test_merge_closing_keeps_earlier_kickoffs():
    """A Saturday needs several captures, and the noon window must survive them.

    Games already under way have left the pre-match board, so a late-window
    capture returns nothing for them; replacing the file would discard their
    close entirely.
    """
    noon = {"game_ml|Georgia ML": ClosingQuote(american=-150, no_vig_prob=0.60)}
    night = {
        "game_ml|Georgia ML": ClosingQuote(american=-160, no_vig_prob=0.615),
        "game_ml|Oregon ML": ClosingQuote(american=-200, no_vig_prob=0.665),
    }
    merged = merge_closing(noon, night)
    assert set(merged) == {"game_ml|Georgia ML", "game_ml|Oregon ML"}
    assert merged["game_ml|Georgia ML"].no_vig_prob == 0.615
    assert merge_closing(merged, {}) == merged


def _entry(odds: float, result: str, tier: str = Tier.MODERATE.value) -> LedgerEntry:
    return LedgerEntry(
        date=DAY.isoformat(),
        matchup="Alabama vs Georgia",
        category="Moneyline",
        market="game_ml",
        selection="Georgia ML",
        line=None,
        book="pinnacle",
        odds=odds,
        tier=tier,
        model_prob=0.55,
        ev=0.05,
        result=result,
        pnl=round(american_to_decimal(odds) - 1.0, 4) if result == "win" else -1.0,
    )


def test_price_buckets_grade_a_dog_against_what_its_price_demands() -> None:
    """A dog winning 33% is not a leak by itself; the price is the yardstick.

    Two of six +200 dogs is exactly the rate a +200 price charges, so the band
    reads as break-even rather than as a 33% disaster.
    """
    entries = [_entry(200, "win")] * 2 + [_entry(200, "loss")] * 4
    rows = {m.tier: m for m in price_bucket_metrics(entries)}
    mid = rows["Mid dog (+200 to +399)"]
    assert mid.n == 6
    assert mid.win_pct == pytest.approx(1 / 3, abs=1e-3)
    assert mid.required_win_pct == pytest.approx(1 / 3, abs=1e-3)
    assert mid.units == pytest.approx(0.0, abs=1e-3)
    assert rows["All underdogs"].n == 6
    assert "All favorites" not in rows


def test_price_buckets_ignore_rows_that_never_carried_a_price() -> None:
    """The -110 stand-in used for P&L would pile unpriced rows into Pick'em."""
    priced = _entry(-150, "win")
    unpriced = _entry(-150, "win")
    unpriced.odds = None
    rows = price_bucket_metrics([priced, unpriced])
    assert sum(m.n for m in rows if m.tier.startswith(("Heavy", "Favorite", "Pick"))) == 1
