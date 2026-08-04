"""Grading, ledger idempotency, and closing-line value."""

from __future__ import annotations

from datetime import date

from cfb_engine.audit.clv import ClosingQuote, compute_clv, merge_closing
from cfb_engine.audit.grade import build_result_index, grade, result_for
from cfb_engine.audit.ledger import entries_from_graded, load_ledger, update_ledger
from cfb_engine.data.cfbd import GameResult
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
