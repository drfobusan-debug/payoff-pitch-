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


def test_a_truncated_card_label_still_finds_its_final_score():
    """The label is capped at 14 characters, and the audit must survive that.

    On the 2026-08-29 board this silently cost three of eight games their grade
    -- ``New Mexico Sta`` never matched ``New Mexico State`` -- including the
    largest bet on the card.
    """
    rec = _ml("away", "New Mexico Sta +31.5")
    rec.market, rec.line = "game_ats", 31.5
    rec.home_abbrev, rec.away_abbrev = "Florida State", "New Mexico Sta"
    index = build_result_index(
        [GameResult(home="Florida State", away="New Mexico State", home_points=34, away_points=17)]
    )

    found = result_for(rec, index)

    assert found is not None
    assert grade(rec, found) == "win"


def test_a_truncated_label_does_not_grab_the_wrong_school():
    """``North Dakota S`` is not North Dakota, and neither is graded on a guess."""
    rec = _ml("away", "North Dakota S ML")
    rec.home_abbrev, rec.away_abbrev = "North Dakota S", "Jacksonville S"
    index = build_result_index(
        [GameResult(home="North Dakota", away="Long Island", home_points=42, away_points=21)]
    )

    assert result_for(rec, index) is None


def test_an_ambiguous_prefix_is_left_ungraded():
    rec = _ml("home", "Miami ML")
    rec.home_abbrev, rec.away_abbrev = "Miami", "Bethune"
    index = build_result_index(
        [
            GameResult(home="Miami (FL)", away="Bethune-Cookman", home_points=30, away_points=10),
            GameResult(home="Miami (OH)", away="Bethune", home_points=20, away_points=17),
        ]
    )

    assert result_for(rec, index) is None


def test_a_truncated_label_does_not_flip_home_and_away():
    """The home/away alignment reads the same names, so it needs the same rule."""
    rec = _ml("home", "Florida State -31.5")
    rec.market, rec.line = "game_ats", -31.5
    rec.home_abbrev, rec.away_abbrev = "Florida State", "New Mexico Sta"
    res = GameResult(home="Florida State", away="New Mexico State", home_points=34, away_points=17)

    assert grade(rec, res) == "loss"  # won by 17, needed 32


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


def test_auditing_a_later_day_keeps_an_earlier_row_whole(tmp_path):
    """Every audit reloads the whole ledger and rewrites it, so a column that is
    written but not read back is erased from history on the next run."""
    path = tmp_path / "ledger.csv"
    rec = _ml("home", "Georgia ML")
    rec.drift, rec.pass_gate = -0.031, "clv_drift"
    entries = entries_from_graded([(rec, "win")], DAY)
    entries[0].clv_pts = 0.5
    update_ledger(path, entries, DAY)

    later = date(2025, 11, 8)
    update_ledger(path, entries_from_graded([(_ml("home", "Georgia ML"), "win")], later), later)

    old = next(r for r in load_ledger(path) if r.date == DAY.isoformat())
    assert (old.drift, old.pass_gate, old.clv_pts) == (-0.031, "clv_drift", 0.5)


def test_clv_positive_when_market_moves_to_us():
    closing = {"game_ml|Georgia ML": ClosingQuote(american=-150, no_vig_prob=0.60)}
    close_odds, close_prob, clv, clv_ev = compute_clv(
        "Georgia @ Alabama",
        "game_ml",
        "Georgia ML",
        bet_american=-120,
        bet_fair_prob=0.55,
        closing=closing,
    ).as_tuple()
    assert close_odds == -150
    assert clv is not None and clv > 0  # 0.60 - 0.55
    assert clv_ev is not None


def test_clv_missing_selection_is_none():
    assert compute_clv("Georgia @ Alabama", "game_ml", "Nobody ML", -120, 0.5, {}).as_tuple() == (
        None,
        None,
        None,
        None,
    )


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
