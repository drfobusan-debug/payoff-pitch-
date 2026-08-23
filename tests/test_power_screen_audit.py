"""The ledger audit must read the receipt the way the note's own scorecard does."""

from __future__ import annotations

from datetime import date as Date

import pytest

from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.audit.power_ledger import GradedPosition, Position

audit = pytest.importorskip("scripts.power_screen_audit")


def _position(
    *,
    stat: str = "H",
    tier: str = "Pass",
    day: str = "2026-08-22",
    odds: float = 100.0,
    model: float = 0.5,
    fair: float | None = 0.5,
) -> Position:
    return Position(
        date=day,
        batter="Some Hitter",
        player_id=1,
        game_pk=2,
        stat=stat,
        line=0.5,
        side="over",
        book="book",
        odds=odds,
        model_prob=model,
        fair_prob=fair,
        edge=None,
        ev=None,
        tier=tier,
        rating="HOLD",
        devigged=fair is not None,
    )


def _graded(position: Position, result: str) -> GradedPosition:
    return GradedPosition(
        position=position,
        result=result,
        actual=1 if result == WIN else 0,
        units=1.0 if result == WIN else -1.0,
    )


def test_a_day_outside_the_range_is_not_audited() -> None:
    p = _position(day="2026-08-22")
    assert audit._in_range(p, Date(2026, 8, 22), Date(2026, 8, 22))
    assert not audit._in_range(p, Date(2026, 8, 23), None)
    assert not audit._in_range(p, None, Date(2026, 8, 21))
    assert audit._in_range(p, None, None)


def test_the_printed_probability_falls_back_to_the_model_on_an_old_row() -> None:
    """A row recorded before the board carried the anchored number still grades."""
    assert _position(model=0.42).shown_prob == pytest.approx(0.42)


def test_a_pass_is_not_counted_as_a_bet() -> None:
    bet = _graded(_position(tier="Strong buy"), WIN)
    passed = _graded(_position(tier="Pass"), WIN)
    assert bet.position.is_buy
    assert not passed.position.is_buy


def test_the_record_line_reports_the_price_it_was_shown_at() -> None:
    line = audit._line("all", [_graded(_position(), WIN), _graded(_position(), LOSS)])
    assert "1-1" in line
    assert "+0.00u" in line


def test_only_a_two_sided_price_is_scored_against_the_market() -> None:
    """A one-way quote has no no-vig mark, so it cannot answer that question."""
    one_way = _graded(_position(fair=None), WIN)
    two_sided = _graded(_position(fair=0.4), LOSS)
    assert audit._priced([one_way, two_sided]) == [(two_sided, 0)]
