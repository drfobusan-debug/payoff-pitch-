"""Unit tests for the lineup-lock staleness read."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mlb_engine.features.lineup_lock import (
    PROJECTED,
    LineupLockGate,
    hours_to_first_pitch,
)
from mlb_engine.output.card import _lineup_note
from mlb_engine.recommendations import Recommendation


def test_hours_to_first_pitch_reads_utc_stamp() -> None:
    now = datetime(2025, 7, 4, 18, 0, tzinfo=timezone.utc)
    hours = hours_to_first_pitch("2025-07-04T23:10:00Z", now=now)
    assert hours is not None
    assert 5.1 < hours < 5.2


def test_hours_negative_once_underway() -> None:
    now = datetime(2025, 7, 4, 23, 30, tzinfo=timezone.utc)
    hours = hours_to_first_pitch("2025-07-04T23:00:00Z", now=now)
    assert hours is not None and hours < 0


def test_hours_none_without_start_time() -> None:
    assert hours_to_first_pitch(None) is None
    assert hours_to_first_pitch("not-a-timestamp") is None


def test_posted_lineup_near_lock_is_fresh() -> None:
    gate = LineupLockGate()
    lock = gate.read(projected=False, hours=0.5)
    assert lock.stale is False
    assert lock.note is None
    keep, reason = gate.allows(lock)
    assert keep is True
    assert reason == ""


def test_projected_lineup_warns_but_keeps_by_default() -> None:
    gate = LineupLockGate()
    lock = gate.read(projected=True, hours=1.0)
    assert lock.status == PROJECTED
    assert lock.stale is True
    keep, reason = gate.allows(lock)
    assert keep is True
    assert "WARN" in reason and "re-run near lock" in reason


def test_early_pricing_is_stale_even_with_posted_lineup() -> None:
    gate = LineupLockGate(stale_hours=3.0)
    lock = gate.read(projected=False, hours=6.0)
    assert lock.stale is True
    assert lock.note is not None and "6.0h before first pitch" in lock.note


def test_demote_flag_passes_stale_moneyline() -> None:
    gate = LineupLockGate(demote=True)
    keep, reason = gate.allows(gate.read(projected=True, hours=None))
    assert keep is False
    assert "PASS" in reason


def test_missing_start_time_with_posted_lineup_stays_neutral() -> None:
    gate = LineupLockGate(demote=True)
    keep, _ = gate.allows(gate.read(projected=False, hours=None))
    assert keep is True


def _rec(**kw: object) -> Recommendation:
    base = dict(
        game_date=datetime.now(timezone.utc).date(),
        game_pk=1,
        matchup="AWY @ HOM",
        category="game",
        market="game_ml",
        selection="HOM ML",
        model_prob=0.55,
    )
    base.update(kw)
    return Recommendation(**base)  # type: ignore[arg-type]


def test_card_note_flags_projected_and_early() -> None:
    note = _lineup_note([_rec(lineup_status="projected", hours_to_first_pitch=7.0)])
    assert note is not None
    assert "projected" in note and "7h before first pitch" in note


def test_card_note_absent_when_fresh() -> None:
    assert _lineup_note([_rec(lineup_status="posted", hours_to_first_pitch=0.5)]) is None


def test_timedelta_import_used_for_relative_stamp() -> None:
    """A relative start time parses the same way as an absolute one."""
    now = datetime(2025, 7, 4, 18, 0, tzinfo=timezone.utc)
    stamp = (now + timedelta(hours=2)).isoformat()
    hours = hours_to_first_pitch(stamp, now=now)
    assert hours is not None
    assert abs(hours - 2.0) < 1e-6
