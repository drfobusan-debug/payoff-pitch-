"""The per-game close: stamped captures, and CLV only on a close near kickoff."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cfb_engine.audit.clv import ClosingQuote, close_due, timely_close, timely_clv, timely_only
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.snapshot import load, save

KICK = "2026-10-24T16:00:00Z"


@pytest.mark.parametrize(
    ("captured", "timely"),
    [
        ("2026-10-24T15:50:00Z", True),  # ten minutes out: the close
        ("2026-10-24T15:15:00Z", True),  # 45 minutes out: still inside
        ("2026-10-24T15:00:00Z", False),  # an hour out: the 11:00 snapshot
        ("2026-10-24T16:00:00Z", False),  # at kickoff
        ("", False),  # written before the stamp existed
    ],
)
def test_only_a_close_inside_45_minutes_of_kickoff_counts(captured: str, timely: bool) -> None:
    assert timely_close(ClosingQuote(-110.0, 0.5, -3.5, captured, KICK)) is timely


def test_stale_sides_are_dropped_and_counted() -> None:
    closing = {
        "A @ B|game_ats|B": ClosingQuote(-110.0, 0.5, -3.5, "2026-10-24T15:50:00Z", KICK),
        "A @ B|game_ml|B": ClosingQuote(-160.0, 0.6, None, "2026-10-24T11:00:00Z", KICK),
    }
    kept, dropped = timely_only(closing)
    assert list(kept) == ["A @ B|game_ats|B"] and dropped == 1


def test_the_stamps_survive_the_round_trip_and_old_files_load_blank(tmp_path: Path) -> None:
    path = tmp_path / "closing.json"
    save({"A @ B|game_ml|B": ClosingQuote(-160.0, 0.6, None, "2026-10-24T15:50:00Z", KICK)}, path)
    assert load(path)["A @ B|game_ml|B"].kickoff_utc == KICK
    path.write_text('{"A @ B|game_ml|B": {"american": -160.0, "no_vig_prob": 0.6, "line": null}}')
    assert load(path)["A @ B|game_ml|B"].captured_at == ""


def test_a_game_is_due_once_inside_its_window() -> None:
    now = datetime(2026, 10, 24, 15, 48, tzinfo=timezone.utc)
    kicks = {"A @ B": KICK, "C @ D": "2026-10-24T19:30:00Z", "E @ F": "2026-10-24T15:30:00Z"}
    assert close_due(kicks, {}, now, 15) == ["A @ B"]
    done = {"A @ B|game_ml|B": ClosingQuote(-160.0, 0.6, None, "2026-10-24T15:47:00Z", KICK)}
    assert close_due(kicks, done, now, 15) == []
    early = {"A @ B|game_ml|B": ClosingQuote(-160.0, 0.6, None, "2026-10-24T11:00:00Z", KICK)}
    assert close_due(kicks, early, now, 15) == ["A @ B"]


def test_a_game_stays_due_until_every_priced_side_has_its_close() -> None:
    now = datetime(2026, 10, 24, 15, 48, tzinfo=timezone.utc)
    sides = {"A @ B": {"A @ B|game_ml|B", "A @ B|game_ats|B"}}
    partial = {"A @ B|game_ml|B": ClosingQuote(-160.0, 0.6, None, "2026-10-24T15:47:00Z", KICK)}
    assert close_due({"A @ B": KICK}, partial, now, 15, sides) == ["A @ B"]
    full = {
        **partial,
        "A @ B|game_ats|B": ClosingQuote(-110.0, 0.5, -3.5, "2026-10-24T15:47:00Z", KICK),
    }
    assert close_due({"A @ B": KICK}, full, now, 15, sides) == []


def _row(date: str, matchup: str, market: str, clv: float | None) -> LedgerEntry:
    return LedgerEntry(
        date,
        matchup,
        "Spread (ATS)",
        market,
        "B -3.5",
        -3.5,
        "dk",
        -110.0,
        "Strong buy",
        0.6,
        0.05,
        "win",
        0.91,
        clv=clv,
        clv_ev=clv,
        clv_pts=0.5 if clv is not None else None,
        close_odds=-110.0 if clv is not None else None,
        close_prob=0.5 if clv is not None else None,
    )


def test_reports_read_clv_only_where_the_close_was_timely() -> None:
    closes = {
        "2026-10-17": {"A @ B|game_ats|B": ClosingQuote(-110.0, 0.5, -3.5)},  # unstamped
        "2026-10-24": {
            "A @ B|game_ats|B": ClosingQuote(-110.0, 0.5, -3.5, "2026-10-24T15:50:00Z", KICK),
            "C @ D|game_ats|D": ClosingQuote(-110.0, 0.5, -3.5, "2026-10-24T13:00:00Z", KICK),
        },
    }
    rows = [
        _row("2026-10-17", "A @ B", "game_ats", 0.02),
        _row("2026-10-24", "A @ B", "game_ats", 0.03),
        _row("2026-10-24", "C @ D", "game_ats", 0.04),
        _row("2026-10-24", "E @ F", "game_ats", None),
    ]
    out = timely_clv(rows, lambda iso: closes.get(iso, {}))
    assert [(e.clv, e.clv_pts) for e in out] == [
        (None, None),
        (0.03, 0.5),
        (None, None),
        (None, None),
    ]
    assert [e.close_prob for e in out] == [None, 0.5, None, None]  # no stale close in Briers
    assert rows[0].clv == 0.02  # the ledger's own rows are left alone
