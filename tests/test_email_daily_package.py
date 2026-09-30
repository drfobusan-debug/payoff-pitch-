"""The morning email is the delivery, so an artifact it does not collect is not sent."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from scripts.email_daily_package import collect_attachments, totals_audit_note


def test_the_power_screen_rides_in_the_package(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    for name in ("mlb_recommendations_2026-08-17.xlsx", "power_screen_2026-08-17.pdf"):
        (tmp_path / name).write_bytes(b"x")
    names = [n for n, _ in collect_attachments(tmp_path, day)]
    assert names == [
        "mlb_recommendations_2026-08-17.xlsx",
        "power_screen_2026-08-17.pdf",
    ]


def test_the_totals_sheet_rides_once_a_day_with_the_daily_package(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    for name in ("mlb_recommendations_2026-08-17.xlsx", "totals_sheet_2026-08-17.xlsx"):
        (tmp_path / name).write_bytes(b"x")
    daily = [n for n, _ in collect_attachments(tmp_path, day, "evening", with_daily=True)]
    assert "totals_sheet_2026-08-17.xlsx" in daily
    block = [n for n, _ in collect_attachments(tmp_path, day, "evening", with_daily=False)]
    assert "totals_sheet_2026-08-17.xlsx" not in block


def test_the_totals_audit_and_its_summary_ride_only_with_the_daily_package(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    (tmp_path / "mlb_recommendations_2026-08-17.xlsx").write_bytes(b"x")
    (tmp_path / "totals_audit_2026-08-17.xlsx").write_bytes(b"x")
    (tmp_path / "totals_audit_2026-08-17.txt").write_text("Totals sheet 2026-08-16: 12 graded\n")
    daily = [n for n, _ in collect_attachments(tmp_path, day, "evening", with_daily=True)]
    assert "totals_audit_2026-08-17.xlsx" in daily
    assert "totals_audit_2026-08-17.xlsx" not in [
        n for n, _ in collect_attachments(tmp_path, day, "evening", with_daily=False)
    ]
    assert totals_audit_note(tmp_path, day) == "Totals sheet 2026-08-16: 12 graded"
    assert totals_audit_note(tmp_path, day, with_daily=False) == ""
    assert totals_audit_note(tmp_path, Date(2026, 8, 18)) == ""


def test_yesterdays_screen_is_not_todays(tmp_path: Path) -> None:
    """Dated by slate, not newest on disk: a failed screen must send nothing."""
    (tmp_path / "power_screen_2026-08-16.pdf").write_bytes(b"x")
    assert collect_attachments(tmp_path, Date(2026, 8, 17)) == []


def test_a_block_email_carries_that_blocks_slate_and_not_the_whole(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    for name in (
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17.pdf",
        "PayoffPitch_Slate_2026-08-17_evening.pdf",
        "PayoffPitch_Slate_2026-08-17_evening.mp3",
        "PayoffPitch_Slate_2026-08-17_late.pdf",
        "PayoffPitch_Regression_2026-08-17.pdf",
        "power_screen_2026-08-17.pdf",
    ):
        (tmp_path / name).write_bytes(b"x")
    names = [n for n, _ in collect_attachments(tmp_path, day, "evening", with_daily=False)]
    assert names == [
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17_evening.pdf",
        "PayoffPitch_Slate_2026-08-17_evening.mp3",
    ]
    assert "PayoffPitch_Slate_2026-08-17_evening.pdf" not in [
        n for n, _ in collect_attachments(tmp_path, day)
    ]


def test_the_once_a_day_pieces_ride_only_with_the_first_pass(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    for name in (
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17_matinee.pdf",
        "PayoffPitch_Slate_2026-08-17_evening.pdf",
        "PayoffPitch_Regression_2026-08-17.pdf",
        "power_screen_2026-08-17.pdf",
    ):
        (tmp_path / name).write_bytes(b"x")
    first = [n for n, _ in collect_attachments(tmp_path, day, "matinee", with_daily=True)]
    assert first == [
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17_matinee.pdf",
        "PayoffPitch_Regression_2026-08-17.pdf",
        "power_screen_2026-08-17.pdf",
    ]
    later = [n for n, _ in collect_attachments(tmp_path, day, "evening", with_daily=False)]
    assert later == [
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17_evening.pdf",
    ]


def test_a_pass_with_no_games_sends_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    import scripts.email_daily_package as edp

    (tmp_path / "mlb_recommendations_2026-08-17.xlsx").write_bytes(b"x")
    (tmp_path / "power_screen_2026-08-17.pdf").write_bytes(b"x")

    class Cfg:
        output_dir = tmp_path

    monkeypatch.setattr(edp, "load_config", lambda: Cfg())
    sent: list[object] = []
    monkeypatch.setattr(edp, "send_card_email", lambda *a, **k: sent.append(a))
    assert edp.main(["prog", "2026-08-17", "--block", "late", "--with-daily"]) == 0
    assert sent == []
    assert "priced no games" in capsys.readouterr().out


def test_the_block_word_is_not_read_as_the_date() -> None:
    from scripts.email_daily_package import _block, _resolve_day

    argv = ["prog", "2026-08-17", "--block", "afternoon"]
    assert _resolve_day(Path("/nonexistent"), argv) == Date(2026, 8, 17)
    assert _block(argv) == "afternoon"
    assert _block(["prog", "2026-08-17"]) is None
    with pytest.raises(SystemExit):
        _block(["prog", "2026-08-17", "--block", "night"])
