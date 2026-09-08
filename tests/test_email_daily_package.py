"""The morning email is the delivery, so an artifact it does not collect is not sent."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from scripts.email_daily_package import collect_attachments


def test_the_power_screen_rides_in_the_package(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    for name in ("mlb_recommendations_2026-08-17.xlsx", "power_screen_2026-08-17.pdf"):
        (tmp_path / name).write_bytes(b"x")
    names = [n for n, _ in collect_attachments(tmp_path, day)]
    assert names == [
        "mlb_recommendations_2026-08-17.xlsx",
        "power_screen_2026-08-17.pdf",
    ]


def test_yesterdays_screen_is_not_todays(tmp_path: Path) -> None:
    """Dated by slate, not newest on disk: a failed screen must send nothing."""
    (tmp_path / "power_screen_2026-08-16.pdf").write_bytes(b"x")
    assert collect_attachments(tmp_path, Date(2026, 8, 17)) == []


def test_a_block_email_carries_that_blocks_slate_and_not_the_whole(tmp_path: Path) -> None:
    day = Date(2026, 8, 17)
    for name in (
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17.pdf",
        "PayoffPitch_Slate_2026-08-17_night.pdf",
        "PayoffPitch_Slate_2026-08-17_night.mp3",
        "PayoffPitch_Regression_2026-08-17.pdf",
    ):
        (tmp_path / name).write_bytes(b"x")
    names = [n for n, _ in collect_attachments(tmp_path, day, "night")]
    assert names == [
        "mlb_recommendations_2026-08-17.xlsx",
        "PayoffPitch_Slate_2026-08-17_night.pdf",
        "PayoffPitch_Slate_2026-08-17_night.mp3",
        "PayoffPitch_Regression_2026-08-17.pdf",
    ]
    assert "PayoffPitch_Slate_2026-08-17_night.pdf" not in [
        n for n, _ in collect_attachments(tmp_path, day)
    ]


def test_the_block_word_is_not_read_as_the_date() -> None:
    from scripts.email_daily_package import _block, _resolve_day

    argv = ["prog", "2026-08-17", "--block", "day"]
    assert _resolve_day(Path("/nonexistent"), argv) == Date(2026, 8, 17)
    assert _block(argv) == "day"
    assert _block(["prog", "2026-08-17"]) is None
