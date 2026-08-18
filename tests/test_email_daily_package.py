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
