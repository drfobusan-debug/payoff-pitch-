"""The default slate day follows the US clock the slate is filed under, not UTC."""

from __future__ import annotations

from datetime import date, datetime, timezone

from cbb_engine import cli
from cbb_engine.data import capture


def test_an_evening_tick_reads_tonights_slate() -> None:
    # 8:30 pm Eastern on Nov 2 is 01:30Z on Nov 3: tonight's 9 pm tip is the Nov 2 slate.
    now = datetime(2026, 11, 3, 1, 30, tzinfo=timezone.utc)
    assert cli._parse_date(None, now) == date(2026, 11, 2)
    assert capture.slate_date_of("2026-11-03T02:00:00Z") == "2026-11-02"


def test_a_morning_tick_and_an_explicit_date() -> None:
    assert cli._parse_date(None, datetime(2026, 11, 2, 14, 0, tzinfo=timezone.utc)) == date(2026, 11, 2)
    assert cli._parse_date("2026-11-05") == date(2026, 11, 5)
