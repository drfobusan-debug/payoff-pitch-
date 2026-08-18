"""The propsheet archive is only worth keeping if the prices are as printed."""

from __future__ import annotations

import os
from datetime import date as Date
from pathlib import Path

import pandas as pd
import pytest

ps = pytest.importorskip("scripts.propsheet_import")


def sheet(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "SITE": "DraftKings",
        "MARKET": "Hits",
        "PLAYER": "Aaron Judge",
        "TM": "NYY",
        "GAME": "NYY@TOR",
        "OVER": "0.5 (-150)",
        "UNDER": "0.5 (+120)",
        "THE BAT X PROJECTION": 0.61,
        "IMPLIED PROJECTION": 0.58,
        "BATTING ORDER": 2.0,
        "THE BAT X PITCH COUNT": float("nan"),
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def test_prices_are_taken_as_printed_not_derived() -> None:
    rows = ps.to_rows(sheet([{}]), Date(2026, 8, 15))
    assert rows[0]["over_american"] == -150.0
    assert rows[0]["under_american"] == 120.0
    assert rows[0]["line"] == 0.5


def test_a_side_the_book_did_not_print_stays_empty() -> None:
    """Filling the missing side in from the other would invent the vig."""
    rows = ps.to_rows(sheet([{"UNDER": "nan"}]), Date(2026, 8, 15))
    assert len(rows) == 1
    assert rows[0]["over_american"] == -150.0
    assert rows[0]["under_american"] is None


def test_two_sides_on_different_numbers_are_not_a_market() -> None:
    assert ps.to_rows(sheet([{"OVER": "1.5 (+130)", "UNDER": "2.5 (-160)"}]), Date(2026, 8, 15)) == []


def test_an_unknown_market_is_kept_but_never_guessed_at() -> None:
    rows = ps.to_rows(sheet([{"MARKET": "Stolen Bases"}]), Date(2026, 8, 15))
    assert rows[0]["sheet_market"] == "Stolen Bases"
    assert rows[0]["market"] == ""


@pytest.mark.parametrize(
    "sheet_market,market",
    [
        ("Hits", "batter_h"),
        ("Singles", "batter_1b"),
        ("Hits Runs and RBIs", "batter_hrr"),
        ("Pitching Outs", "pitcher_outs"),
        ("Hits Allowed", "pitcher_h"),
        ("Walks Allowed", "pitcher_bb"),
        ("Walks", "batter_bb"),
        ("Strikeouts", "pitcher_k"),
        ("Hitter Strikeouts", "batter_k"),
    ],
)
def test_the_two_walk_and_two_strikeout_markets_do_not_cross(sheet_market: str, market: str) -> None:
    """A hitter's walks and a pitcher's walks allowed print under similar names."""
    rows = ps.to_rows(sheet([{"MARKET": sheet_market}]), Date(2026, 8, 15))
    assert rows[0]["market"] == market


def test_the_sheet_supplies_the_day_and_today_supplies_the_year() -> None:
    assert ps.parse_day("Aug 15", Date(2026, 8, 15)) == Date(2026, 8, 15)


def test_the_year_is_the_nearest_one_not_the_current_one() -> None:
    """A sheet saved either side of New Year must not land eleven months away."""
    assert ps.parse_day("Dec 30", Date(2027, 1, 2)) == Date(2026, 12, 30)
    assert ps.parse_day("Jan 2", Date(2026, 12, 30)) == Date(2027, 1, 2)


def test_a_row_keeps_its_own_date() -> None:
    """A sheet spanning midnight is archived row-accurately."""
    rows = ps.to_rows(sheet([{"DATE": "Aug 15"}, {"DATE": "Aug 16"}]), Date(2026, 8, 15))
    assert [r["date"] for r in rows] == ["2026-08-15", "2026-08-16"]


def test_an_unreadable_date_falls_back_to_the_slate() -> None:
    rows = ps.to_rows(sheet([{"DATE": "later today"}]), Date(2026, 8, 15))
    assert rows[0]["date"] == "2026-08-15"


def test_the_file_is_named_for_the_date_most_games_are_on() -> None:
    frame = sheet([{"DATE": "Aug 15"}, {"DATE": "Aug 15"}, {"DATE": "Aug 16"}])
    assert ps.slate_day(frame, Date(2026, 8, 15)) == Date(2026, 8, 15)


def test_a_sheet_with_no_readable_date_asks_for_one() -> None:
    assert ps.slate_day(sheet([{"DATE": "n/a"}]), Date(2026, 8, 15)) is None
    assert ps.slate_day(pd.DataFrame([{"MARKET": "Hits"}]), Date(2026, 8, 15)) is None


def test_every_page_of_tonights_slate_is_imported(tmp_path: Path) -> None:
    """The sheet paginates, so a slate is several saves, not one."""
    stale = tmp_path / "MLB_Betting_Model_-_Player_Prop_Odd_Predictions.html"
    page1 = tmp_path / "MLB_Betting_Model_-_1Player_Prop_Odd_Predictions.html"
    page2 = tmp_path / "MLB_Betting_Model_-_2Player_Prop_Odd_Predictions.html"
    for path in (stale, page1, page2):
        path.write_text("x")
    os.utime(stale, (1_000_000, 1_000_000))  # yesterday's save, still in the folder
    os.utime(page1, (2_000_000, 2_000_000))
    os.utime(page2, (2_000_060, 2_000_060))
    assert ps.find_saved_sheets(str(tmp_path)) == [page1, page2]


def test_unrelated_downloads_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "bank-statement.html").write_text("no")
    (tmp_path / "propsheet.csv").write_text("no")
    assert ps.find_saved_sheets(str(tmp_path)) == []


def test_a_missing_downloads_folder_is_not_a_crash(tmp_path: Path) -> None:
    assert ps.find_saved_sheets(str(tmp_path / "nope")) == []


def test_an_unpriced_row_is_dropped() -> None:
    assert ps.to_rows(sheet([{"OVER": "nan", "UNDER": "nan"}]), Date(2026, 8, 15)) == []
