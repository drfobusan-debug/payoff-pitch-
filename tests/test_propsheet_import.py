"""The propsheet archive is only worth keeping if the prices are as printed."""

from __future__ import annotations

from datetime import date as Date

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


def test_the_slate_date_comes_from_the_caller() -> None:
    """The sheet prints 'Aug 15' with no year, so it cannot supply the date."""
    rows = ps.to_rows(sheet([{}]), Date(2026, 8, 15))
    assert rows[0]["date"] == "2026-08-15"


def test_an_unpriced_row_is_dropped() -> None:
    assert ps.to_rows(sheet([{"OVER": "nan", "UNDER": "nan"}]), Date(2026, 8, 15)) == []
