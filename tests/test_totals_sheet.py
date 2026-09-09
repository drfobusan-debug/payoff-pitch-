"""The hand totals sheet: + leans over, - leans under, and the row is the sum."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from openpyxl import load_workbook

from mlb_engine.data.parks import PARKS
from mlb_engine.data.vsin import Split, TotalSplit
from mlb_engine.filters.weather import WeatherConditions
from mlb_engine.output.totals_sheet import (
    SheetRow,
    TeamSide,
    book_pts,
    weather_pts,
    write_workbook,
)


def test_sharp_over_handle_is_positive_and_sharp_under_is_negative() -> None:
    over = TotalSplit(8.5, over=Split(76, 56), under=Split(24, 44))
    assert book_pts(over) == 2
    under = TotalSplit(8.5, over=Split(30, 45), under=Split(70, 55))
    assert book_pts(under) == -1
    flat = TotalSplit(8.5, over=Split(52, 50), under=Split(48, 50))
    assert book_pts(flat) == 0
    assert book_pts(None) == 0


def test_a_roof_replaces_the_weather_score_with_minus_one() -> None:
    roofed = next(p for p in PARKS.values() if p.roof == "retractable")
    hot = WeatherConditions(95, 80, 20, 180, 20)
    pts, label = weather_pts(roofed, hot)
    assert pts == -1 and "roof" in label


def test_wind_blowing_in_scores_minus_two_and_out_keeps_its_speed_points() -> None:
    open_park = next(p for p in PARKS.values() if p.roof == "open")
    blowing_in = WeatherConditions(70, 50, 12, 0, -12)
    blowing_out = WeatherConditions(70, 50, 12, 180, 12)
    assert weather_pts(open_park, blowing_in)[0] == -2
    assert weather_pts(open_park, blowing_out)[0] == 2


def test_the_workbook_row_sum_is_the_sum_of_its_signed_columns(tmp_path: Path) -> None:
    side = TeamSide(
        "AZ",
        "Zac Gallen (R)",
        off=3,
        sp=5,
        rp=1,
        kbb_sp=3,
        kbb_rp=1,
        bsr_pg=4.4,
        fatigue=1,
        fatigue_detail="pen",
    )
    row = SheetRow(
        "AZ @ KC",
        None,
        "8.5",
        side,
        side,
        circa=1,
        dk=2,
        weather=-1,
        weather_detail="wx",
        park=0,
        park_detail="Kauffman",
        bsr=0,
        ump=-1,
        ump_name="Ump",
        ump_status="posted",
        ump_detail="",
    )
    path = write_workbook([row], Date(2026, 9, 9), tmp_path / "t.xlsx")
    ws = load_workbook(path)["Totals 2026-09-09"]
    header = [c.value for c in ws[1]]
    values = [c.value for c in ws[2]]
    pts = [v for h, v in zip(header, values, strict=True) if isinstance(v, int) and h != "SUM"]
    assert row.total_pts == 2 * (3 + 5 + 1 + 3 + 1 + 1) + 1 + 2 - 1 - 1
    assert values[header.index("SUM")] == sum(pts) == row.total_pts
    assert "Legend" in load_workbook(path).sheetnames
