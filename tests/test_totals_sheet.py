"""The hand totals sheet: + leans over, - leans under, and the row is the sum."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest
from openpyxl import load_workbook

from mlb_engine.data.parks import PARKS
from mlb_engine.data.vsin import Split, TotalSplit
from mlb_engine.filters.weather import WeatherConditions
from mlb_engine.output import totals_sheet
from mlb_engine.output.totals_audit import BANDS, sheet_bands
from mlb_engine.output.totals_sheet import (
    SheetRow,
    TeamSide,
    barrel_pts,
    book_pts,
    csw_pts,
    kbb_pts,
    sheet_is_current,
    siera_pts,
    weather_pts,
    woba_pts,
    wrc_pts,
    write_workbook,
    xera_pts,
)


def test_a_league_average_arm_and_lineup_score_zero() -> None:
    """2026 medians: team wRC+ 99 / wOBA .315 / Barrel% 7.6; SP SIERA 4.26,
    xERA 4.31, CSW% 26.4, K-BB% 12.4; pens SIERA 3.85, CSW% 27.5, K-BB% 12.8."""
    assert wrc_pts(99) == woba_pts(0.315) == barrel_pts(7.6) == 0
    assert siera_pts(4.26) == xera_pts(4.31) == csw_pts(26.4) == kbb_pts(12.4) == 0
    assert siera_pts(3.85) == csw_pts(27.5) == kbb_pts(12.8) == 0


def test_the_bands_are_symmetric_around_the_centre() -> None:
    assert wrc_pts(110) == 1 and wrc_pts(90) == -1
    assert wrc_pts(135) == 3 and wrc_pts(65) == -3
    assert siera_pts(4.6) == 1 and siera_pts(3.5) == -1
    assert siera_pts(5.5) == 3 and siera_pts(2.5) == -3
    assert kbb_pts(8) == 1 and kbb_pts(17) == -1
    assert csw_pts(24) == 1 and csw_pts(30) == -1
    assert xera_pts(4.8) == 1 and xera_pts(3.4) == -1


def test_sharp_over_handle_is_positive_and_sharp_under_is_negative() -> None:
    over = TotalSplit(8.5, over=Split(76, 56), under=Split(24, 44))
    assert book_pts(over) == 2
    under = TotalSplit(8.5, over=Split(30, 45), under=Split(70, 55))
    assert book_pts(under) == -1
    flat = TotalSplit(8.5, over=Split(52, 50), under=Split(48, 50))
    assert book_pts(flat) == 0
    one_ticket = TotalSplit(8.5, over=Split(100, 100), under=Split(0, 0))
    assert book_pts(one_ticket) == 0
    assert book_pts(None) == 0


def test_a_roof_replaces_the_weather_score_with_minus_one() -> None:
    roofed = next(p for p in PARKS.values() if p.roof == "retractable")
    hot = WeatherConditions(95, 80, 20, 180, 20)
    pts, label = weather_pts(roofed, hot)
    assert pts == -1 and "roof" in label


def test_wind_is_signed_by_direction_and_a_cross_wind_is_nothing() -> None:
    open_park = next(p for p in PARKS.values() if p.roof == "open")
    blowing_in = WeatherConditions(70, 50, 12, 0, -12)
    blowing_out = WeatherConditions(70, 50, 12, 180, 12)
    cross = WeatherConditions(70, 50, 12, 90, 0)
    assert weather_pts(open_park, blowing_in)[0] == -2
    assert weather_pts(open_park, blowing_out)[0] == 2
    assert weather_pts(open_park, cross)[0] == 0
    assert weather_pts(open_park, WeatherConditions(75, 50, 3, 90, 0))[0] == 0


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
        engine_total=9.0,
        engine_p_over=0.5678,
    )
    path = write_workbook([row], Date(2026, 9, 9), tmp_path / "t.xlsx")
    ws = load_workbook(path)["Totals 2026-09-09"]
    header = [c.value for c in ws[1]]
    values = [c.value for c in ws[2]]
    engine_cols = {"Engine", "Eng vs line", "Eng O%"}
    pts = [
        v for h, v in zip(header, values, strict=True)
        if isinstance(v, int) and h != "SUM" and h not in engine_cols
    ]
    assert row.total_pts == 2 * (3 + 5 + 1 + 3 + 1 + 1) + 1 + 2 - 1 - 1
    assert values[header.index("SUM")] == sum(pts) == row.total_pts
    # the engine's read sits beside SUM and is never added to it
    assert header.index("Engine") == header.index("SUM") + 1
    assert values[header.index("Engine")] == 9.0 and values[header.index("Eng vs line")] == 0.5
    assert values[header.index("Eng O%")] == 0.568 and row.engine_delta == 0.5
    assert "Legend" in load_workbook(path).sheetnames
    assert sheet_bands(path) == BANDS and sheet_is_current(path)


def test_a_sheet_on_the_current_bands_is_kept_and_an_older_one_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    day = Date(2026, 9, 10)
    out = tmp_path / f"totals_sheet_{day.isoformat()}.xlsx"
    monkeypatch.setattr(totals_sheet, "output_path", lambda cfg, d: out)
    monkeypatch.setattr(totals_sheet, "MLBStatsClient", lambda: pytest.fail("rebuilt a current sheet"))
    assert not sheet_is_current(out)
    # Rewrite the Legend without the stamp or the Centre rule: an older band set.
    write_workbook([], day, out)
    wb = load_workbook(out)
    for r in wb["Legend"].iter_rows(min_row=2):
        if r[0].value in ("Bands", "Centre"):
            r[0].value, r[1].value = None, None
    wb.save(out)
    assert not sheet_is_current(out)
    write_workbook([], day, out)
    assert sheet_is_current(out)
    assert totals_sheet.build_totals_sheet(None, day, if_stale=True) == out  # type: ignore[arg-type]
