"""The hand totals sheet: + leans over, - leans under, and the row is the sum."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest
from openpyxl import load_workbook

from mlb_engine.audit.ledger import LedgerEntry
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
    middle_relief,
    outs_under,
    sheet_is_current,
    short_start,
    siera_pts,
    weather_pts,
    woba_pts,
    wrc_pts,
    write_workbook,
    xera_pts,
)
from mlb_engine.schemas import Pitcher


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
        pen_detail="full pen",
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


def _rel(name: str, team: str, ip: float, siera: float, sv: int = 0, hld: int = 0, pid: int = 0) -> dict:
    return {
        "Name": f'<a href="p">{name}</a>', "Team": f'<a href="x">{team}</a>', "IP": ip, "SIERA": siera,
        "xERA": siera + 0.1, "xFIP": siera + 0.3, "C+SwStr%": 0.27, "K-BB%": 0.13,
        "SV": sv, "HLD": hld, "xMLBAMID": pid or hash(name) % 10_000,
    }


def test_middle_relief_leaves_out_the_closer_the_setup_arms_and_the_cups_of_coffee() -> None:
    rows = [
        _rel("Closer", "ATH", 60, 2.5, sv=30, pid=1),
        _rel("Setup1", "ATH", 55, 3.0, hld=25, pid=2),
        _rel("Setup2", "ATH", 50, 3.2, hld=20, pid=3),
        _rel("Bulk1", "ATH", 40, 4.8, pid=4),
        _rel("Bulk2", "ATH", 20, 5.4, pid=5),
        _rel("Cup", "ATH", 4, 1.0, pid=6),
        _rel("Traded", "- - -", 30, 6.0, pid=7),
        _rel("Ace", "PHI", 30, 3.3, pid=8),
    ]
    leverage, mid = middle_relief(rows)
    assert leverage["ATH"] == [1, 2, 3]
    ath = mid["ATH"]
    assert ath.arms == ["Bulk1", "Bulk2"] and ath.ip == 60
    assert ath.siera == pytest.approx((4.8 * 40 + 5.4 * 20) / 60)
    assert ath.row["xERA"] == pytest.approx(ath.siera + 0.1) and "SIERA 5.00" in ath.detail
    assert "- - -" not in mid and "PHI" in leverage
    # PHI's only arm is a leverage arm, so it has no middle relief read
    assert "PHI" not in mid


def _outs(selection: str, line: float, fair: float, day: str = "2026-09-15") -> LedgerEntry:
    return LedgerEntry(
        day, "SF @ STL", "prop", "pitcher_outs", selection, line, "dk", -110, "Pass", 0.5, None, "", 0.0,
        fair_prob=fair,
    )


def test_the_outs_prop_is_read_as_the_chance_the_starter_leaves_before_the_sixth() -> None:
    entries = [
        _outs("Logan Webb Outs u17.5", 17.5, 0.42),
        _outs("Logan Webb Outs o15.5", 15.5, 0.80),
        _outs("Sonny Gray Outs u15.5", 15.5, 0.55),
        _outs("Sonny Gray Outs u18.5", 18.5, 0.70),  # a rung at 6+ innings says nothing about a short start
        _outs("Sonny Gray Outs u17.5", 17.5, 0.60, day="2026-09-14"),
    ]
    outs = outs_under(entries, Date(2026, 9, 15))
    assert outs[("SF @ STL", "Logan Webb")] == pytest.approx(0.42)
    assert outs[("SF @ STL", "Sonny Gray")] == pytest.approx(0.55)


def test_a_short_start_is_the_market_first_then_the_season_line() -> None:
    webb = Pitcher(mlbam_id=1, name="Logan Webb")
    assert short_start(webb, {"IP": 100.0, "GS": 20.0}, 0.62).startswith("market 62%")
    assert short_start(webb, {"IP": 100.0, "GS": 20.0}, 0.45) == ""
    assert short_start(webb, {"IP": 100.0, "GS": 20.0}, None) == "5.0 IP/GS this season"
    assert short_start(webb, {"IP": 130.0, "Start-IP": 126.0, "GS": 20.0}, None) == ""
    assert short_start(webb, {"IP": 20.0, "GS": 0.0}, None) == "no starts this season"
    assert short_start(None, None, None) == "TBD starter"


def test_the_middle_relief_ranking_is_written_as_its_own_sheet(tmp_path: Path) -> None:
    rows = [_rel(f"L{i}", team, 40, 3.0, hld=10, pid=10 * i + k) for k, team in enumerate(("SFG", "PHI")) for i in (1, 2, 3)]
    rows += [_rel("A", "SFG", 30, 4.6, pid=101), _rel("B", "PHI", 30, 3.3, pid=102)]
    _, mid = middle_relief(rows)
    path = write_workbook([], Date(2026, 9, 15), tmp_path / "t.xlsx", mid)
    ws = load_workbook(path)["Middle relief"]
    ranked = [(r[1], r[2], r[10]) for r in ws.iter_rows(min_row=2, values_only=True)]
    assert ranked == [("PHI", -2, "B"), ("SFG", 2, "A")]


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
