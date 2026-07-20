"""Write daily recommendations to a formatted Excel workbook.

Sheets:
  * Buys      : Strong/Moderate buys only, sorted by tier then EV.
  * All       : every priced market.
  * By game   : one section per game.
The workbook is styled with tier color-coding and auto-filters.
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

HEADER_FILL = PatternFill("solid", fgColor="1F2937")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TIER_FILL = {
    Tier.STRONG.value: PatternFill("solid", fgColor="C6EFCE"),
    Tier.MODERATE.value: PatternFill("solid", fgColor="FFEB9C"),
    Tier.PASS.value: PatternFill("solid", fgColor="F2F2F2"),
}

COLUMNS = [
    "Date",
    "Matchup",
    "Category",
    "Market",
    "Selection",
    "Line",
    "Model %",
    "Fair Odds",
    "Book",
    "Book Odds",
    "EV",
    "Edge",
    "Handle %",
    "Bets %",
    "Tier",
    "Notes",
]

TIER_ORDER = {Tier.STRONG.value: 0, Tier.MODERATE.value: 1, Tier.PASS.value: 2}


def _write_sheet(ws: Worksheet, recs: list[Recommendation]) -> None:
    for c, name in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=c, value=name)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")
    for r, rec in enumerate(recs, start=2):
        row = rec.as_row()
        for c, name in enumerate(COLUMNS, start=1):
            ws.cell(row=r, column=c, value=row[name])
        tier_fill = TIER_FILL.get(rec.tier.value)
        if tier_fill:
            ws.cell(row=r, column=COLUMNS.index("Tier") + 1).fill = tier_fill
    # widths + filter
    widths = [11, 12, 9, 14, 26, 6, 8, 9, 11, 10, 8, 8, 9, 8, 12, 40]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if recs:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(recs) + 1}"
    ws.freeze_panes = "A2"


def write_workbook(recs: list[Recommendation], out_path: Path, slate_date: Date) -> Path:
    def sort_key(r: Recommendation):
        return (TIER_ORDER.get(r.tier.value, 3), -(r.ev if r.ev is not None else -99))

    wb = Workbook()
    buys = sorted(
        [r for r in recs if r.tier in (Tier.STRONG, Tier.MODERATE)], key=sort_key
    )
    ws_buys = wb.active
    ws_buys.title = "Buys"
    _write_sheet(ws_buys, buys)

    ws_all = wb.create_sheet("All")
    _write_sheet(ws_all, sorted(recs, key=sort_key))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path
