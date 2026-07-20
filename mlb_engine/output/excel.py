"""Write daily recommendations to a formatted Excel workbook.

Sheets:
  * Strong Buys : strong-tier picks, categorized, green spectrum (dark = BEST).
  * Moderate Buys : moderate-tier picks, categorized, yellow spectrum.
  * All        : every priced market, tier color-coded.

Within the tier sheets picks are grouped by market category (Moneyline, Totals,
Run Lines, First-5, Batter Props, Pitcher Props, Comeback) and shaded on a
light->dark spectrum by EV, with a ``BEST`` flag highlighting the
high-conviction plays.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
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

# --- categorized tier-sheet layout -----------------------------------------

CATEGORY_ORDER = [
    "Moneyline",
    "Totals",
    "Run Lines",
    "First-5 (F5)",
    "Batter Props",
    "Pitcher Props",
    "Comeback (info)",
]
_GAME_CATEGORIES = {"Moneyline", "Totals", "Run Lines", "First-5 (F5)"}
_PROP_CATEGORIES = {"Batter Props", "Pitcher Props"}

TIER_COLUMNS = [
    "Best",
    "Category",
    "EV",
    "Selection",
    "Matchup",
    "Book",
    "Odds",
    "Model %",
    "Edge",
    "Handle %",
    "Bets %",
    "Notes",
]
TIER_COLUMN_WIDTHS = [8, 15, 8, 30, 13, 13, 8, 8, 8, 9, 8, 40]

# Green spectrum (strong) / yellow spectrum (moderate).
@dataclass(frozen=True)
class _Scheme:
    best: str  # BEST-row fill
    best_font: str  # BEST-row font color
    light: tuple[int, int, int]  # low-EV end of the gradient
    dark: tuple[int, int, int]  # high-EV end of the gradient
    header: str
    border: str


_SPECTRUM: dict[str, _Scheme] = {
    Tier.STRONG.value: _Scheme(
        best="1B5E20",
        best_font="FFFFFF",
        light=(0xE8, 0xF5, 0xE9),
        dark=(0x66, 0xBB, 0x6A),
        header="0D3311",
        border="C8E6C9",
    ),
    Tier.MODERATE.value: _Scheme(
        best="F9A825",
        best_font="000000",
        light=(0xFF, 0xFD, 0xE7),
        dark=(0xFF, 0xEB, 0x3B),
        header="7A5B00",
        border="FFF59D",
    ),
}


def _display_category(rec: Recommendation) -> str:
    m = rec.market
    if m == "game_ml":
        return "Moneyline"
    if m == "game_total":
        return "Totals"
    if m == "game_rl":
        return "Run Lines"
    if m.startswith("f5"):
        return "First-5 (F5)"
    if m.startswith("batter_"):
        return "Batter Props"
    if m.startswith("pitcher_"):
        return "Pitcher Props"
    if m == "comeback":
        return "Comeback (info)"
    return m


def _is_best(rec: Recommendation, category: str) -> bool:
    ev = rec.ev
    odds = rec.market_american
    if ev is None or odds is None:
        return False
    if category in _GAME_CATEGORIES:
        return ev >= 0.15
    if category in _PROP_CATEGORIES:
        return ev >= 0.20 and -200 <= odds <= 120
    return False


def _american(odds: float | None) -> str:
    if odds is None:
        return "n/a"
    o = round(odds)
    return f"+{o}" if o > 0 else str(o)


def _interp(light: tuple[int, int, int], dark: tuple[int, int, int], t: float) -> str:
    rgb = tuple(int(light[i] + (dark[i] - light[i]) * t) for i in range(3))
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


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
    widths = [11, 12, 9, 14, 26, 6, 8, 9, 11, 10, 8, 8, 9, 8, 12, 40]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if recs:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(recs) + 1}"
    ws.freeze_panes = "A2"


def _write_tier_sheet(ws: Worksheet, recs: list[Recommendation], tier: Tier) -> None:
    """Categorized, EV-shaded sheet for one tier (green=strong, yellow=moderate)."""
    scheme = _SPECTRUM[tier.value]
    header_fill = PatternFill("solid", fgColor=scheme.header)
    best_fill = PatternFill("solid", fgColor=scheme.best)
    best_font = Font(color=scheme.best_font, bold=True)
    side = Side(style="thin", color=scheme.border)
    border = Border(left=side, right=side, top=side, bottom=side)
    center = Alignment(horizontal="center")

    tagged = [(r, _display_category(r)) for r in recs]
    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    tagged.sort(
        key=lambda t: (
            cat_rank.get(t[1], len(CATEGORY_ORDER)),
            0 if _is_best(t[0], t[1]) else 1,
            -(t[0].ev if t[0].ev is not None else -99.0),
        )
    )

    # EV range per category for shading.
    ev_range: dict[str, tuple[float, float]] = {}
    for rec, cat in tagged:
        ev = rec.ev if rec.ev is not None else 0.0
        lo, hi = ev_range.get(cat, (ev, ev))
        ev_range[cat] = (min(lo, ev), max(hi, ev))

    for c, name in enumerate(TIER_COLUMNS, start=1):
        cell = ws.cell(row=1, column=c, value=name)
        cell.fill = header_fill
        cell.font = HEADER_FONT
        cell.alignment = center
        cell.border = border

    centered_cols = {"Best", "EV", "Odds", "Model %", "Edge", "Handle %", "Bets %"}
    for r, (rec, cat) in enumerate(tagged, start=2):
        best = _is_best(rec, cat)
        ev = rec.ev if rec.ev is not None else 0.0
        lo, hi = ev_range[cat]
        t = 0.5 if hi == lo else (ev - lo) / (hi - lo)
        fill = best_fill if best else PatternFill(
            "solid", fgColor=_interp(scheme.light, scheme.dark, 0.15 + 0.7 * t)
        )
        values = {
            "Best": "BEST" if best else "",
            "Category": cat,
            "EV": round(rec.ev, 3) if rec.ev is not None else "",
            "Selection": rec.selection,
            "Matchup": rec.matchup,
            "Book": rec.book or "",
            "Odds": _american(rec.market_american),
            "Model %": round(rec.model_prob * 100, 1),
            "Edge": round(rec.edge, 3) if rec.edge is not None else "",
            "Handle %": rec.handle_pct if rec.handle_pct is not None else "",
            "Bets %": rec.bets_pct if rec.bets_pct is not None else "",
            "Notes": "; ".join(rec.reasons),
        }
        for c, name in enumerate(TIER_COLUMNS, start=1):
            cell = ws.cell(row=r, column=c, value=values[name])
            cell.fill = fill
            cell.border = border
            if best:
                cell.font = best_font
            if name in centered_cols:
                cell.alignment = center

    for i, w in enumerate(TIER_COLUMN_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if tagged:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(TIER_COLUMNS))}{len(tagged) + 1}"
    ws.freeze_panes = "A2"


def write_workbook(recs: list[Recommendation], out_path: Path, slate_date: Date) -> Path:
    def sort_key(r: Recommendation) -> tuple[int, float]:
        return (TIER_ORDER.get(r.tier.value, 3), -(r.ev if r.ev is not None else -99))

    wb = Workbook()
    strong = [r for r in recs if r.tier == Tier.STRONG]
    moderate = [r for r in recs if r.tier == Tier.MODERATE]

    ws_strong = wb.active
    ws_strong.title = "Strong Buys"
    _write_tier_sheet(ws_strong, strong, Tier.STRONG)

    ws_moderate = wb.create_sheet("Moderate Buys")
    _write_tier_sheet(ws_moderate, moderate, Tier.MODERATE)

    ws_all = wb.create_sheet("All")
    _write_sheet(ws_all, sorted(recs, key=sort_key))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path
