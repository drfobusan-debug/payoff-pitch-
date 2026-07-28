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

from mlb_engine.audit.analysis import (
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    TRUE_POSITIVE,
    PropInsight,
)
from mlb_engine.audit.clv import ClvSummary
from mlb_engine.audit.ledger import LedgerEntry, OverallMetrics
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
    "Market %",
    "Fair Odds",
    "Book",
    "Book Odds",
    "EV",
    "Edge",
    "Handle %",
    "Bets %",
    "Tier",
    "Signal",
    "Factor",
    "Score",
    "Profile",
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
    "Signal",
    "Factor",
    "Score",
    "Profile",
    "Notes",
]
TIER_COLUMN_WIDTHS = [8, 15, 8, 30, 13, 13, 8, 8, 8, 9, 8, 8, 7, 7, 30, 40]

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
    widths = [11, 12, 9, 14, 26, 6, 8, 9, 9, 11, 10, 8, 8, 9, 8, 12, 8, 7, 7, 30, 40]
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

    tagged = [(r, r.display_category) for r in recs]
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
            "Signal": rec.signal or "",
            "Factor": round(rec.factor, 3) if rec.factor is not None else "",
            "Score": round(rec.score, 2) if rec.score is not None else "",
            "Profile": rec.profile or "",
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


_METRIC_HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
_RESULT_FILL = {
    "win": PatternFill("solid", fgColor="C6EFCE"),
    "loss": PatternFill("solid", fgColor="FFC7CE"),
    "push": PatternFill("solid", fgColor="E7E6E6"),
}
_OVERALL_COLUMNS = [
    "Tier",
    "N",
    "Wins",
    "Losses",
    "Pushes",
    "Win %",
    "PPV",
    "NPV",
    "Sensitivity",
    "Specificity",
    "Needs %",
    "ROI",
    "Units",
]
_DAILY_COLUMNS = [
    "Date",
    "Buy N",
    "Wins",
    "Losses",
    "Win %",
    "ROI",
    "Units",
]
_BET_COLUMNS = [
    "Date",
    "Category",
    "Selection",
    "Matchup",
    "Book",
    "Odds",
    "Tier",
    "Model %",
    "Market %",
    "EV",
    "Close",
    "CLV",
    "CLV EV",
    "Result",
    "P/L",
]


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _metric_row(m: OverallMetrics) -> list[object]:
    return [
        m.tier,
        m.n,
        m.wins,
        m.losses,
        m.pushes,
        _pct(m.win_pct),
        m.ppv,
        m.npv,
        m.sensitivity,
        m.specificity,
        _pct(m.required_win_pct),
        _pct(m.roi),
        m.units,
    ]


_INSIGHT_COLUMNS = ["Type", "Market", "N", "Rate", "Recommendation"]
_INSIGHT_FILL = {
    FALSE_POSITIVE: PatternFill("solid", fgColor="FFC7CE"),
    FALSE_NEGATIVE: PatternFill("solid", fgColor="FFEB9C"),
    TRUE_POSITIVE: PatternFill("solid", fgColor="C6EFCE"),
}
_INSIGHT_LABEL = {
    FALSE_POSITIVE: "False positive",
    FALSE_NEGATIVE: "False negative",
    TRUE_POSITIVE: "True positive",
}


def _write_metric_header(ws: Worksheet, columns: list[str]) -> None:
    for c, name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=c, value=name)
        cell.fill = _METRIC_HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center")


def _write_metric_sheet(
    ws: Worksheet, rows: list[OverallMetrics], label_header: str = "Tier"
) -> None:
    cols = [label_header, *_OVERALL_COLUMNS[1:]]
    _write_metric_header(ws, cols)
    for r, m in enumerate(rows, start=2):
        for c, v in enumerate(_metric_row(m), start=1):
            ws.cell(row=r, column=c, value=v)
    for i, w in enumerate([14, 6, 6, 7, 7, 8, 7, 7, 12, 12, 9, 8, 8], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(rows) + 1}"
    ws.freeze_panes = "A2"


def _write_insight_sheet(ws: Worksheet, insights: list[PropInsight]) -> None:
    _write_metric_header(ws, _INSIGHT_COLUMNS)
    order = {FALSE_POSITIVE: 0, FALSE_NEGATIVE: 1, TRUE_POSITIVE: 2}
    ordered = sorted(insights, key=lambda i: (order.get(i.kind, 9), i.market))
    for r, ins in enumerate(ordered, start=2):
        vals = [
            _INSIGHT_LABEL.get(ins.kind, ins.kind),
            ins.market,
            ins.n,
            _pct(ins.rate),
            ins.finding,
        ]
        for c, v in enumerate(vals, start=1):
            ws.cell(row=r, column=c, value=v)
        fill = _INSIGHT_FILL.get(ins.kind)
        if fill:
            ws.cell(row=r, column=1).fill = fill
    for i, w in enumerate([15, 14, 6, 8, 100], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    if ordered:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(_INSIGHT_COLUMNS))}{len(ordered) + 1}"
    ws.freeze_panes = "A2"


_CLV_COLUMNS = ["Market", "N", "Mean CLV", "Beat close %", "Mean CLV EV"]
_CLV_FILL = {
    True: PatternFill("solid", fgColor="C6EFCE"),
    False: PatternFill("solid", fgColor="FFC7CE"),
}


def _write_clv_sheet(ws: Worksheet, rows: list[ClvSummary]) -> None:
    _write_metric_header(ws, _CLV_COLUMNS)
    for r, m in enumerate(rows, start=2):
        vals: list[object] = [
            m.label,
            m.n,
            f"{m.mean_clv * 100:+.2f}",
            _pct(m.beat_close_pct),
            f"{m.mean_clv_ev:+.4f}",
        ]
        for c, v in enumerate(vals, start=1):
            ws.cell(row=r, column=c, value=v)
        ws.cell(row=r, column=5).fill = _CLV_FILL[m.positive]
    for i, w in enumerate([16, 7, 11, 13, 12], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"


def write_ledger_workbook(
    entries: list[LedgerEntry],
    overall: list[OverallMetrics],
    daily: list[OverallMetrics],
    out_path: Path,
    daily_engine: list[OverallMetrics] | None = None,
    prop_rows: list[OverallMetrics] | None = None,
    insights: list[PropInsight] | None = None,
    runline_rows: list[OverallMetrics] | None = None,
    clv_rows: list[ClvSummary] | None = None,
) -> Path:
    wb = Workbook()

    ws_overall = wb.active
    ws_overall.title = "Overall"
    _write_metric_sheet(ws_overall, overall)

    if daily_engine:
        _write_metric_sheet(wb.create_sheet("Daily PPV-NPV"), daily_engine, "Date")

    if prop_rows:
        _write_metric_sheet(wb.create_sheet("Prop PPV-NPV"), prop_rows, "Prop market")

    if runline_rows:
        _write_metric_sheet(wb.create_sheet("Run Line NPV"), runline_rows, "Run line / gate")

    if clv_rows:
        _write_clv_sheet(wb.create_sheet("Closing Line Value"), clv_rows)

    if insights:
        _write_insight_sheet(wb.create_sheet("Prop Insights"), insights)

    ws_daily = wb.create_sheet("Daily")
    _write_metric_header(ws_daily, _DAILY_COLUMNS)
    for r, m in enumerate(daily, start=2):
        vals = [m.tier, m.n, m.wins, m.losses, _pct(m.win_pct), _pct(m.roi), m.units]
        for c, v in enumerate(vals, start=1):
            ws_daily.cell(row=r, column=c, value=v)
    for i, w in enumerate([12, 8, 7, 8, 8, 8, 8], start=1):
        ws_daily.column_dimensions[get_column_letter(i)].width = w
    if daily:
        ws_daily.auto_filter.ref = f"A1:{get_column_letter(len(_DAILY_COLUMNS))}{len(daily) + 1}"
    ws_daily.freeze_panes = "A2"

    ws_bets = wb.create_sheet("Bets")
    _write_metric_header(ws_bets, _BET_COLUMNS)
    for r, e in enumerate(entries, start=2):
        vals = [
            e.date,
            e.category,
            e.selection,
            e.matchup,
            e.book,
            _american(e.odds),
            e.tier,
            round(e.model_prob * 100, 1),
            round(e.fair_prob * 100, 1) if e.fair_prob is not None else "",
            e.ev if e.ev is not None else "",
            _american(e.close_odds),
            round(e.clv * 100, 2) if e.clv is not None else "",
            round(e.clv_ev, 3) if e.clv_ev is not None else "",
            e.result,
            round(e.pnl, 3),
        ]
        for c, v in enumerate(vals, start=1):
            ws_bets.cell(row=r, column=c, value=v)
        fill = _RESULT_FILL.get(e.result)
        if fill:
            ws_bets.cell(row=r, column=_BET_COLUMNS.index("Result") + 1).fill = fill
    for i, w in enumerate([12, 15, 30, 13, 13, 8, 10, 8, 9, 8, 8, 8, 8, 8, 8], start=1):
        ws_bets.column_dimensions[get_column_letter(i)].width = w
    if entries:
        ws_bets.auto_filter.ref = f"A1:{get_column_letter(len(_BET_COLUMNS))}{len(entries) + 1}"
    ws_bets.freeze_panes = "A2"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return out_path
