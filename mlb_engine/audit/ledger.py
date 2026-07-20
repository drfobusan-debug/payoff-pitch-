"""A persistent per-bet ledger plus cumulative engine performance metrics.

The ledger records every graded recommendation (one row per bet) across all
audited slates, and rolls the whole history up into overall sensitivity,
specificity, PPV, NPV, win rate, ROI and net units by tier.

Confusion-matrix framing (same as the daily scorecard): for a tier T the
positive *prediction* is "pick is in tier T" and the positive *outcome* is "the
bet won" (pushes excluded):

    PPV         = TP / (TP + FP)   # win rate of that tier's picks
    NPV         = TN / (TN + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation

_DEFAULT_DECIMAL = 1.91  # assume -110 when no price was captured


@dataclass
class LedgerEntry:
    date: str
    matchup: str
    category: str
    market: str
    selection: str
    line: float | None
    book: str
    odds: float | None  # american
    tier: str
    model_prob: float
    ev: float | None
    result: str  # win | loss | push
    pnl: float  # net units on a 1u stake (win: dec-1, loss: -1, push: 0)


LEDGER_FIELDS = [
    "date",
    "matchup",
    "category",
    "market",
    "selection",
    "line",
    "book",
    "odds",
    "tier",
    "model_prob",
    "ev",
    "result",
    "pnl",
]


def _pnl(result: str, odds: float | None) -> float:
    if result == WIN:
        dec = american_to_decimal(odds) if odds is not None else _DEFAULT_DECIMAL
        return round(dec - 1.0, 4)
    if result == LOSS:
        return -1.0
    return 0.0


def entries_from_graded(
    graded: list[tuple[Recommendation, str]], date: Date
) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    for rec, result in graded:
        entries.append(
            LedgerEntry(
                date=date.isoformat(),
                matchup=rec.matchup,
                category=rec.display_category,
                market=rec.market,
                selection=rec.selection,
                line=rec.line,
                book=rec.book or "",
                odds=rec.market_american,
                tier=rec.tier.value,
                model_prob=round(rec.model_prob, 4),
                ev=round(rec.ev, 4) if rec.ev is not None else None,
                result=result,
                pnl=_pnl(result, rec.market_american),
            )
        )
    return entries


def _to_float(v: str) -> float | None:
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_ledger(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        return []
    out: list[LedgerEntry] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                LedgerEntry(
                    date=row["date"],
                    matchup=row["matchup"],
                    category=row["category"],
                    market=row["market"],
                    selection=row["selection"],
                    line=_to_float(row["line"]),
                    book=row["book"],
                    odds=_to_float(row["odds"]),
                    tier=row["tier"],
                    model_prob=_to_float(row["model_prob"]) or 0.0,
                    ev=_to_float(row["ev"]),
                    result=row["result"],
                    pnl=_to_float(row["pnl"]) or 0.0,
                )
            )
    return out


def update_ledger(path: Path, new_entries: list[LedgerEntry], date: Date) -> list[LedgerEntry]:
    """Replace any rows for ``date`` with ``new_entries`` (re-audit safe), persist, return all."""
    iso = date.isoformat()
    kept = [e for e in load_ledger(path) if e.date != iso]
    merged = kept + new_entries
    merged.sort(key=lambda e: (e.date, e.category, e.matchup))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        w.writeheader()
        for e in merged:
            row = asdict(e)
            row["line"] = "" if e.line is None else e.line
            row["odds"] = "" if e.odds is None else e.odds
            row["ev"] = "" if e.ev is None else e.ev
            w.writerow(row)
    return merged


def _safe(n: float, d: float) -> float:
    return round(n / d, 4) if d else 0.0


@dataclass
class OverallMetrics:
    tier: str
    n: int
    wins: int
    losses: int
    pushes: int
    win_pct: float
    ppv: float
    npv: float
    sensitivity: float
    specificity: float
    roi: float
    units: float


def _metrics_for(entries: list[LedgerEntry], positive: set[str], label: str) -> OverallMetrics:
    tp = fp = fn = tn = pushes = 0
    stake = 0.0
    units = 0.0
    for e in entries:
        pred_pos = e.tier in positive
        if e.result == PUSH:
            if pred_pos:
                pushes += 1
            continue
        actual_pos = e.result == WIN
        if pred_pos and actual_pos:
            tp += 1
        elif pred_pos and not actual_pos:
            fp += 1
        elif not pred_pos and actual_pos:
            fn += 1
        else:
            tn += 1
        if pred_pos:
            stake += 1.0
            units += e.pnl
    return OverallMetrics(
        tier=label,
        n=tp + fp,
        wins=tp,
        losses=fp,
        pushes=pushes,
        win_pct=_safe(tp, tp + fp),
        ppv=_safe(tp, tp + fp),
        npv=_safe(tn, tn + fn),
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        roi=_safe(units, stake),
        units=round(units, 3),
    )


def overall_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    return [
        _metrics_for(entries, {Tier.STRONG.value}, Tier.STRONG.value),
        _metrics_for(entries, {Tier.MODERATE.value}, Tier.MODERATE.value),
        _metrics_for(entries, {Tier.PASS.value}, Tier.PASS.value),
        _metrics_for(entries, {Tier.STRONG.value, Tier.MODERATE.value}, "Buy (S+M)"),
    ]


def daily_rollup(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """One Buy (S+M) metrics row per audited date, oldest first."""
    buy = {Tier.STRONG.value, Tier.MODERATE.value}
    by_date: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        by_date.setdefault(e.date, []).append(e)
    return [_metrics_for(by_date[d], buy, d) for d in sorted(by_date)]
