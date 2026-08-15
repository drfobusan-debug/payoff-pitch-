"""A persistent per-bet ledger plus cumulative performance metrics.

Every graded recommendation becomes one row, appended across all audited
slates, and the whole history rolls up into win rate, PPV, NPV, sensitivity,
specificity, ROI and net units -- by tier, by market, and for the whole engine.

Confusion-matrix framing: for a tier T the positive *prediction* is "pick is in
tier T" and the positive *outcome* is "the bet won" (pushes excluded):

    PPV = TP / (TP + FP)   # win rate of that tier's picks
    NPV = TN / (TN + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

from cfb_engine.audit.grade import LOSS, PUSH, WIN
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation

_DEFAULT_DECIMAL = 1.91  # assume -110 when no price was captured


@dataclass
class LedgerEntry:
    date: str
    matchup: str
    category: str  # display category (Moneyline / Spread (ATS) / Totals)
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
    raw_prob: float | None = None
    fair_prob: float | None = None
    bet_prob: float | None = None
    under_odds: float | None = None  # opposing side price at bet time
    # Closing line value (populated by ``cfb-engine close`` + audit).
    close_odds: float | None = None
    close_prob: float | None = None
    clv: float | None = None
    clv_ev: float | None = None


LEDGER_FIELDS = [
    "date", "matchup", "category", "market", "selection", "line", "book",
    "odds", "under_odds", "tier", "model_prob", "ev", "result", "pnl",
    "raw_prob", "fair_prob", "bet_prob",
    "close_odds", "close_prob", "clv", "clv_ev",
]
_OPTIONAL_FLOAT_FIELDS = (
    "line", "odds", "under_odds", "ev", "fair_prob", "bet_prob",
    "close_odds", "close_prob", "clv", "clv_ev",
)


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
                under_odds=rec.opposite_american,
                tier=rec.tier.value,
                model_prob=round(rec.model_prob, 4),
                ev=round(rec.ev, 4) if rec.ev is not None else None,
                result=result,
                pnl=_pnl(result, rec.market_american),
                raw_prob=round(rec.raw_prob, 4) if rec.raw_prob is not None else None,
                fair_prob=round(rec.fair_prob, 4) if rec.fair_prob is not None else None,
                bet_prob=round(rec.bet_prob, 4) if rec.bet_prob is not None else None,
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
                    under_odds=_to_float(row.get("under_odds", "") or ""),
                    tier=row["tier"],
                    model_prob=_to_float(row["model_prob"]) or 0.0,
                    ev=_to_float(row["ev"]),
                    result=row["result"],
                    pnl=_to_float(row["pnl"]) or 0.0,
                    raw_prob=_to_float(row.get("raw_prob", "")),
                    fair_prob=_to_float(row.get("fair_prob", "")),
                    bet_prob=_to_float(row.get("bet_prob", "")),
                    close_odds=_to_float(row.get("close_odds", "")),
                    close_prob=_to_float(row.get("close_prob", "")),
                    clv=_to_float(row.get("clv", "")),
                    clv_ev=_to_float(row.get("clv_ev", "")),
                )
            )
    return out


def update_ledger(path: Path, new_entries: list[LedgerEntry], date: Date) -> list[LedgerEntry]:
    """Replace any rows for ``date`` with ``new_entries`` (re-audit safe)."""
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
            for name in _OPTIONAL_FLOAT_FIELDS:
                if row[name] is None:
                    row[name] = ""
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
    required_win_pct: float = 0.0


def _metrics(
    entries: list[LedgerEntry], is_positive: Callable[[LedgerEntry], bool], label: str
) -> OverallMetrics:
    tp = fp = fn = tn = pushes = 0
    stake = units = breakeven = 0.0
    for e in entries:
        pred_pos = is_positive(e)
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
            dec = american_to_decimal(e.odds) if e.odds is not None else _DEFAULT_DECIMAL
            breakeven += 1.0 / dec
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
        required_win_pct=_safe(breakeven, stake),
    )


def _metrics_for(entries: list[LedgerEntry], positive: set[str], label: str) -> OverallMetrics:
    return _metrics(entries, lambda e: e.tier in positive, label)


def overall_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    return [
        _metrics_for(entries, {Tier.STRONG.value}, Tier.STRONG.value),
        _metrics_for(entries, {Tier.MODERATE.value}, Tier.MODERATE.value),
        _metrics_for(entries, {Tier.PASS.value}, Tier.PASS.value),
        _metrics_for(entries, {Tier.STRONG.value, Tier.MODERATE.value}, "Buy (S+M)"),
    ]


ENGINE_PROB_THRESHOLD = 0.5
ENGINE_LABEL = "ENGINE (p>=.5)"


def _favors(e: LedgerEntry) -> bool:
    return e.model_prob >= ENGINE_PROB_THRESHOLD


def engine_metrics(entries: list[LedgerEntry]) -> OverallMetrics:
    """Whole-engine directional discrimination across every graded market."""
    return _metrics(entries, _favors, ENGINE_LABEL)


def _by(entries: list[LedgerEntry], key: Callable[[LedgerEntry], str]) -> dict[str, list[LedgerEntry]]:
    out: dict[str, list[LedgerEntry]] = {}
    for e in entries:
        out.setdefault(key(e), []).append(e)
    return out


def daily_rollup(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """One Buy (S+M) metrics row per audited date, oldest first."""
    buy = {Tier.STRONG.value, Tier.MODERATE.value}
    by_date = _by(entries, lambda e: e.date)
    return [_metrics_for(by_date[d], buy, d) for d in sorted(by_date)]


# A dog is *supposed* to win less than half the time, so a sub-50% win rate is
# not evidence of anything by itself: what matters is the win rate against the
# break-even the price demands. Banding the buys by price length reports that gap
# -- and it is the axis where the MLB leak turned out to live (short dogs came in
# 9.7 points under the rate their prices charged while the favorites were fine).
PRICE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Heavy favorite (-200 and shorter)", -1e9, -200.0),
    ("Favorite (-199 to -110)", -199.0, -110.0),
    ("Pick'em (-109 to +109)", -109.0, 109.0),
    ("Short dog (+110 to +199)", 110.0, 199.0),
    ("Mid dog (+200 to +399)", 200.0, 399.0),
    ("Longshot (+400 and up)", 400.0, 1e9),
)


def price_bucket_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """Buy (S+M) performance per price band, plus the dog/favorite split.

    Only rows carrying a real price are counted: a bucket keyed on ``odds`` is
    meaningless for a bet whose price was never captured, and the -110 stand-in
    would pile every such row into Pick'em.
    """
    buy = {Tier.STRONG.value, Tier.MODERATE.value}
    priced = [e for e in entries if e.odds is not None and e.tier in buy]
    rows: list[OverallMetrics] = []
    for label, lo, hi in PRICE_BUCKETS:
        band = [e for e in priced if e.odds is not None and lo <= e.odds <= hi]
        if band:
            rows.append(_metrics_for(band, buy, label))
    for label, plus in (("All underdogs", True), ("All favorites", False)):
        side = [e for e in priced if e.odds is not None and (e.odds > 0) == plus]
        if side:
            rows.append(_metrics_for(side, buy, label))
    return rows


def market_metrics(entries: list[LedgerEntry]) -> list[OverallMetrics]:
    """Buy (S+M) PPV/ROI per market family, sorted by ROI (high to low)."""
    buy = {Tier.STRONG.value, Tier.MODERATE.value}
    by_market = _by(entries, lambda e: e.category)
    rows = [_metrics_for(by_market[m], buy, m) for m in by_market]
    rows.sort(key=lambda m: (m.n == 0, -m.roi))
    return rows
