"""PPV / NPV / sensitivity / specificity per tier *and per market*.

Framing (matches the MLB engine): for a tier T the positive *prediction* is
"recommendation is in tier T"; the positive *outcome* is "the bet won" (pushes
excluded). Per tier:

    PPV = TP / (TP + FP)      # realized win rate of that tier's picks
    NPV = TN / (TN + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

Rows are produced overall and split by market (Moneyline / ATS / Totals) so the
user can see realized PPV/NPV per market -- the concrete answer to "does the
marking layer raise PPV", measured on graded, out-of-sample recommendations.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

from cfb_engine.audit.grade import LOSS, WIN
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation

_MARKET_LABEL = {"game_ml": "ML", "game_ats": "ATS", "game_total": "Totals"}


@dataclass
class TierMetrics:
    date: str
    market: str  # "All" | "ML" | "ATS" | "Totals"
    tier: str
    n: int
    wins: int
    losses: int
    ppv: float
    npv: float
    # Break-even win rate the tier's *prices* charge, and PPV minus that. This
    # -- not raw PPV -- is the number that decides whether the tier makes money:
    # a 58% PPV at -150 (break-even 60%) is a losing bet, so edge_vs_be < 0.
    breakeven: float
    edge_vs_be: float
    sensitivity: float
    specificity: float
    roi: float


def _safe(n: float, d: float) -> float:
    return round(n / d, 4) if d else 0.0


def _breakeven(recs_grades: list[tuple[Recommendation, str]]) -> float:
    """Average implied (no-adjust) win rate the bet prices demand."""
    rates: list[float] = []
    for rec, g in recs_grades:
        if g not in (WIN, LOSS):
            continue
        dec = american_to_decimal(rec.market_american) if rec.market_american is not None else 1.91
        if dec > 0:
            rates.append(1.0 / dec)
    return round(sum(rates) / len(rates), 4) if rates else 0.0


def _roi(recs_grades: list[tuple[Recommendation, str]]) -> float:
    stake = profit = 0.0
    for rec, g in recs_grades:
        if g not in (WIN, LOSS):
            continue
        stake += 1.0
        dec = american_to_decimal(rec.market_american) if rec.market_american is not None else 1.91
        profit += (dec - 1.0) if g == WIN else -1.0
    return _safe(profit, stake)


def _tier_metrics(
    graded: list[tuple[Recommendation, str]],
    positive: set[str],
    label: str,
    market_label: str,
    date: Date,
) -> TierMetrics:
    tp = fp = fn = tn = 0
    subset: list[tuple[Recommendation, str]] = []
    for rec, g in graded:
        if g not in (WIN, LOSS):
            continue
        pred_pos = rec.tier.value in positive
        actual_pos = g == WIN
        if pred_pos and actual_pos:
            tp += 1
        elif pred_pos and not actual_pos:
            fp += 1
        elif not pred_pos and actual_pos:
            fn += 1
        else:
            tn += 1
        if pred_pos:
            subset.append((rec, g))
    ppv = _safe(tp, tp + fp)
    breakeven = _breakeven(subset)
    return TierMetrics(
        date=date.isoformat(),
        market=market_label,
        tier=label,
        n=tp + fp,
        wins=tp,
        losses=fp,
        ppv=ppv,
        npv=_safe(tn, tn + fn),
        breakeven=breakeven,
        edge_vs_be=round(ppv - breakeven, 4),
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        roi=_roi(subset),
    )


def _rows_for(
    graded: list[tuple[Recommendation, str]], market_label: str, date: Date
) -> list[TierMetrics]:
    return [
        _tier_metrics(graded, {Tier.STRONG.value}, Tier.STRONG.value, market_label, date),
        _tier_metrics(graded, {Tier.MODERATE.value}, Tier.MODERATE.value, market_label, date),
        _tier_metrics(graded, {Tier.PASS.value}, Tier.PASS.value, market_label, date),
        _tier_metrics(
            graded,
            {Tier.STRONG.value, Tier.MODERATE.value},
            "Buy (S+M)",
            market_label,
            date,
        ),
    ]


def build_scorecard(graded: list[tuple[Recommendation, str]], date: Date) -> list[TierMetrics]:
    rows = _rows_for(graded, "All", date)
    for market, label in _MARKET_LABEL.items():
        subset = [(r, g) for r, g in graded if r.market == market]
        if subset:
            rows.extend(_rows_for(subset, label, date))
    return rows


FIELDS = [
    "date",
    "market",
    "tier",
    "n",
    "wins",
    "losses",
    "ppv",
    "npv",
    "breakeven",
    "edge_vs_be",
    "sensitivity",
    "specificity",
    "roi",
]


def append_scorecard(rows: list[TierMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
