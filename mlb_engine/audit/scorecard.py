"""Compute sensitivity / specificity / PPV / NPV per tier and persist a scorecard.

Framing: for a tier T, the positive *prediction* is "recommendation is in tier
T"; the positive *outcome* is "the bet won" (pushes excluded). This yields, per
tier:

    PPV = TP / (TP + FP)      # win rate of that tier's picks
    NPV = TN / (TN + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

A combined "Buy" row treats Strong+Moderate as the positive prediction.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier
from mlb_engine.recommendations import Recommendation


@dataclass
class TierMetrics:
    date: str
    tier: str
    n: int
    wins: int
    losses: int
    ppv: float
    npv: float
    sensitivity: float
    specificity: float
    roi: float


def _safe(n: float, d: float) -> float:
    return round(n / d, 4) if d else 0.0


def _roi(recs_grades: list[tuple[Recommendation, str]]) -> float:
    stake = 0.0
    profit = 0.0
    for rec, g in recs_grades:
        if g not in (WIN, LOSS):
            continue
        stake += 1.0
        if rec.market_american is not None:
            dec = american_to_decimal(rec.market_american)
        else:
            dec = 1.91  # assume -110 if no price captured
        profit += (dec - 1.0) if g == WIN else -1.0
    return _safe(profit, stake)


def _tier_metrics(
    graded: list[tuple[Recommendation, str]],
    positive: set[str],
    label: str,
    date: Date,
) -> TierMetrics:
    tp = fp = fn = tn = 0
    subset = []
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
    return TierMetrics(
        date=date.isoformat(),
        tier=label,
        n=tp + fp,
        wins=tp,
        losses=fp,
        ppv=_safe(tp, tp + fp),
        npv=_safe(tn, tn + fn),
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        roi=_roi(subset),
    )


def build_scorecard(graded: list[tuple[Recommendation, str]], date: Date) -> list[TierMetrics]:
    rows = [
        _tier_metrics(graded, {Tier.STRONG.value}, Tier.STRONG.value, date),
        _tier_metrics(graded, {Tier.MODERATE.value}, Tier.MODERATE.value, date),
        _tier_metrics(graded, {Tier.PASS.value}, Tier.PASS.value, date),
        _tier_metrics(graded, {Tier.STRONG.value, Tier.MODERATE.value}, "Buy (S+M)", date),
    ]
    return rows


FIELDS = [
    "date",
    "tier",
    "n",
    "wins",
    "losses",
    "ppv",
    "npv",
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
