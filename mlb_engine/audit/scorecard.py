"""Compute sensitivity / specificity / PPV / NPV per tier and persist a scorecard.

Framing: for a tier T, the positive *prediction* is "recommendation is in tier
T"; the positive *outcome* is "the bet won" (pushes excluded). This yields, per
tier:

    PPV = TP / (TP + FP)      # win rate of that tier's picks
    NPV = TN / (TN + FN)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)

A combined "Buy" row treats Strong+Moderate as the positive prediction.

PPV and NPV are both reported alongside the rate they would show with no skill
at all, because on their own they are easy to misread in opposite directions.
NPV especially: when a prop only wins 15% of the time, declining to bet it scores
85% NPV for free, and a tier that passes on everything scores a perfect one. The
lift columns are the part that reflects the engine rather than the base rate.
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
    base_win: float
    base_loss: float
    ppv_lift: float
    npv_lift: float
    sensitivity: float
    specificity: float
    #: ROI over every graded pick in the tier, paying the ones with no captured
    #: price at an assumed -110.
    roi: float
    #: The same return over the picks that carried a real price, and how many
    #: there were. The assumed rows win more often than the priced ones -- a prop
    #: the board never posted a beatable number on is one the model finds easy --
    #: so the blended figure reads high and only this one is a return.
    priced_n: int = 0
    priced_roi: float = 0.0


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


def _priced(recs_grades: list[tuple[Recommendation, str]]) -> tuple[int, float]:
    """(picks that carried a real price, their ROI)."""
    priced = [(r, g) for r, g in recs_grades if r.market_american is not None]
    return len([1 for _, g in priced if g in (WIN, LOSS)]), _roi(priced)


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
    graded_n = tp + fp + fn + tn
    priced_n, priced_roi = _priced(subset)
    base_win = _safe(tp + fn, graded_n)
    base_loss = _safe(fp + tn, graded_n)
    return TierMetrics(
        date=date.isoformat(),
        tier=label,
        n=tp + fp,
        wins=tp,
        losses=fp,
        ppv=_safe(tp, tp + fp),
        npv=_safe(tn, tn + fn),
        base_win=base_win,
        base_loss=base_loss,
        ppv_lift=round(_safe(tp, tp + fp) - base_win, 4) if tp + fp else 0.0,
        npv_lift=round(_safe(tn, tn + fn) - base_loss, 4) if tn + fn else 0.0,
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        roi=_roi(subset),
        priced_n=priced_n,
        priced_roi=priced_roi,
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
    "base_win",
    "base_loss",
    "ppv_lift",
    "npv_lift",
    "sensitivity",
    "specificity",
    "roi",
    "priced_n",
    "priced_roi",
]


def _migrate(path: Path) -> None:
    """Widen an existing scorecard to the current columns.

    Appending new fields to a CSV written under an older header silently shifts
    every value in the new rows one column left of where the reader expects it.
    Rewrite the file with the full header instead, leaving the added columns
    blank on historical rows -- they cannot be recomputed, since the scorecard
    never stored the true-negative counts the lift is derived from.
    """
    with path.open(newline="") as f:
        prior = list(csv.DictReader(f))
    if not prior or set(prior[0]) >= set(FIELDS):
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in prior:
            w.writerow({k: row.get(k, "") for k in FIELDS})


def append_scorecard(rows: list[TierMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        _migrate(path)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
