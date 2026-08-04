"""Classify EV results into Strong buy / Moderate buy / Pass tiers.

Base tier is set by EV thresholds, then adjusted by the model edge over the
no-vig market price (guards against thin edges).
"""

from __future__ import annotations

from enum import Enum

from cfb_engine.config import EVThresholds
from cfb_engine.market.ev import EVResult


class Tier(str, Enum):
    STRONG = "Strong buy"
    MODERATE = "Moderate buy"
    PASS = "Pass"


def _base_tier(ev: float, thr: EVThresholds) -> Tier:
    if ev >= thr.strong_buy:
        return Tier.STRONG
    if ev >= thr.moderate_buy:
        return Tier.MODERATE
    return Tier.PASS


def bump_tier(tier: Tier, steps: int) -> Tier:
    order = [Tier.PASS, Tier.MODERATE, Tier.STRONG]
    i = max(0, min(len(order) - 1, order.index(tier) + steps))
    return order[i]


def classify(result: EVResult, thr: EVThresholds) -> tuple[Tier, list[str]]:
    reasons: list[str] = []
    tier = _base_tier(result.ev, thr)
    reasons.append(f"EV={result.ev:+.3f}")

    # Thin-edge guard: never buy without a real edge over the no-vig line.
    if tier != Tier.PASS and result.edge < thr.min_edge:
        tier = Tier.PASS
        reasons.append(f"edge {result.edge:+.3f} < {thr.min_edge} -> pass")
        return tier, reasons

    # Strict selection: keep only Strong buys.
    if thr.strong_only and tier == Tier.MODERATE:
        tier = Tier.PASS
        reasons.append("strong-only mode -> pass")

    return tier, reasons
