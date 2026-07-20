"""Classify EV results into Strong buy / Moderate buy / Pass tiers.

Base tier is set by EV thresholds, then adjusted by:
  * model edge over the no-vig market price (guards against thin edges), and
  * VSIN handle/bets divergence (sharp money agreement/disagreement).
"""

from __future__ import annotations

from enum import Enum

from mlb_engine.config import EVThresholds
from mlb_engine.market.ev import EVResult

MIN_EDGE = 0.02  # require at least 2% edge over no-vig market to buy
SHARP_DIVERGENCE_STRONG = 15.0  # handle-bets gap considered a strong sharp signal


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


def _bump(tier: Tier, steps: int) -> Tier:
    return bump_tier(tier, steps)


def classify(result: EVResult, thr: EVThresholds) -> tuple[Tier, list[str]]:
    reasons: list[str] = []
    tier = _base_tier(result.ev, thr)
    reasons.append(f"EV={result.ev:+.3f}")

    # Thin-edge guard: never buy without a real edge over the no-vig line.
    if tier != Tier.PASS and result.edge < MIN_EDGE:
        tier = Tier.PASS
        reasons.append(f"edge {result.edge:+.3f} < {MIN_EDGE} -> pass")
        return tier, reasons

    div = result.sharp_divergence
    if div is not None:
        reasons.append(f"handle-bets={div:+.0f}")
        if div <= -SHARP_DIVERGENCE_STRONG and tier != Tier.PASS:
            tier = _bump(tier, -1)
            reasons.append("sharp money against -> downgrade")
        elif div >= SHARP_DIVERGENCE_STRONG and tier == Tier.MODERATE:
            tier = _bump(tier, +1)
            reasons.append("sharp money with -> upgrade")

    return tier, reasons
