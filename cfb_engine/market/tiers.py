"""Classify EV results into Strong buy / Moderate buy / Pass tiers.

A bet has to clear the EV floor at the price we can actually get, and is then
ranked on its edge over the no-vig market -- not on EV, because ``EV =
decimal_odds x edge`` makes an EV cutoff a cheaper bar the longer the price.
Edge also has a ceiling: past ``max_edge`` a disagreement reads as a model error,
which matters more here than in baseball because the closing spread is a strong
forecaster (r=+.647 against the final margin over 8,009 games) and the engine's
own efficiency layer adds nothing to it.
"""

from __future__ import annotations

from enum import Enum

from cfb_engine.config import EVThresholds
from cfb_engine.market.ev import EVResult


class Tier(str, Enum):
    STRONG = "Strong buy"
    MODERATE = "Moderate buy"
    PASS = "Pass"


def _base_tier(edge: float, thr: EVThresholds) -> Tier:
    """Rank a buy by how far the model departs from the devigged price."""
    if edge >= thr.min_edge + thr.strong_edge_gap:
        return Tier.STRONG
    return Tier.MODERATE


def bump_tier(tier: Tier, steps: int) -> Tier:
    order = [Tier.PASS, Tier.MODERATE, Tier.STRONG]
    i = max(0, min(len(order) - 1, order.index(tier) + steps))
    return order[i]


def classify(result: EVResult, thr: EVThresholds) -> tuple[Tier, list[str]]:
    reasons: list[str] = [f"EV={result.ev:+.3f} edge={result.edge:+.3f}"]

    # The price still has to pay at the best number we can bet.
    if result.ev <= thr.min_ev:
        reasons.append(f"EV {result.ev:+.3f} <= {thr.min_ev} -> pass")
        return Tier.PASS, reasons

    # Thin-edge guard: never buy without a real edge over the no-vig line.
    if result.edge < thr.min_edge:
        reasons.append(f"edge {result.edge:+.3f} < {thr.min_edge} -> pass")
        return Tier.PASS, reasons

    # Implausible-edge guard: the market is the better forecaster, so a large
    # departure from it is evidence against the model, not a bigger bet.
    if result.edge > thr.max_edge:
        reasons.append(f"edge {result.edge:+.3f} > {thr.max_edge} -> pass")
        return Tier.PASS, reasons

    tier = _base_tier(result.edge, thr)

    # Strict selection: keep only Strong buys.
    if thr.strong_only and tier == Tier.MODERATE:
        tier = Tier.PASS
        reasons.append("strong-only mode -> pass")

    return tier, reasons
