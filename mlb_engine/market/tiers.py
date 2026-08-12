"""Classify EV results into Strong buy / Moderate buy / Pass tiers.

A bet has to clear the EV floor at the price we can actually get, and is then
ranked on its edge over the no-vig market -- not on EV, because ``EV =
decimal_odds x edge`` makes an EV cutoff a cheaper bar the longer the price.
Edge also has a ceiling: past ``max_edge`` a disagreement reads as a model
error. Tiers are then adjusted by VSIN handle/bets divergence.
"""

from __future__ import annotations

from enum import Enum

from mlb_engine.config import EVThresholds
from mlb_engine.market.ev import EVResult

SHARP_DIVERGENCE_STRONG = 15.0  # handle-bets gap considered a strong sharp signal


class Tier(str, Enum):
    STRONG = "Strong buy"
    MODERATE = "Moderate buy"
    PASS = "Pass"


def price_screen(result: EVResult, thr: EVThresholds) -> tuple[str, str] | None:
    """Name the price screen that rejects this selection, or None if it clears.

    Returns ``(gate, reason)``: a short machine-readable gate name for the
    ledger, and the human sentence :func:`classify` puts in ``reasons``. The
    name is what makes a screen gradeable -- ``edge_ceiling`` in particular
    rejects picks the model likes *most*, so whether it removes losers or
    winners can only be settled by grading its own rows.
    """
    # The price still has to pay at the best number we can bet.
    if result.ev <= thr.min_ev:
        return "ev_floor", f"EV {result.ev:+.3f} <= {thr.min_ev} -> pass"
    # Thin-edge guard: never buy without a real edge over the no-vig line.
    if result.edge < thr.min_edge:
        return "thin_edge", f"edge {result.edge:+.3f} < {thr.min_edge} -> pass"
    # Implausible-edge guard: the market is the better forecaster, so a large
    # departure from it is evidence against the model, not a bigger bet.
    if result.edge > thr.max_edge:
        return "edge_ceiling", f"edge {result.edge:+.3f} > {thr.max_edge} -> pass"
    return None


def _base_tier(edge: float, thr: EVThresholds) -> Tier:
    """Rank a buy by how far the model departs from the devigged price."""
    if edge >= thr.min_edge + thr.strong_edge_gap:
        return Tier.STRONG
    return Tier.MODERATE


def bump_tier(tier: Tier, steps: int) -> Tier:
    order = [Tier.PASS, Tier.MODERATE, Tier.STRONG]
    i = max(0, min(len(order) - 1, order.index(tier) + steps))
    return order[i]


def _bump(tier: Tier, steps: int) -> Tier:
    return bump_tier(tier, steps)


def classify(result: EVResult, thr: EVThresholds) -> tuple[Tier, list[str]]:
    reasons: list[str] = []
    reasons.append(f"EV={result.ev:+.3f} edge={result.edge:+.3f}")

    screened = price_screen(result, thr)
    if screened is not None:
        reasons.append(screened[1])
        return Tier.PASS, reasons

    tier = _base_tier(result.edge, thr)

    div = result.sharp_divergence
    if div is not None:
        reasons.append(f"handle-bets={div:+.0f}")
        if div <= -SHARP_DIVERGENCE_STRONG:
            tier = _bump(tier, -1)
            reasons.append("sharp money against -> downgrade")
        elif div >= SHARP_DIVERGENCE_STRONG and tier == Tier.MODERATE:
            tier = _bump(tier, +1)
            reasons.append("sharp money with -> upgrade")

    # Strict selection: keep only Strong buys.
    if thr.strong_only and tier == Tier.MODERATE:
        tier = Tier.PASS
        reasons.append("strong-only mode -> pass")

    return tier, reasons
