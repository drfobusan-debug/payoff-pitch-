"""Classify EV results into Strong buy / Moderate buy / Pass tiers.

A bet has to clear the EV floor at the price we can actually get, and its edge
over the no-vig market then has to clear a floor -- not EV, because ``EV =
decimal_odds x edge`` makes an EV cutoff a cheaper bar the longer the price.
Edge also has a ceiling: past ``max_edge`` a disagreement reads as a model
error. Two level screens sit past it -- a conviction floor on the probability
being bet and a ceiling on claimed EV -- because a bet can clear every relative
test while still being a cheap ticket or an unmeasurable tail.

What survives is then *ranked* by the market's own devigged probability rather
than by our edge over it, because on graded buys the edge ranks the two tiers
backwards (see :func:`_base_tier`). Tiers are finally adjusted by VSIN
handle/bets divergence.
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
    # Price ceiling. The engine's plus-money buys are its overconfidence being
    # cashed, so a long price is a veto rather than a bigger payout.
    price = result.best_quote.american
    if price > thr.max_buy_odds:
        return (
            "price_ceiling",
            f"price {price:+.0f} longer than {thr.max_buy_odds:+.0f} -> pass",
        )
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
    # Conviction floor, read on the probability the screen bets on -- anchored
    # where the market anchor is on, so it asks whether the selection survives
    # being pulled toward the price rather than whether the model liked it.
    if result.model_prob < thr.min_prob:
        return (
            "prob_floor",
            f"bet prob {result.model_prob:.3f} < {thr.min_prob} -> pass",
        )
    # EV tail guard: ``max_edge`` caps the disagreement, but a long price turns a
    # capped edge into an uncapped EV, and return falls as claimed EV rises.
    if result.ev > thr.max_ev:
        return "ev_ceiling", f"EV {result.ev:+.3f} > {thr.max_ev} -> pass"
    return None


def _base_tier(result: EVResult, thr: EVThresholds) -> tuple[Tier, str]:
    """Rank a buy by the market's own probability, not by our departure from it.

    The edge gap ranked backwards: its Strong tier went 47.7% for -9.9% ROI
    (n=1,435) against Moderate's 51.1% for -2.6% (n=919), because the model's
    biggest disagreements are its worst rows -- realized .623 at zero claimed
    edge falling to .408 past 20 points. The devigged price is monotone the
    right way on the same buys: 33.1% below .45 fair, 46.8% at .45-.50, 56.5%
    at .55-.60, 62.8% from .58 up, 65.9% at .65-.75.

    A row whose vig was never removed carries no such number -- the raw implied
    price overstates the market by about half the hold, so reading it as a fair
    probability would promote exactly the one-sided longshots this is meant to
    demote. Those keep the edge ranking rather than borrowing evidence they do
    not have.
    """
    if thr.strong_fair_prob < 1.0 and result.devig_coverage > 0.0:
        if result.fair_prob >= thr.strong_fair_prob:
            return Tier.STRONG, f"fair {result.fair_prob:.3f} >= {thr.strong_fair_prob} -> strong"
        return Tier.MODERATE, f"fair {result.fair_prob:.3f} < {thr.strong_fair_prob} -> moderate"
    if result.edge >= thr.min_edge + thr.strong_edge_gap:
        return Tier.STRONG, "no devigged price; edge gap -> strong"
    return Tier.MODERATE, "no devigged price; edge gap -> moderate"


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

    tier, basis = _base_tier(result, thr)
    reasons.append(basis)

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
