"""American odds conversions and vig removal."""

from __future__ import annotations


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def american_to_prob(american: float) -> float:
    """Implied probability including vig."""
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        return 0.0
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def prob_to_american(prob: float) -> float:
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return decimal_to_american(1.0 / prob)


def remove_vig(probs: list[float]) -> list[float]:
    """Normalize a set of vigged implied probabilities to sum to 1 (no-vig)."""
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


def no_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    na, nb = remove_vig([pa, pb])
    return na, nb
