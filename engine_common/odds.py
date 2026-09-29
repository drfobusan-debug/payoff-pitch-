"""American odds <-> probability, and multiplicative devigging."""

from __future__ import annotations

from collections.abc import Mapping


def american_to_prob(american: float) -> float:
    """Implied (vigged) probability of an American price."""
    if american == 0:
        raise ValueError("american odds cannot be 0")
    if american > 0:
        return 100.0 / (american + 100.0)
    return -american / (-american + 100.0)


def prob_to_american(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def devig(prices: Mapping[str, float]) -> dict[str, float]:
    """Multiplicative devig of one book's full market: implied probs scaled to sum 1."""
    raw = {k: american_to_prob(v) for k, v in prices.items()}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def overround(prices: Mapping[str, float]) -> float:
    """Sum of implied probabilities minus one (the book's hold on this market)."""
    return sum(american_to_prob(v) for v in prices.values()) - 1.0


__all__ = ["american_to_prob", "devig", "overround", "prob_to_american"]
