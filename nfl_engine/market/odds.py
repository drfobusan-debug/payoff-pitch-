"""American-odds arithmetic."""

from __future__ import annotations


def american_to_decimal(american: float) -> float:
    if american >= 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def american_to_prob(american: float) -> float:
    dec = american_to_decimal(american)
    return 1.0 / dec if dec > 0 else 0.0


def prob_to_american(prob: float) -> float:
    p = min(max(prob, 1e-6), 1.0 - 1e-6)
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p
