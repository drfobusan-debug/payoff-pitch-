"""Combine batter and pitcher PA-outcome rates into matchup probabilities.

Uses the log5 / odds-ratio method: for each outcome, the matchup rate is
proportional to (batter_rate * pitcher_rate / league_rate), then renormalized so
the seven mutually-exclusive outcomes sum to 1.
"""

from __future__ import annotations

from mlb_engine.features.rolling import LEAGUE_RATES, OutcomeRates

OUTCOME_KEYS = ["1B", "2B", "3B", "HR", "BB", "K", "OUT"]


def combine(batter: OutcomeRates, pitcher: OutcomeRates) -> dict[str, float]:
    b = batter.as_dict()
    p = pitcher.as_dict()
    raw: dict[str, float] = {}
    for k in OUTCOME_KEYS:
        lg = LEAGUE_RATES[k]
        if lg <= 0:
            raw[k] = 0.0
            continue
        raw[k] = max(b[k] * p[k] / lg, 1e-9)
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def apply_multipliers(rates: dict[str, float], multipliers: dict[str, float]) -> dict[str, float]:
    """Multiply selected outcome probabilities and renormalize.

    Used by the weather / biomechanics / travel filters to nudge outcomes.
    ``multipliers`` maps outcome key -> factor (missing keys default to 1.0).
    """
    adj = {k: rates[k] * multipliers.get(k, 1.0) for k in rates}
    total = sum(adj.values())
    if total <= 0:
        return rates
    return {k: v / total for k, v in adj.items()}
