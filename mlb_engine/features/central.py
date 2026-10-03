"""Central-tendency switch for continuous Statcast reads.

Proportions (K%, barrel%, hard-hit%) have no mean/median choice; this only
governs the per-event continuous reads (exit velocity, bat speed, xwOBA and
wOBA on contact, pitches and batters faced per start). ``MLBE_CENTRAL=median``
swaps the mean for the median so the two can be graded against each other.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pandas as pd


def use_median() -> bool:
    return os.environ.get("MLBE_CENTRAL", "mean").strip().lower() == "median"


def central(values: pd.Series) -> float:
    s = values.dropna()
    if s.empty:
        return float("nan")
    return float(s.median()) if use_median() else float(s.mean())


def central_list(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    if use_median():
        v = sorted(values)
        n = len(v)
        return float(v[n // 2]) if n % 2 else float((v[n // 2 - 1] + v[n // 2]) / 2)
    return float(sum(values) / len(values))
