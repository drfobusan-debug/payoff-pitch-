"""Empirical-Bayes shrinkage for rates, and the reliability fit that sets its ``k``.

A rate observed over exposure ``n`` (minutes, games, shots -- whatever the
metric's denominator is) is combined with a prior as

    posterior = (k * prior + n * observed) / (k + n)

``k`` is the exposure at which the observation and the prior get equal weight.
It is *fitted*, never chosen: if two independent halves of a season, each with
exposure ``n_half``, correlate at ``r`` across teams (or players), then the
signal fraction of one half is ``r`` and

    k = n_half * (1 - r) / r

because reliability at exposure ``n`` is ``n / (n + k)`` under the standard
split-half model. A metric that barely repeats (small ``r``) gets a large ``k``
and stays near its prior most of the season; a metric that repeats quickly gets
a small ``k`` and follows the data within a few weeks. This is what replaces a
single fixed "20-game ramp".
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def eb_posterior(prior: float, observed: float, n: float, k: float) -> float:
    """Shrink ``observed`` (over exposure ``n``) toward ``prior`` with strength ``k``."""
    if n <= 0:
        return prior
    if k <= 0:
        return observed
    return (k * prior + n * observed) / (k + n)


def reliability_at(n: float, k: float) -> float:
    """Fraction of the observed rate that is signal at exposure ``n``."""
    if n <= 0:
        return 0.0
    return n / (n + k)


def k_from_split_half(r: float, n_half: float) -> float:
    """Exposure at which observation and prior weigh equally, from split-half ``r``.

    Returns ``inf`` when ``r <= 0`` (the metric does not repeat at this exposure;
    the caller should ship the prior and say so).
    """
    if r <= 0:
        return math.inf
    if r >= 1:
        return 0.0
    return n_half * (1.0 - r) / r


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Plain Pearson correlation; ``nan`` for fewer than three pairs or a constant side."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return math.nan
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return math.nan
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return sxy / math.sqrt(sxx * syy)


def spearman_brown(r_half: float) -> float:
    """Reliability of the full sample given the correlation between its two halves."""
    if math.isnan(r_half):
        return math.nan
    return 2.0 * r_half / (1.0 + r_half)


__all__ = ["eb_posterior", "k_from_split_half", "pearson", "reliability_at", "spearman_brown"]
