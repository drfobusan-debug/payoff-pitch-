"""Confidence intervals that respect the slate: a whole date is one draw."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def clustered_mean_ci(
    values: Sequence[float] | np.ndarray,
    dates: Sequence[object] | np.ndarray,
    b: int = 2000,
    seed: int = 7,
) -> tuple[float, float]:
    """95% interval for a row-weighted mean, resampling whole dates with replacement.

    Each of ``b`` replicates draws as many dates as there are, with replacement,
    from ``numpy.random.default_rng(seed)`` and takes the mean over every row of
    the drawn dates. The bounds are the 2.5 and 97.5 percentiles with NumPy's
    default linear interpolation. Rows on one date move together, so copying a
    date's rows does not narrow the interval.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    _, inv = np.unique(np.asarray([str(d) for d in dates]), return_inverse=True)
    k = int(inv.max()) + 1
    sums = np.bincount(inv, weights=v, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(b, k))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
