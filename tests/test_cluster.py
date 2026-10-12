"""Phase 3: date-clustered bootstrap interval."""

from __future__ import annotations

import numpy as np

from mlb_engine.audit.cluster import clustered_mean_ci


def _data():
    rng = np.random.default_rng(0)
    dates = np.repeat([f"2026-08-{d:02d}" for d in range(1, 21)], 15)
    return rng.normal(-0.03, 0.1, dates.size), dates


def test_seed_fixes_bounds_and_matches_the_named_recipe() -> None:
    v, d = _data()
    lo, hi = clustered_mean_ci(v, d)
    assert (lo, hi) == clustered_mean_ci(v, d, b=2000, seed=7)
    assert (lo, hi) != clustered_mean_ci(v, d, seed=8)
    # the recipe written out: whole dates, default_rng(7), linear percentiles
    uniq = sorted(set(d))
    sums = np.array([v[d == u].sum() for u in uniq])
    cnts = np.array([(d == u).sum() for u in uniq], dtype=float)
    idx = np.random.default_rng(7).integers(0, len(uniq), size=(2000, len(uniq)))
    ref = np.percentile(sums[idx].sum(1) / cnts[idx].sum(1), [2.5, 97.5])
    assert np.allclose((lo, hi), ref, rtol=0, atol=1e-15)
    assert lo < v.mean() < hi


def test_duplicating_rows_within_dates_does_not_narrow() -> None:
    v, d = _data()
    lo, hi = clustered_mean_ci(v, d)
    lo2, hi2 = clustered_mean_ci(np.concatenate([v, v]), np.concatenate([d, d]))
    assert np.isclose(hi - lo, hi2 - lo2, rtol=0, atol=1e-12)


def test_empty_is_nan() -> None:
    lo, hi = clustered_mean_ci([], [])
    assert np.isnan(lo) and np.isnan(hi)
