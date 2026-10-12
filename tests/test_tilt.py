"""Phase 2b: 1D and 2D exponential tilts, root choice, conditional props, holdouts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import poisson

from mlb_engine.audit import tilt

K = np.arange(26)


def _grid(lh: float, la: float, mix: float = 0.3, bump: float = 1.0) -> np.ndarray:
    base = np.outer(poisson.pmf(K, lh), poisson.pmf(K, la))
    corr = np.outer(poisson.pmf(K, lh + bump), poisson.pmf(K, la + bump))
    g = (1 - mix) * base + mix * corr
    return g / g.sum()


def _markets(g: np.ndarray, line: float = 8.5):
    s, m = tilt.grid_axes(g.shape)
    over = float(g[s > line].sum() / (g[s > line].sum() + g[s < line].sum()))
    home = float(g[m > 0].sum() / (g[m > 0].sum() + g[m < 0].sum()))
    pt = -1.5 if home >= 0.5 else 1.5
    rl = tilt.rl_market(g.shape, pt, 0.0).prob(g)
    return over, home, pt, rl


def test_pin_prop_matches_target_and_drops_push() -> None:
    w = poisson.pmf(K, 5.8)
    for line, target in ((5.5, 0.47), (6.0, 0.52)):
        pin = tilt.pin_prop(K, w, line, target)
        assert pin.status == tilt.PINNED and pin.q is not None
        assert tilt.prop_over(K, pin.q, line) == pytest.approx(target, abs=1e-9)
    assert tilt.pin_prop(K[:5], w[:5], 5.5, 0.4).status == tilt.UNPINNED


def test_pin_game_lands_both_markets() -> None:
    sim, mkt = _grid(4.8, 4.4), _grid(4.3, 4.1, mix=0.1)
    over, home, _, _ = _markets(mkt)
    mk = [tilt.total_market(sim.shape, 8.5, over), tilt.ml_market(sim.shape, home)]
    pin = tilt.pin_game(sim, mk)
    assert pin.status == tilt.PINNED and pin.n_landed >= 1 and pin.solver
    assert abs(pin.residual_1 or 1) <= tilt.TOL and abs(pin.residual_2 or 1) <= tilt.TOL
    assert pin.alpha is not None and pin.alpha < 0  # sim was hot: the sum knob turns down


def test_tail_jacobian_matches_finite_difference() -> None:
    sim = _grid(4.6, 4.2)
    mk = [tilt.total_market(sim.shape, 8.0, 0.5), tilt.ml_market(sim.shape, 0.55),
          tilt.rl_market(sim.shape, -1.5, 0.35)]
    x = np.array([0.07, -0.05])
    jac = tilt.jacobian(sim, mk, x)
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fd = (tilt.residuals(sim, mk, x + e) - tilt.residuals(sim, mk, x - e)) / (2 * h)
        assert np.allclose(jac[:, j], fd, atol=1e-7)


def test_a_miss_never_beats_a_land() -> None:
    near_miss = tilt.Candidate("newton", (0.01, 0.0), (5e-3, 0.0))
    far_land = tilt.Candidate("lsq_restart", (0.9, -0.4), (1e-6, -2e-6))
    nearer_land = tilt.Candidate("lsq", (0.5, 0.1), (1e-5, 0.0))
    assert tilt.select_root([near_miss, far_land]) == far_land
    assert tilt.select_root([near_miss, far_land, nearer_land]) == nearer_land
    assert tilt.select_root([near_miss]) is None


def test_infeasible_target_is_unpinned() -> None:
    sim = _grid(4.6, 4.2)
    pin = tilt.pin_game(sim, [tilt.total_market(sim.shape, 60.5, 0.5), tilt.ml_market(sim.shape, 0.5)])
    assert pin.status == tilt.UNPINNED and pin.alpha is None


def test_ess_scales_with_paths() -> None:
    n = np.array([10, 20, 30, 40])
    w = np.array([0.5, 1.0, 1.5, 2.0])
    assert tilt.ess(n, np.ones(4)) == pytest.approx(100)
    assert tilt.ess(2 * n, w) == pytest.approx(2 * tilt.ess(n, w))


def test_conditional_prop_weights_and_drift() -> None:
    rng = np.random.default_rng(1)
    t = rng.poisson(8.6, 20000)
    k = rng.binomial(t, 0.12)
    counts = np.zeros((k.max() + 1, t.max() + 3))
    np.add.at(counts, (k, t), 1)
    counts[0, t.max() + 2] = 1  # a one-path total bin far out in the tail
    p_sim = counts.sum(axis=0) / counts.sum()
    p_tilt = tilt.tilt_1d(np.arange(len(p_sim), dtype=float), p_sim, -0.04)
    one = tilt.conditional_prop(counts, p_tilt, 0.5, 0.60, 8.5)
    ipf = tilt.conditional_prop(counts, p_tilt, 0.5, 0.60, 8.5, ipf_steps=5)
    kv = np.arange(counts.shape[0], dtype=float)
    for read in (one, ipf):
        assert read.status == tilt.PINNED and read.q is not None
        assert tilt.prop_over(kv, read.q, 0.5) == pytest.approx(0.60, abs=1e-9)
    assert abs(ipf.total_drift or 0) <= abs(one.total_drift or 0) + 1e-9
    assert one.ess is not None and 0 < one.ess <= counts.sum()


def _sweep(n: int = 200):
    rng = np.random.default_rng(7)
    reads = []
    while len(reads) < n:
        sim = _grid(rng.uniform(3.6, 5.6), rng.uniform(3.6, 5.6), rng.uniform(0, 0.4))
        mkt = _grid(rng.uniform(3.6, 5.6), rng.uniform(3.6, 5.6), rng.uniform(0, 0.4), rng.uniform(0.5, 2))
        over, home, pt, rl = _markets(mkt)
        r = tilt.read_game(sim, 8.5, over, home, pt, rl)
        if r.pin.status == tilt.PINNED and r.ml_holdout_gap is not None:
            reads.append(r)
    return reads


def test_ml_holdout_opposite_sign_and_side_swap_fails() -> None:
    good = _sweep()
    assert len(good) >= 200 and tilt.sign_violations(good) == 0
    # The read taken on the other run-line side against this side's price: the
    # two sides are complements, so the residual is the true one negated and
    # every game now shares the holdout's sign.
    swapped = [replace(r, rl_residual=-(r.rl_residual or 0.0)) for r in good]
    assert tilt.sign_violations(swapped) > 0


def test_no_rl_skips_holdouts() -> None:
    sim = _grid(4.6, 4.2)
    r = tilt.read_game(sim, 8.5, 0.5, 0.55, None, None)
    assert r.total_holdout_status == tilt.NO_RL and r.rl_residual is None
