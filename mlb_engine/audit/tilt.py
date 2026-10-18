"""Exponential tilts: the market owns the main line, the simulator the shape around it.

A prop count is tilted by one ``lam``: ``p(k) ~ p_sim(k) exp(lam k)``. A game's
home x away runs grid is tilted by ``alpha`` on the sum and ``beta`` on the
margin: ``p(h, a) ~ p_sim(h, a) exp(alpha (h + a) + beta (h - a))``. (The pair
``lam_h = alpha + beta``, ``lam_a = alpha - beta`` is the same tilt written per
team; it is not stored.) Each knob is solved so a two-way devigged price is
matched exactly, pushes on a whole-number line dropped from both sides the way
the card's own two-way pricing drops them.

Measurement only: nothing here is read by the live card.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

TOL = 1e-4
LAM_BOUND = 25.0
PINNED = "pinned"
UNPINNED = "unpinned"
NO_RL = "no_rl"
Point = np.ndarray | Sequence[float]
_STARTS = tuple((a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0) if (a, b) != (0, 0))


# --- events on the game grid ------------------------------------------------


@dataclass(frozen=True)
class TwoWay:
    """A two-way market as two disjoint events on the grid: ``P(yes) / (P(yes) + P(no))``.

    Bins in neither event (a push on a whole-number line, a tie on the
    moneyline) belong to neither side, exactly as the card prices them.
    """

    yes: np.ndarray
    no: np.ndarray
    target: float

    def prob(self, q: np.ndarray) -> float:
        py, pn = float(q[self.yes].sum()), float(q[self.no].sum())
        return py / (py + pn) if py + pn > 0 else float("nan")

    def feasible(self, p: np.ndarray) -> bool:
        return bool(0.0 < self.target < 1.0 and p[self.yes].sum() > 0 and p[self.no].sum() > 0)


def grid_axes(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Sum and margin of every ``[home, away]`` bin."""
    h, a = np.indices(shape)
    return (h + a).astype(float), (h - a).astype(float)


def total_market(shape: tuple[int, int], line: float, over_prob: float) -> TwoWay:
    s, _ = grid_axes(shape)
    return TwoWay(s > line, s < line, over_prob)


def ml_market(shape: tuple[int, int], home_prob: float) -> TwoWay:
    _, m = grid_axes(shape)
    return TwoWay(m > 0, m < 0, home_prob)


def rl_market(shape: tuple[int, int], home_point: float, home_prob: float) -> TwoWay:
    """The home side's run line at ``home_point`` (``-1.5``: home must win by 2+)."""
    _, m = grid_axes(shape)
    return TwoWay(m + home_point > 0, m + home_point < 0, home_prob)


# --- 2D solve ---------------------------------------------------------------


def tilt_grid(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    s, m = grid_axes(p.shape)
    with np.errstate(over="ignore"):
        logw = alpha * s + beta * m
        w = np.where(p > 0, np.exp(logw - logw[p > 0].max(initial=0.0)), 0.0)
    q = p * w
    tot = q.sum()
    return q / tot if tot > 0 else q


def residuals(p: np.ndarray, mk: Sequence[TwoWay], x: Point) -> np.ndarray:
    q = tilt_grid(p, float(x[0]), float(x[1]))
    return np.array([c.prob(q) - c.target for c in mk])


def jacobian(p: np.ndarray, mk: Sequence[TwoWay], x: Point) -> np.ndarray:
    """d prob / d(alpha, beta) for each two-way market.

    For a tail ``E``, ``dP(E)/d alpha = Cov(1_E, s)`` and ``dP(E)/d beta =
    Cov(1_E, m)`` under the tilted grid. A two-way price ``r = P(Y) / (P(Y) +
    P(N))`` takes the quotient rule over its two tails. (The covariance of the
    features with each other is the Jacobian for matching *means*; aimed at a
    tail it walks to a different root.)
    """
    q = tilt_grid(p, float(x[0]), float(x[1]))
    s, m = grid_axes(p.shape)
    es, em = float((q * s).sum()), float((q * m).sum())
    out = np.zeros((len(mk), 2))
    for i, c in enumerate(mk):
        py, pn = float(q[c.yes].sum()), float(q[c.no].sum())
        dy = np.array([(q * s)[c.yes].sum() - py * es, (q * m)[c.yes].sum() - py * em])
        dn = np.array([(q * s)[c.no].sum() - pn * es, (q * m)[c.no].sum() - pn * em])
        den = (py + pn) ** 2
        out[i] = (dy * pn - py * dn) / den if den > 0 else 0.0
    return out


def _newton(p: np.ndarray, mk: Sequence[TwoWay], x0: tuple[float, float]) -> np.ndarray:
    x = np.array(x0, dtype=float)
    r = residuals(p, mk, x)
    for _ in range(60):
        if np.max(np.abs(r)) <= TOL * 1e-3:
            break
        try:
            step = np.linalg.solve(jacobian(p, mk, x), -r)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-4:
            cand = x + t * step
            rc = residuals(p, mk, cand)
            if np.all(np.isfinite(rc)) and np.linalg.norm(rc) < np.linalg.norm(r):
                x, r = cand, rc
                break
            t /= 2
        else:
            break
    return x


def _lsq(p: np.ndarray, mk: Sequence[TwoWay], x0: tuple[float, float]) -> np.ndarray:
    sol = least_squares(
        lambda x: residuals(p, mk, x), np.array(x0, dtype=float),
        jac=lambda x: jacobian(p, mk, x), xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=200,
    )
    return np.asarray(sol.x, dtype=float)


@dataclass(frozen=True)
class Candidate:
    solver: str
    x: tuple[float, float]
    res: tuple[float, float]

    @property
    def landed(self) -> bool:
        return all(np.isfinite(self.res)) and max(abs(v) for v in self.res) <= TOL

    @property
    def norm(self) -> float:
        return float(np.hypot(*self.x))


def select_root(cands: Sequence[Candidate]) -> Candidate | None:
    """The smallest step among the candidates that landed; a miss never beats a land."""
    landed = [c for c in cands if c.landed]
    return min(landed, key=lambda c: c.norm) if landed else None


@dataclass(frozen=True)
class Pin2D:
    status: str
    alpha: float | None
    beta: float | None
    solver: str
    alpha_newton: float | None
    beta_newton: float | None
    alpha_lsq: float | None
    beta_lsq: float | None
    n_landed: int
    residual_1: float | None
    residual_2: float | None
    q: np.ndarray | None = field(default=None, repr=False, compare=False)


def pin_game(p: np.ndarray, mk: Sequence[TwoWay]) -> Pin2D:
    """Solve ``(alpha, beta)`` so the tilted grid prices both markets exactly.

    Newton and ``least_squares`` both start at (0, 0). A candidate *lands* when
    both residuals are within ``TOL``. If neither lands, ``least_squares``
    retries from the eight starts in {-1, 0, 1}^2. The smallest landed step is
    kept and ``solver`` names it; nothing landing is ``unpinned``.
    """
    p = p / p.sum()
    if len(mk) != 2 or not all(c.feasible(p) for c in mk):
        return Pin2D(UNPINNED, None, None, "", None, None, None, None, 0, None, None)
    cands: list[Candidate] = []
    xn = _newton(p, mk, (0.0, 0.0))
    xl = _lsq(p, mk, (0.0, 0.0))
    for name, x in (("newton", xn), ("lsq", xl)):
        r = residuals(p, mk, x)
        cands.append(Candidate(name, (float(x[0]), float(x[1])), (float(r[0]), float(r[1]))))
    if not any(c.landed for c in cands):
        for start in _STARTS:
            x = _lsq(p, mk, start)
            r = residuals(p, mk, x)
            cands.append(
                Candidate("lsq_restart", (float(x[0]), float(x[1])), (float(r[0]), float(r[1])))
            )
    best = select_root(cands)
    n_landed = sum(c.landed for c in cands)
    newton, lsq = cands[0], cands[1]
    if best is None:
        worst = min(cands, key=lambda c: float(np.hypot(*c.res)) if all(np.isfinite(c.res)) else 1e9)
        return Pin2D(
            UNPINNED, None, None, "", newton.x[0], newton.x[1], lsq.x[0], lsq.x[1], 0,
            worst.res[0], worst.res[1],
        )
    return Pin2D(
        PINNED, best.x[0], best.x[1], best.solver, newton.x[0], newton.x[1], lsq.x[0], lsq.x[1],
        n_landed, best.res[0], best.res[1], tilt_grid(p, *best.x),
    )


# --- 1D prop tilt -----------------------------------------------------------


def _two_way_1d(k: np.ndarray, w: np.ndarray, line: float) -> float:
    py, pn = float(w[k > line].sum()), float(w[k < line].sum())
    return py / (py + pn) if py + pn > 0 else float("nan")


def tilt_1d(k: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    with np.errstate(over="ignore"):
        logw = lam * k
        e = np.where(w > 0, np.exp(logw - logw[w > 0].max(initial=0.0)), 0.0)
    q = w * e
    tot = q.sum()
    return q / tot if tot > 0 else q


@dataclass(frozen=True)
class Pin1D:
    status: str
    lam: float | None
    residual: float | None
    q: np.ndarray | None = field(default=None, repr=False, compare=False)


def pin_prop(k: np.ndarray, w: np.ndarray, line: float, over_prob: float) -> Pin1D:
    """Solve ``lam`` so the tilted count prices the two-way over at ``over_prob``."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if not (0 < over_prob < 1) or w[k > line].sum() <= 0 or w[k < line].sum() <= 0:
        return Pin1D(UNPINNED, None, None)

    def f(lam: float) -> float:
        return _two_way_1d(k, tilt_1d(k, w, lam), line) - over_prob

    lo, hi = -LAM_BOUND, LAM_BOUND
    if f(lo) * f(hi) > 0:
        return Pin1D(UNPINNED, None, None)
    lam = float(brentq(f, lo, hi, xtol=1e-12))
    return Pin1D(PINNED, lam, f(lam), tilt_1d(k, w, lam))


def prop_over(k: np.ndarray, q: np.ndarray, line: float) -> float:
    return _two_way_1d(np.asarray(k, dtype=float), q, line)


def ess(counts: np.ndarray, weights: np.ndarray) -> float:
    """Kish effective sample size of path counts ``n_i`` reweighted by ``w_i``."""
    n, w = np.asarray(counts, dtype=float).ravel(), np.asarray(weights, dtype=float).ravel()
    den = float((n * w * w).sum())
    return float((n * w).sum()) ** 2 / den if den > 0 else 0.0


# --- prop conditional on the pinned game -----------------------------------


@dataclass(frozen=True)
class ConditionalRead:
    status: str
    lam: float | None
    q: np.ndarray | None  # prop count distribution after the read
    total_drift: float | None
    ess: float | None


def conditional_prop(
    counts: np.ndarray,
    p_tilt_total: np.ndarray,
    line: float,
    over_prob: float,
    total_line: float,
    *,
    ipf_steps: int = 0,
) -> ConditionalRead:
    """A prop read on the pinned game: reweight ``(k, t)`` by the tilted total, then pin ``k``.

    ``w(t) = p_tilt(t) / p_sim(t)``; a bin with no simulated mass, or with a
    single simulated path, gets weight 0 before ``lam`` is solved, so one path
    never sets the prop. ``total_drift`` is how far the joint's over-the-main-
    total probability then sits from the pin. ``ipf_steps`` > 0 alternates the
    total marginal and the prop pin that many times (scored separately).
    """
    c = np.asarray(counts, dtype=float)
    col = c.sum(axis=0)
    n = col.sum()
    nt = min(len(col), len(p_tilt_total))
    p_sim = col / n if n > 0 else col
    target_t = np.zeros(len(col))
    target_t[:nt] = np.asarray(p_tilt_total, dtype=float)[:nt]
    wt = np.where((col > 1) & (p_sim > 0), target_t / np.where(p_sim > 0, p_sim, 1.0), 0.0)
    joint = c * wt[None, :]
    if joint.sum() <= 0:
        return ConditionalRead(UNPINNED, None, None, None, None)
    joint /= joint.sum()
    kvals = np.arange(c.shape[0], dtype=float)
    tvals = np.arange(c.shape[1], dtype=float)
    lam = 0.0
    for _ in range(max(1, ipf_steps)):
        pin = pin_prop(kvals, joint.sum(axis=1), line, over_prob)
        if pin.status != PINNED or pin.lam is None:
            return ConditionalRead(UNPINNED, None, None, None, None)
        lam += pin.lam
        joint = joint * np.exp(pin.lam * (kvals - kvals.max()))[:, None]
        joint /= joint.sum()
        if ipf_steps:
            marg = joint.sum(axis=0)
            fix = np.where(marg > 0, target_t / np.where(marg > 0, marg, 1.0), 0.0)
            joint = joint * fix[None, :]
            joint /= joint.sum()
    if ipf_steps:
        final = pin_prop(kvals, joint.sum(axis=1), line, over_prob)
        if final.status == PINNED and final.lam is not None:
            lam += final.lam
            joint = joint * np.exp(final.lam * (kvals - kvals.max()))[:, None]
            joint /= joint.sum()
    marg_t = joint.sum(axis=0)
    drift = _two_way_1d(tvals, marg_t, total_line) - _two_way_1d(tvals, target_t, total_line)
    weights = np.exp(lam * kvals)[:, None] * wt[None, :]
    return ConditionalRead(PINNED, lam, joint.sum(axis=1), drift, ess(c, weights))


# --- holdouts ---------------------------------------------------------------


@dataclass(frozen=True)
class GameRead:
    pin: Pin2D
    rl_residual: float | None
    ml_holdout_gap: float | None
    total_holdout_gap: float | None
    total_holdout_status: str


def read_game(
    p: np.ndarray,
    total_line: float,
    total_over: float,
    ml_home: float,
    rl_home_point: float | None,
    rl_home_prob: float | None,
) -> GameRead:
    """Pin total + moneyline; read the run line; run both holdouts.

    All gaps are model minus market. With the total pinned, the moneyline
    holdout's gap and the run-line residual have opposite signs on a correct
    solve -- one is the other rescaled -- so only the violation count is
    reported. The total holdout (moneyline + run line pinned, total read) is the
    rotation that asks something the main pin cannot.
    """
    shape = p.shape
    tot = total_market(shape, total_line, total_over)
    ml = ml_market(shape, ml_home)
    pin = pin_game(p, [tot, ml])
    if rl_home_point is None or rl_home_prob is None:
        return GameRead(pin, None, None, None, NO_RL)
    rl = rl_market(shape, rl_home_point, rl_home_prob)
    rl_res = rl.prob(pin.q) - rl_home_prob if pin.q is not None else None
    hold_ml = pin_game(p, [tot, rl])
    ml_gap = ml.prob(hold_ml.q) - ml_home if hold_ml.q is not None else None
    hold_tot = pin_game(p, [ml, rl])
    tot_gap = tot.prob(hold_tot.q) - total_over if hold_tot.q is not None else None
    return GameRead(pin, rl_res, ml_gap, tot_gap, hold_tot.status)


def sign_violations(reads: Sequence[GameRead]) -> int:
    """Games where the moneyline holdout and run-line residual share a sign (expected 0)."""
    bad = 0
    for r in reads:
        a, b = r.rl_residual, r.ml_holdout_gap
        if a is None or b is None or abs(a) < TOL or abs(b) < TOL:
            continue
        if (a > 0) == (b > 0):
            bad += 1
    return bad


