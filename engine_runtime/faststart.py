"""Fast-start study: does an early scoring surprise stay in the final, and does the rest of the
game play to the pregame total? Graded only; this prices nothing and gates nothing.

For each finished game with a timely pregame total ``T`` and ``P`` points scored by a checkpoint
(end of Q1, half, ...)::

    gap  = P - share * T      # share: points by the checkpoint per point of pregame total
    miss = final - T
    miss = a + b * gap

``share`` for a game is fit only on games from earlier days (no look-ahead); a game is used once
``MIN_PRIOR`` earlier games exist. ``b = 1``: the start stays in and the rest plays to ``T``.
``b > 1``: the start carries on. ``b < 1``: it fades. The converse is read on its own: slow
(gap < 0) and fast (gap > 0) starts get separate slopes under one intercept.

Each slope is read at its own precision, not a blanket sample cut: a slope is called only once its
standard error is at most ``SE_TARGET`` (enough to tell 1.10 from 1.00 at 2 SE), and the slow-vs-fast
difference only once its SE is at most ``DIFF_SE_TARGET``. Before that the row says underpowered
and roughly how many games it would need. SEs are heteroskedasticity-robust (HC1).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

MIN_PRIOR = 30
SE_TARGET = 0.05
DIFF_SE_TARGET = 0.10


@dataclass(frozen=True)
class Obs:
    """One finished game at one checkpoint, all in points (goals for hockey)."""

    game_id: str
    day: str  # slate date, ISO
    pregame: float  # the timely pregame total
    so_far: float  # both teams' points at the checkpoint
    final: float  # both teams' final points, as the total is graded


def gaps(obs: Iterable[Obs], *, min_prior: int = MIN_PRIOR) -> list[tuple[Obs, float, float]]:
    """(obs, share, gap) for each game with at least ``min_prior`` games on earlier days."""
    by_day: dict[str, list[Obs]] = {}
    for o in obs:
        by_day.setdefault(o.day, []).append(o)
    out: list[tuple[Obs, float, float]] = []
    n = 0
    pts = tot = 0.0
    for day in sorted(by_day):
        if n >= min_prior and tot > 0:
            share = pts / tot
            out += [(o, share, o.so_far - share * o.pregame) for o in by_day[day]]
        for o in by_day[day]:
            n += 1
            pts += o.so_far
            tot += o.pregame
    return out


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients, HC1 covariance and residuals."""
    n, k = x.shape
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ beta
    xtxi = np.linalg.pinv(x.T @ x)
    cov = xtxi @ (x.T * e**2) @ x @ xtxi * (n / max(n - k, 1))
    return beta, cov, e


def _need(n: int, se: float, target: float) -> int:
    return math.ceil(n * (se / target) ** 2)


@dataclass(frozen=True)
class Read:
    checkpoint: str
    n: int
    share: float  # the latest share used
    a: float
    b: float
    se: float
    slow_n: int
    b_slow: float
    se_slow: float
    fast_n: int
    b_fast: float
    se_fast: float
    diff_se: float
    rmse_frozen: float  # final predicted as pregame total + gap
    rmse_fit: float  # final predicted with the fitted a + b * gap

    @property
    def verdict(self) -> str:
        if self.se > SE_TARGET:
            return f"underpowered (needs ~{_need(self.n, self.se, SE_TARGET)} games)"
        z = (self.b - 1.0) / self.se
        if z > 2:
            return f"carry-over: {self.b - 1:+.0%} of the gap added"
        if z < -2:
            return f"fade: {self.b - 1:+.0%} of the gap given back"
        return "holds: start stays in, rest plays to the pregame total"

    @property
    def converse(self) -> str:
        if self.slow_n < 3 or self.fast_n < 3:
            return "too few slow or fast starts"
        diff = self.b_fast - self.b_slow
        if self.diff_se > DIFF_SE_TARGET:
            return f"underpowered (needs ~{_need(self.n, self.diff_se, DIFF_SE_TARGET)} games)"
        if abs(diff) > 2 * self.diff_se:
            return f"differ: fast {diff:+.2f} vs slow"
        return "same both ways"


def read(checkpoint: str, obs: Iterable[Obs], *, min_prior: int = MIN_PRIOR) -> Read | None:
    """The checkpoint's slope, its slow/fast split and the two misses; None under 4 usable games."""
    rows = gaps(obs, min_prior=min_prior)
    if len(rows) < 4:
        return None
    g = np.array([r[2] for r in rows], float)
    y = np.array([r[0].final - r[0].pregame for r in rows], float)
    one = np.ones(len(g))
    beta, cov, e = _ols(np.c_[one, g], y)
    pw, pcov, _ = _ols(np.c_[one, np.minimum(g, 0.0), np.maximum(g, 0.0)], y)
    frozen = y - g
    return Read(
        checkpoint=checkpoint,
        n=len(g),
        share=rows[-1][1],
        a=float(beta[0]),
        b=float(beta[1]),
        se=float(math.sqrt(cov[1, 1])),
        slow_n=int((g < 0).sum()),
        b_slow=float(pw[1]),
        se_slow=float(math.sqrt(pcov[1, 1])),
        fast_n=int((g > 0).sum()),
        b_fast=float(pw[2]),
        se_fast=float(math.sqrt(pcov[2, 2])),
        diff_se=float(math.sqrt(max(pcov[1, 1] + pcov[2, 2] - 2 * pcov[1, 2], 0.0))),
        rmse_frozen=float(math.sqrt(np.mean((frozen - frozen.mean()) ** 2))),
        rmse_fit=float(math.sqrt(np.mean(e**2))),
    )


HEAD = (
    "checkpoint",
    "games",
    "share",
    "b (all)",
    "verdict",
    "b slow starts",
    "b fast starts",
    "slow vs fast",
    "miss frozen / fitted",
)

NOTE = (
    "final - pregame total = a + b x (points so far - share x pregame total); share fit on earlier days only. "
    "b = 1: the start stays in and the rest plays to the pregame total; > 1 carries on; < 1 fades. "
    f"Called at SE <= {SE_TARGET} (slope) and <= {DIFF_SE_TARGET} (slow vs fast), 2 SE. "
    "Graded only: whether live books misprice it is not answerable without archived in-play totals."
)


def table(reads: Sequence[Read]) -> list[tuple[str, ...]]:
    return [
        (
            r.checkpoint,
            str(r.n),
            f"{r.share:.1%}",
            f"{r.b:.2f} ± {r.se:.2f}",
            r.verdict,
            f"{r.b_slow:.2f} ± {r.se_slow:.2f} (n={r.slow_n})",
            f"{r.b_fast:.2f} ± {r.se_fast:.2f} (n={r.fast_n})",
            r.converse,
            f"{r.rmse_frozen:.2f} / {r.rmse_fit:.2f}",
        )
        for r in reads
    ]


__all__ = ["HEAD", "NOTE", "Obs", "Read", "gaps", "read", "table"]
