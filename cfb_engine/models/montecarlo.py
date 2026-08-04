"""Monte Carlo game-score simulation.

Every game reduces to two numbers: the expected **margin** (home minus away)
and the expected **total** (home plus away). The simulator draws a correlated
bivariate-normal sample of (margin, total) ``n_sims`` times and reads the three
markets straight off the sampled arrays:

* moneyline  -> ``P(margin > 0)`` for the home side,
* ATS        -> ``P(margin + home_spread > 0)``,
* total      -> ``P(total > line)`` for the over.

Ties are handled explicitly (a moneyline/ATS push at margin exactly 0, a total
push at the line) so probabilities are conditioned on a decision, matching how
the book grades and refunds pushes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfb_engine.config import ModelParams


@dataclass
class ExpectedGame:
    """The ratings/market-implied means the simulator draws around."""

    exp_margin: float  # home minus away, HFA included
    exp_total: float
    margin_sd: float
    total_sd: float


@dataclass
class GameSimResult:
    margins: np.ndarray  # home - away, per sim
    totals: np.ndarray  # home + away, per sim

    @property
    def exp_margin(self) -> float:
        return float(self.margins.mean())

    @property
    def exp_margin_sd(self) -> float:
        return float(self.margins.std())

    @property
    def exp_total(self) -> float:
        return float(self.totals.mean())

    @property
    def exp_total_sd(self) -> float:
        return float(self.totals.std())

    # -- market probabilities --------------------------------------------
    def home_win_prob(self) -> float:
        """P(home wins | not a tie)."""
        decided = self.margins != 0
        n = int(decided.sum())
        if n == 0:
            return 0.5
        return float((self.margins > 0).sum()) / n

    def cover_prob(self, home_point: float) -> float:
        """P(home covers a spread of ``home_point`` | not a push).

        A home favorite is quoted -X, so ``home_point`` is negative; the home
        side covers when ``margin + home_point > 0``.
        """
        adj = self.margins + home_point
        decided = adj != 0
        n = int(decided.sum())
        if n == 0:
            return 0.5
        return float((adj > 0).sum()) / n

    def over_prob(self, line: float) -> float:
        """P(total goes over ``line`` | not a push)."""
        decided = self.totals != line
        n = int(decided.sum())
        if n == 0:
            return 0.5
        return float((self.totals > line).sum()) / n


class MonteCarlo:
    def __init__(self, params: ModelParams, *, seed: int = 7) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def simulate(self, exp: ExpectedGame) -> GameSimResult:
        n = self.params.n_sims
        rho = max(-0.95, min(0.95, self.params.margin_total_corr))
        cov = [
            [exp.margin_sd**2, rho * exp.margin_sd * exp.total_sd],
            [rho * exp.margin_sd * exp.total_sd, exp.total_sd**2],
        ]
        draws = self.rng.multivariate_normal([exp.exp_margin, exp.exp_total], cov, size=n)
        return GameSimResult(margins=draws[:, 0], totals=draws[:, 1])
