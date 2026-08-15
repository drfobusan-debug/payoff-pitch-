"""Bivariate-normal score model, kept as a control rather than a contender.

This is the CFB engine's simulator, ported deliberately unchanged in spirit: draw
(margin, total) from a correlated normal and split them into two scores. It is
the wrong shape for the NFL -- it puts 2.7% of games on a 3-point margin against
14.8% actual, so it prices -3 and -2.5 as very nearly the same bet -- and it is
here for exactly that reason. Because it consumes the same
:class:`~nfl_engine.models.drives.ExpectedGame` means, a backtest can run both
and attribute the difference to the distribution's shape rather than to the
forecast. Any result the possession simulator claims has to beat this.
"""

from __future__ import annotations

import numpy as np

from nfl_engine.models.distribution import ScoreDistribution
from nfl_engine.models.drives import ExpectedGame


class NormalSim:
    def __init__(
        self,
        *,
        n_sims: int = 40000,
        seed: int = 7,
        margin_sd: float = 13.2,
        total_sd: float = 13.4,
        corr: float = 0.05,
    ) -> None:
        self.n_sims = n_sims
        self.margin_sd = margin_sd
        self.total_sd = total_sd
        self.corr = max(-0.95, min(0.95, corr))
        self.rng = np.random.default_rng(seed)

    def simulate(self, exp: ExpectedGame) -> ScoreDistribution:
        cov = [
            [self.margin_sd**2, self.corr * self.margin_sd * self.total_sd],
            [self.corr * self.margin_sd * self.total_sd, self.total_sd**2],
        ]
        draws = self.rng.multivariate_normal([exp.margin(), exp.total()], cov, size=self.n_sims)
        margin = np.round(draws[:, 0])
        total = np.round(draws[:, 1])
        # Whole-point scores, so pushes exist at all: without them every
        # half-point question answers identically and the control flatters
        # itself. Margin and total must share parity to split evenly; nudge the
        # margin by a coin flip rather than always up, which would bias it.
        odd = (total + margin) % 2 != 0
        margin = margin + odd * self.rng.choice([-1.0, 1.0], size=self.n_sims)
        home = np.clip((total + margin) / 2.0, 0.0, None)
        away = np.clip(total - home, 0.0, None)
        return ScoreDistribution(home=home, away=away)
