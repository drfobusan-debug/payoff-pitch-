"""Drive-based Markov score simulator (an alternative to the normal MC).

The normal engine draws (margin, total) from a fixed-SD bivariate normal. This
engine instead simulates the game **possession by possession**, so the score
distribution -- especially its spread and tails -- emerges from tempo and
per-drive scoring structure rather than a hand-set standard deviation.

Each drive is a small absorbing Markov chain over "series" (first-down attempts)
starting at roughly the own 25:

    backed-up --c--> ... --c--> red-zone --c--> TOUCHDOWN
         |               |                |
       (1-c) stall     (1-c) FG try     (1-c) FG try

The offense converts each series with probability ``c``; reaching the end zone
(``K`` conversions) is a touchdown, stalling in field-goal range (2-3
conversions) is a field goal, and stalling earlier is no score. ``c`` is the
single free parameter, solved per team so the chain's expected points per drive
equals the ratings-implied scoring rate. Crucially the engine is anchored to the
**same means** as the normal MC (it splits ``exp_total``/``exp_margin`` into each
team's expected points), so a backtest isolates the effect of the distribution
*shape*, not a different forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfb_engine.config import ModelParams
from cfb_engine.models.montecarlo import ExpectedGame, GameSimResult

PTS_TD = 7.0  # touchdown + assumed extra point
PTS_FG = 3.0
_K = 4  # series conversions needed to reach the end zone
_DEFAULT_DRIVES = 12.0
_MIN_DRIVES, _MAX_DRIVES = 8, 17


@dataclass(frozen=True)
class DriveShape:
    """Per-team pace that shapes each side's possession count."""

    home_drives: float = _DEFAULT_DRIVES
    away_drives: float = _DEFAULT_DRIVES


def _expected_points(c: float) -> float:
    """Expected points for one drive given per-series conversion prob ``c``."""
    p_td = c**_K
    p_fg = (1.0 - c) * (c**2 + c**3)  # stall at 2 or 3 conversions (FG range)
    return PTS_TD * p_td + PTS_FG * p_fg


def _solve_conversion(target_ppd: float) -> float:
    """Bisection-solve the conversion prob whose drive EV matches ``target_ppd``."""
    target = max(0.0, min(target_ppd, _expected_points(0.985)))
    lo, hi = 0.0, 0.985
    for _ in range(40):
        mid = (lo + hi) / 2
        if _expected_points(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class MarkovSim:
    def __init__(self, params: ModelParams, *, seed: int = 7) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def _team_points(self, mean_pts: float, drives: float) -> np.ndarray:
        n = self.params.n_sims
        d = int(round(max(_MIN_DRIVES, min(_MAX_DRIVES, drives or _DEFAULT_DRIVES))))
        ppd = max(0.0, mean_pts) / d
        c = _solve_conversion(ppd)
        if c <= 0.0:
            return np.zeros(n)
        # Successful series before the first stall, capped at the end zone.
        succ = self.rng.geometric(1.0 - c, size=(n, d)) - 1
        np.clip(succ, 0, _K, out=succ)
        td = succ >= _K
        fg = (succ >= 2) & (succ < _K)
        pts = PTS_TD * td.sum(axis=1) + PTS_FG * fg.sum(axis=1)
        return pts.astype(float)

    def simulate(self, exp: ExpectedGame, shape: DriveShape | None = None) -> GameSimResult:
        shape = shape or DriveShape()
        home_mean = (exp.exp_total + exp.exp_margin) / 2.0
        away_mean = (exp.exp_total - exp.exp_margin) / 2.0
        home_pts = self._team_points(home_mean, shape.home_drives)
        away_pts = self._team_points(away_mean, shape.away_drives)
        return GameSimResult(margins=home_pts - away_pts, totals=home_pts + away_pts)
