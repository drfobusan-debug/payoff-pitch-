"""Joint period-by-period game simulation (master plan §5.3).

One draw walks regulation in ``step_seconds`` slices and produces per-period
goals for both sides, then an OT block and a shootout when tied. Every market
(ML, puck line, totals, team totals, each period's ML/PL/total) is read off the
*same* draws in ``models/markets.py``, so the prices are mutually coherent by
construction rather than through a bolted-on correlation parameter.

Per slice, each side scores Poisson(lam x dt) where ``lam`` depends on:

* manpower -- a stochastic minor-penalty process (per-side rates from
  ``SideRates.minors``); a minor puts the opponent on a ``minor_seconds`` power
  play that ends early on a PP goal. Two overlapping minors are 4v4, played at
  5v5 rates; a second minor against the same side extends the PP rather than
  creating 5v3 (both approximations are registered candidates, not measured).
* score state -- 5v5 rates carry the study's multipliers by the shooter's
  state (trail 2+/1, tied, lead 1/2+) and the period's share of tied-state xG.
* empty net -- in the third period a trailing side pulls its goalie once the
  clock is inside a pull time drawn per game from the study's time-left
  quantiles for that deficit (down 3 pulls with the measured 0.29 probability);
  while pulled the attacker scores at the 6v5 rate and the opponent at the
  empty-net rate. The goalie returns when the deficit closes.

Tied after 60: the OT block is *not* 5v5 x k. The game is decided in OT with
``p_ot_goal``; the winner follows ``sigmoid(a + b x (home xG share - away xG
share))`` from the study, tilted by the two goalie factors (a better goalie
faces the same 3v3 chances and stops more of them). Otherwise the shootout is a
coin at ``so_home`` (measured 0.512 +/- 0.031: not distinguishable from 0.5).
Goalie pulls in OT are not modelled: the PBP shows no meaningful rate.

Output is arrays over draws; no probabilities are computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nhl_engine.config import SimParams
from nhl_engine.models.goals import GameRates

PERIOD = 1200
REG = 3 * PERIOD
QUANTILE_PROBS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


@dataclass(frozen=True)
class SimResult:
    """Per-draw outcomes. ``ph``/``pa`` are ``(n, 3)`` regulation goals per period."""

    ph: np.ndarray
    pa: np.ndarray
    ot_h: np.ndarray  # 1 where home scored the OT winner
    ot_a: np.ndarray
    so_h: np.ndarray  # 1 where home won the shootout
    so_a: np.ndarray

    @property
    def n(self) -> int:
        return int(self.ph.shape[0])

    @property
    def reg_h(self) -> np.ndarray:
        return self.ph.sum(axis=1)

    @property
    def reg_a(self) -> np.ndarray:
        return self.pa.sum(axis=1)

    @property
    def ot_goals_h(self) -> np.ndarray:
        return self.reg_h + self.ot_h

    @property
    def ot_goals_a(self) -> np.ndarray:
        return self.reg_a + self.ot_a

    @property
    def final_h(self) -> np.ndarray:
        """Official final: the shootout winner is credited one goal."""
        return self.reg_h + self.ot_h + self.so_h

    @property
    def final_a(self) -> np.ndarray:
        return self.reg_a + self.ot_a + self.so_a

    @property
    def home_win(self) -> np.ndarray:
        return self.final_h > self.final_a

    @property
    def went_ot(self) -> np.ndarray:
        return self.reg_h == self.reg_a

    @property
    def went_so(self) -> np.ndarray:
        return (self.so_h + self.so_a) > 0


def _state_mult(diff: np.ndarray, params: SimParams) -> np.ndarray:
    """Score-state multiplier for a side whose goal differential is ``diff``."""
    m = params.score_mult
    out = np.full(diff.shape, m["tied"])
    out[diff <= -2] = m["trail2"]
    out[diff == -1] = m["trail1"]
    out[diff == 1] = m["lead1"]
    out[diff >= 2] = m["lead2"]
    return out


def _pull_times(rng: np.random.Generator, n: int, params: SimParams) -> dict[int, np.ndarray]:
    """Time left (s) at which a side down ``d`` pulls; ``-1`` where it never pulls."""
    out: dict[int, np.ndarray] = {}
    for d, qs in params.pull_quantiles.items():
        u = rng.uniform(0.0, 1.0, n)
        t = np.interp(u, QUANTILE_PROBS, np.asarray(qs, dtype=float))
        pulls = rng.uniform(0.0, 1.0, n) < params.pull_prob.get(d, 0.0)
        out[d] = np.where(pulls, t, -1.0)
    return out


def _pulled(deficit: np.ndarray, time_left: int, pulls: dict[int, np.ndarray]) -> np.ndarray:
    out = np.zeros(deficit.shape, dtype=bool)
    for d, t in pulls.items():
        sel = (deficit == d) if d < 3 else (deficit >= 3)
        out |= sel & (time_left <= t)
    return out


def simulate(
    rates: GameRates,
    params: SimParams,
    *,
    draws: int | None = None,
    seed: int | None = None,
) -> SimResult:
    n = draws or params.draws
    rng = np.random.default_rng(seed)
    dt = params.step_seconds
    h, a = rates.home, rates.away

    ph = np.zeros((n, 3), dtype=np.int64)
    pa = np.zeros((n, 3), dtype=np.int64)
    pp_h: np.ndarray = np.zeros(n, dtype=np.int64)  # seconds of home power play remaining
    pp_a: np.ndarray = np.zeros(n, dtype=np.int64)
    pull_h = _pull_times(rng, n, params)
    pull_a = _pull_times(rng, n, params)

    for p in range(3):
        pmult = params.period_mult[p]
        for t in range(0, PERIOD, dt):
            hg = ph.sum(axis=1)
            ag = pa.sum(axis=1)
            diff = hg - ag
            home_pp = (pp_h > 0) & (pp_a == 0)
            away_pp = (pp_a > 0) & (pp_h == 0)
            even = ~(home_pp | away_pp)

            lam_h = np.where(
                home_pp, h.pp, np.where(away_pp, h.sh, h.g5 * _state_mult(diff, params) * pmult)
            )
            lam_a = np.where(
                away_pp, a.pp, np.where(home_pp, a.sh, a.g5 * _state_mult(-diff, params) * pmult)
            )

            if p == 2:
                left = PERIOD - t
                h_pulled = _pulled(-diff, left, pull_h) & even
                a_pulled = _pulled(diff, left, pull_a) & even
                lam_h = np.where(h_pulled, h.en_for, np.where(a_pulled, a.en_against, lam_h))
                lam_a = np.where(a_pulled, a.en_for, np.where(h_pulled, h.en_against, lam_a))

            gh = rng.poisson(lam_h * dt)
            ga = rng.poisson(lam_a * dt)
            ph[:, p] += gh
            pa[:, p] += ga

            # a PP goal ends the power play
            pp_h = np.where(home_pp & (gh > 0), 0, pp_h)
            pp_a = np.where(away_pp & (ga > 0), 0, pp_a)
            pp_h = np.maximum(pp_h - dt, 0)
            pp_a = np.maximum(pp_a - dt, 0)

            # new minors: taken by home -> away power play
            took_h = rng.uniform(0.0, 1.0, n) < h.minors * dt
            took_a = rng.uniform(0.0, 1.0, n) < a.minors * dt
            pp_a = np.where(took_h, np.maximum(pp_a, params.minor_seconds), pp_a)
            pp_h = np.where(took_a, np.maximum(pp_h, params.minor_seconds), pp_h)

    tied = ph.sum(axis=1) == pa.sum(axis=1)
    ot_h = np.zeros(n, dtype=np.int64)
    ot_a = np.zeros(n, dtype=np.int64)
    so_h = np.zeros(n, dtype=np.int64)
    so_a = np.zeros(n, dtype=np.int64)
    if tied.any():
        p_home_ot = ot_home_prob(rates, params)
        decided = tied & (rng.uniform(0.0, 1.0, n) < params.p_ot_goal)
        home_wins_ot = rng.uniform(0.0, 1.0, n) < p_home_ot
        ot_h[decided & home_wins_ot] = 1
        ot_a[decided & ~home_wins_ot] = 1
        shootout = tied & ~decided
        home_wins_so = rng.uniform(0.0, 1.0, n) < params.so_home
        so_h[shootout & home_wins_so] = 1
        so_a[shootout & ~home_wins_so] = 1
    return SimResult(ph=ph, pa=pa, ot_h=ot_h, ot_a=ot_a, so_h=so_h, so_a=so_a)


def ot_home_prob(rates: GameRates, params: SimParams) -> float:
    """P(home scores the OT winner | decided in OT): study link + goalie tilt."""
    gap = rates.home.xg_share - rates.away.xg_share
    z = params.ot_home_intercept + params.ot_strength_slope * gap
    odds = float(np.exp(z))
    # home scores against the away goalie and vice versa
    odds *= rates.away.goalie_factor / max(rates.home.goalie_factor, 1e-6)
    return odds / (1.0 + odds)


__all__ = ["PERIOD", "REG", "SimResult", "ot_home_prob", "simulate"]
