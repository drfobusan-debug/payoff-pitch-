"""Prop distributions on top of the game sim (master plan §5.5).

Every prop reads the *same* joint draws that priced the game markets:

- **score state**: the share of draws leading / tied / trailing at the start
  of P2 and P3 (P1 opens tied) scales a skater's shot rate by the league
  score-effect multipliers -- a team the sim expects to lead shoots less;
- **opponent**: shots-against factor (opp SA/60 over league) for SOG and
  blocks, the sim's goalie factor for goals;
- **saves**: the goalie's expected shots faced come from team shot rates, his
  expected goals against from the sim; saves = shots - goals.

Distributions: SOG and saves negative binomial with variance ``v x mean`` (SOG's
``v`` fitted in ``scripts/nhl/props_study.py``); goals / assists / points / PPP / blocks
Poisson; anytime goal = 1 - P(0 goals). Lines on a half-integer never push;
whole-number lines return the push mass separately, like every other market.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nhl_engine.models.markets import Prob
from nhl_engine.models.periods import SimResult

# Score-effect multipliers on a team's shot rate (Corsi-tied ≈ 50%, leading ≈ 47%,
# trailing ≈ 53% of attempts in public score-effects work); study candidates.
LEADING_MULT = 0.92
TRAILING_MULT = 1.08
# Negative-binomial variance slope for SOG (var = v x mean), props_study.py on
# 2024-25 (31,858 skater-games): 1.087 -- SOG are barely over-dispersed.
SOG_VAR_SLOPE = 1.09
# Team shots on goal are much more dispersed than a Poisson (var ≈ 1.6 x mean on
# 2024-25 team-games); saves inherit it. Study candidate.
SAVES_VAR_SLOPE = 1.6
MAX_COUNT = 40


@dataclass(frozen=True)
class ScoreState:
    leading: float
    tied: float
    trailing: float

    @property
    def shot_factor(self) -> float:
        return self.leading * LEADING_MULT + self.tied + self.trailing * TRAILING_MULT


def score_state(sim: SimResult, *, home: bool) -> ScoreState:
    """Minutes-weighted share of regulation spent leading / tied / trailing.

    P1 opens tied; P2 and P3 take the state at their start from each draw, so
    the three periods average to the shares used.
    """
    margin_after = np.cumsum(sim.ph - sim.pa, axis=1)  # after P1, P2, P3
    if not home:
        margin_after = -margin_after
    states = margin_after[:, :2]  # state entering P2 and P3
    lead = float((states > 0).mean()) * 2 / 3
    trail = float((states < 0).mean()) * 2 / 3
    return ScoreState(leading=lead, tied=1.0 - lead - trail, trailing=trail)


def exp_goals_against(sim: SimResult, *, home: bool) -> float:
    """Goals the side concedes per draw, regulation + OT (a shootout goal is not a save)."""
    ga = sim.ot_goals_a if home else sim.ot_goals_h
    return float(ga.mean())


def poisson_pmf(mean: float, n: int = MAX_COUNT) -> np.ndarray:
    k = np.arange(n + 1)
    mean = max(mean, 1e-9)
    log_p = k * math.log(mean) - mean - np.array([math.lgamma(i + 1) for i in k])
    p = np.exp(log_p)
    p[-1] += max(0.0, 1.0 - p.sum())
    return p


def negbin_pmf(mean: float, var_slope: float = SOG_VAR_SLOPE, n: int = MAX_COUNT) -> np.ndarray:
    """NB with variance ``var_slope x mean``; Poisson when the slope is ≤ 1."""
    mean = max(mean, 1e-9)
    if var_slope <= 1.0 + 1e-9:
        return poisson_pmf(mean, n)
    p = 1.0 / var_slope
    r = mean * p / (1.0 - p)
    k = np.arange(n + 1)
    log_pmf = (
        np.array([math.lgamma(i + r) for i in k])
        - math.lgamma(r)
        - np.array([math.lgamma(i + 1) for i in k])
        + r * math.log(p)
        + k * math.log(1.0 - p)
    )
    out = np.exp(log_pmf)
    out[-1] += max(0.0, 1.0 - out.sum())
    return out


def over_under(pmf: np.ndarray, line: float, side: str) -> Prob:
    k = np.arange(len(pmf))
    push = float(pmf[k == line].sum()) if float(line).is_integer() else 0.0
    over = float(pmf[k > line].sum())
    under = float(pmf[k < line].sum())
    win = over if side == "over" else under
    lose = under if side == "over" else over
    denom = win + lose
    return Prob(win=win / denom if denom else 0.0, push=push)


def anytime(pmf: np.ndarray, side: str) -> Prob:
    p_yes = 1.0 - float(pmf[0])
    return Prob(win=p_yes if side == "yes" else 1.0 - p_yes, push=0.0)


def sog_mean(sog60: float, toi: float, *, shot_factor: float, opp_sa_factor: float) -> float:
    return sog60 * toi / 3600.0 * shot_factor * opp_sa_factor


def saves_mean(shots_faced: float, exp_ga: float) -> float:
    return max(shots_faced - exp_ga, 0.5)


__all__ = [
    "LEADING_MULT",
    "SAVES_VAR_SLOPE",
    "SOG_VAR_SLOPE",
    "TRAILING_MULT",
    "ScoreState",
    "anytime",
    "exp_goals_against",
    "negbin_pmf",
    "over_under",
    "poisson_pmf",
    "saves_mean",
    "score_state",
    "sog_mean",
]
