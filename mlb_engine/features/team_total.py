"""Market implied team total as a run-environment anchor for R and RBI.

Runs and RBIs are the two batter stats that depend least on the hitter and most
on the offense around him, and the market prices that offense directly: the
game total plus the moneyline pin each side's expected runs. The simulator
derived its run environment purely from lineup x opposing staff and never read
the market, so a lineup the book expects to score 5.6 was simulated at the same
level as one it expects to score 3.4 whenever the underlying Statcast inputs
happened to agree.

The implied team total is recovered from the two prices::

    expected margin = Phi^-1(no-vig win prob) * sd(run differential)
    team total       = total / 2 + margin / 2

and the simulated R/RBI distributions are then nudged toward it by a bounded
fraction. This is an anchor, not an override: at the default weight the model
keeps roughly two thirds of its own view, and the move is capped so a stale or
thin market line cannot swing a prop.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from statistics import NormalDist

# SD of a single game's run differential. ~4.3 runs league-wide, which is what
# maps a moneyline probability back onto an expected margin.
RUN_DIFF_SD = 4.3

DEFAULT_WEIGHT = 0.35
DEFAULT_CAP = 0.12


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def implied_team_total(
    game_total: float, win_prob: float, sd: float = RUN_DIFF_SD
) -> float:
    """Market's expected runs for the side whose no-vig win probability is given."""
    margin = NormalDist().inv_cdf(_clip(win_prob, 0.01, 0.99)) * sd
    return game_total / 2.0 + margin / 2.0


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val not in ("0", "false", "False", "")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class TeamTotalAnchor:
    """Config + logic for the market run-environment anchor."""

    enabled: bool = True
    weight: float = DEFAULT_WEIGHT
    cap: float = DEFAULT_CAP
    sd: float = RUN_DIFF_SD

    @classmethod
    def from_env(cls) -> TeamTotalAnchor:
        return cls(
            enabled=_env_flag("MLBE_TEAM_TOTAL_ANCHOR", True),
            weight=_env_float("MLBE_TEAM_TOTAL_W", DEFAULT_WEIGHT),
            cap=_env_float("MLBE_TEAM_TOTAL_CAP", DEFAULT_CAP),
            sd=_env_float("MLBE_TEAM_TOTAL_SD", RUN_DIFF_SD),
        )

    def factor(
        self,
        game_total: float | None,
        win_prob: float | None,
        model_runs: float | None,
    ) -> float:
        """Bounded multiplier pulling simulated R/RBI toward the market's total.

        Returns 1.0 (inert) when disabled or when either price or the model's own
        run expectation is unavailable, so an unpriced game is untouched.
        """
        if not self.enabled or self.weight == 0.0:
            return 1.0
        if game_total is None or win_prob is None or model_runs is None:
            return 1.0
        if model_runs <= 0 or math.isnan(model_runs) or math.isnan(game_total):
            return 1.0
        implied = implied_team_total(game_total, win_prob, self.sd)
        if implied <= 0:
            return 1.0
        gap = implied / model_runs - 1.0
        return 1.0 + _clip(gap * self.weight, -self.cap, self.cap)
