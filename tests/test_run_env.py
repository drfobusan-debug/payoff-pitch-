"""The run-environment correction, and the two simulator constants it solves from.

``BASELINE_TOTAL`` and ``RUNS_PER_SCALE`` are properties of the simulator, not of
the league, so a change to ``LEAGUE_RATES`` or to the run-scoring mechanics has to
move them. Re-measuring them here is the point: the correction is only a league
measurement while they are right.
"""

from __future__ import annotations

from mlb_engine.features.removal import RemovalHazard
from mlb_engine.features.rolling import LEAGUE_RATES
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.run_env import (
    BASELINE_TOTAL,
    MIN_OUT_SHARE,
    NON_OUT,
    RUNS_PER_SCALE,
    SCALE_CLAMP,
    scale_all,
    scale_for_total,
    scale_rates,
)

HANDS = ("L", "R", "L", "R", "L", "R", "L", "R", "R")


def league_team(scale: float) -> TeamSimConfig:
    rates = scale_rates(LEAGUE_RATES, scale)
    return TeamSimConfig(
        bat_vs_starter=[dict(rates) for _ in range(9)],
        bat_vs_pen=[dict(rates) for _ in range(9)],
        bat_vs_pen_close=[dict(rates) for _ in range(9)],
        bat_vs_pen_bridge=[dict(rates) for _ in range(9)],
        bat_hands=HANDS,
        starter_hand="R",
        removal_hazard=RemovalHazard(),
    )


def sim_total(scale: float, sims: int = 3000) -> float:
    res = MonteCarlo(sims, seed=7).simulate(league_team(scale), league_team(scale))
    return float((res.home_runs_full + res.away_runs_full).mean())


def test_scale_leaves_the_pa_summing_to_one_and_keeps_the_strikeout_share() -> None:
    scaled = scale_rates(LEAGUE_RATES, 0.98)
    assert abs(sum(scaled.values()) - 1.0) < 1e-12
    assert scaled["K"] == LEAGUE_RATES["K"]
    for key in NON_OUT:
        assert scaled[key] < LEAGUE_RATES[key]
    assert scaled["OUT"] > LEAGUE_RATES["OUT"]


def test_unit_scale_and_degenerate_rates_are_left_alone() -> None:
    assert scale_rates(LEAGUE_RATES, 1.0) == dict(LEAGUE_RATES)
    # No room for the residual: the input was not what the correction assumes.
    packed = {"1B": 0.5, "HR": 0.4, "K": 0.05, "OUT": 1.0 - MIN_OUT_SHARE}
    assert scale_rates(packed, 1.03) == packed
    assert scale_all([LEAGUE_RATES, LEAGUE_RATES], 1.0) == [dict(LEAGUE_RATES)] * 2


def test_scale_for_total_is_a_league_measurement_and_is_clamped() -> None:
    assert scale_for_total(BASELINE_TOTAL) == 1.0
    assert scale_for_total(BASELINE_TOTAL - RUNS_PER_SCALE * 0.01) < 1.0
    assert scale_for_total(BASELINE_TOTAL + RUNS_PER_SCALE * 0.01) > 1.0
    assert scale_for_total(2.0) == SCALE_CLAMP[0]
    assert scale_for_total(20.0) == SCALE_CLAMP[1]


def test_baseline_total_still_describes_the_simulator() -> None:
    assert abs(sim_total(1.0) - BASELINE_TOTAL) < 0.30


def test_runs_per_scale_still_describes_the_simulator() -> None:
    lo, hi = 0.96, 1.04
    measured = (sim_total(hi) - sim_total(lo)) / (hi - lo)
    assert abs(measured - RUNS_PER_SCALE) < 3.0


def test_the_solved_scale_lands_on_the_target() -> None:
    target = BASELINE_TOTAL - 0.4
    assert abs(sim_total(scale_for_total(target)) - target) < 0.30
