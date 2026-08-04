"""Drive-based Markov simulator: mean-anchoring and market direction."""

from __future__ import annotations

from dataclasses import replace

from cfb_engine.config import ModelParams
from cfb_engine.models.markov import DriveShape, MarkovSim, _expected_points, _solve_conversion
from cfb_engine.models.montecarlo import ExpectedGame


def _sim() -> MarkovSim:
    return MarkovSim(replace(ModelParams(), n_sims=40000), seed=5)


def test_conversion_solver_matches_target_scoring_rate():
    for target in (0.5, 1.5, 2.5, 3.5):
        c = _solve_conversion(target)
        assert abs(_expected_points(c) - target) < 0.05


def test_markov_is_anchored_to_the_same_means():
    exp = ExpectedGame(exp_margin=7.0, exp_total=52.0, margin_sd=16.0, total_sd=13.0)
    sim = _sim().simulate(exp, DriveShape(home_drives=12, away_drives=12))
    # Shares the normal engine's means (the whole point of a clean A/B).
    assert abs(sim.exp_total - 52.0) < 2.5
    assert abs(sim.exp_margin - 7.0) < 2.0


def test_markov_home_favorite_wins_more_than_half():
    exp = ExpectedGame(exp_margin=10.0, exp_total=55.0, margin_sd=16.0, total_sd=13.0)
    sim = _sim().simulate(exp)
    assert sim.home_win_prob() > 0.6


def test_markov_pickem_is_near_fifty():
    exp = ExpectedGame(exp_margin=0.0, exp_total=48.0, margin_sd=16.0, total_sd=13.0)
    sim = _sim().simulate(exp)
    assert 0.45 < sim.home_win_prob() < 0.55


def test_faster_pace_widens_the_total_distribution():
    exp = ExpectedGame(exp_margin=0.0, exp_total=56.0, margin_sd=16.0, total_sd=13.0)
    slow = _sim().simulate(exp, DriveShape(home_drives=9, away_drives=9))
    fast = _sim().simulate(exp, DriveShape(home_drives=16, away_drives=16))
    # Same mean total, but more possessions => more scoring events => wider spread.
    assert abs(fast.exp_total - slow.exp_total) < 3.0
    assert fast.exp_total_sd > slow.exp_total_sd
