"""Monte Carlo score-simulation probabilities."""

from __future__ import annotations

from dataclasses import replace

from cfb_engine.config import ModelParams
from cfb_engine.models.montecarlo import ExpectedGame, MonteCarlo


def _mc() -> MonteCarlo:
    return MonteCarlo(replace(ModelParams(), n_sims=20000), seed=3)


def test_home_favorite_wins_more_than_half():
    exp = ExpectedGame(exp_margin=7.0, exp_total=52.0, margin_sd=16.0, total_sd=13.0)
    sim = _mc().simulate(exp)
    assert sim.home_win_prob() > 0.6
    assert abs(sim.exp_margin - 7.0) < 1.0


def test_pickem_is_near_fifty():
    exp = ExpectedGame(exp_margin=0.0, exp_total=50.0, margin_sd=16.0, total_sd=13.0)
    sim = _mc().simulate(exp)
    assert 0.45 < sim.home_win_prob() < 0.55


def test_cover_and_over_probabilities_are_complementary():
    exp = ExpectedGame(exp_margin=3.0, exp_total=48.0, margin_sd=16.0, total_sd=13.0)
    sim = _mc().simulate(exp)
    # A home favorite laying 3 (home_point = -3) should cover under half the time.
    assert sim.cover_prob(-3.0) < 0.55
    over = sim.over_prob(48.0)
    assert 0.45 < over < 0.55
