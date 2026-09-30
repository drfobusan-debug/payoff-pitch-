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


def test_market_win_prob_uses_market_implied_sd():
    from cfb_engine.models.montecarlo import market_win_prob, ml_margin_sd

    params = ModelParams()
    # Inside the knee the SD is the flat market base; past it, it widens.
    assert ml_margin_sd(7.0, params) == params.ml_sd_base
    assert ml_margin_sd(-28.0, params) > ml_margin_sd(-14.0, params) > params.ml_sd_base
    # A 7-pt favourite prices near the market's ~70%, not the flat-16 normal's 67%.
    p7 = market_win_prob(7.0, params)
    assert 0.69 < p7 < 0.71
    assert abs(market_win_prob(-7.0, params) - (1 - p7)) < 1e-12
    assert market_win_prob(0.0, params) == 0.5
    # Blowouts stay near the market's cap rather than running to certainty.
    assert 0.94 < market_win_prob(30.0, params) < 0.97


def test_pipeline_ml_reads_market_curve_by_default():
    from cfb_engine.config import Config
    from cfb_engine.models.montecarlo import GameSimResult

    assert Config().model.ml_market_sd is True
    import numpy as np

    sim = GameSimResult(margins=np.full(100, 7.0), totals=np.full(100, 50.0))
    assert sim.home_win_prob() == 1.0  # the raw sim would call this certain
