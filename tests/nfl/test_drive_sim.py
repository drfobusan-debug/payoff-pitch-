"""The possession simulator: discreteness, anchoring, and the key numbers."""

from __future__ import annotations

import numpy as np

from nfl_engine.models.drives import DriveSim, ExpectedGame
from nfl_engine.models.montecarlo import NormalSim


def _even_game() -> ExpectedGame:
    return ExpectedGame(home_points=23.8, away_points=21.9)


def _sim(n: int = 40000) -> DriveSim:
    return DriveSim(n_sims=n, seed=5)


def test_scores_are_whole_numbers():
    dist = _sim().simulate(_even_game())
    assert np.all(dist.home == np.floor(dist.home))
    assert np.all(dist.away == np.floor(dist.away))
    assert dist.home.min() >= 0


def test_means_match_what_was_requested():
    exp = ExpectedGame(home_points=27.0, away_points=20.0)
    dist = _sim().simulate(exp)
    assert abs(dist.mean_total() - 47.0) < 1.0
    assert abs(dist.mean_margin() - 7.0) < 1.0


def test_key_numbers_beat_the_normal_control():
    """The reason the engine is not a Gaussian: 14.8% of games land on 3."""
    exp = _even_game()
    drives = _sim().simulate(exp)
    normal = NormalSim(n_sims=40000, seed=5).simulate(exp)
    assert drives.margin_frequency(3) > 0.11
    assert drives.margin_frequency(7) > 0.055
    assert drives.margin_frequency(3) > 2.0 * normal.margin_frequency(3)


def test_three_is_the_most_common_margin():
    dist = _sim().simulate(_even_game())
    counts = {k: dist.margin_frequency(k) for k in range(1, 15)}
    assert max(counts, key=lambda k: counts[k]) == 3


def test_moneyline_and_spread_agree_on_the_favourite():
    exp = ExpectedGame(home_points=27.5, away_points=20.5)
    dist = _sim().simulate(exp)
    home_ml = dist.moneyline(home=True)
    assert home_ml.win > dist.moneyline(home=False).win
    # Laying the exact expected margin is close to a coin flip once pushes are out.
    assert 0.40 < dist.spread(-7.0).conditional < 0.60
    # Getting a shorter number is worth more than laying a longer one.
    assert dist.spread(-6.5).win > dist.spread(-7.5).win


def test_the_tie_is_the_moneyline_push():
    dist = _sim().simulate(_even_game())
    home = dist.moneyline(home=True)
    away = dist.moneyline(home=False)
    assert home.push == away.push
    assert abs(home.win + away.win + home.push - 1.0) < 1e-9
    # Ties are rare but not impossible: 0.33% of games since 2015.
    assert 0.0 <= home.push < 0.01


def test_spread_push_only_on_whole_numbers():
    dist = _sim().simulate(_even_game())
    assert dist.spread(-2.5).push == 0.0
    assert dist.spread(-3.0).push > 0.05
    # The half-point buy is worth roughly the push it removes.
    assert dist.spread(-2.5).win - dist.spread(-3.0).win > 0.04


def test_total_push_and_direction():
    dist = _sim().simulate(_even_game())
    over = dist.total(45.0, over=True)
    under = dist.total(45.0, over=False)
    assert over.push == under.push
    assert over.push > 0.01
    assert abs(over.win + under.win + over.push - 1.0) < 1e-9
    assert dist.total(38.5, over=True).win > dist.total(52.5, over=True).win


def test_pace_widens_the_total_without_moving_the_margin():
    slow = _sim().simulate(ExpectedGame(home_points=22.0, away_points=22.0, home_drives=9.0, away_drives=9.0))
    fast = _sim().simulate(ExpectedGame(home_points=22.0, away_points=22.0, home_drives=13.0, away_drives=13.0))
    assert fast.totals().std() > slow.totals().std()
    assert abs(fast.mean_margin() - slow.mean_margin()) < 1.0


def test_possession_order_is_not_worth_points():
    """Half the trials give each team the last possession, because the
    final-drive table would otherwise hand the receiving side free equity."""
    dist = _sim().simulate(ExpectedGame(home_points=22.0, away_points=22.0))
    assert abs(dist.mean_margin()) < 0.5
