"""The transition table has to be a distribution, and it has to conserve runners."""

from __future__ import annotations

import numpy as np
import pytest

from mlb_engine.models import baserunning as B
from mlb_engine.models.markov_f5 import _inning_distribution, team_f5_distribution
from mlb_engine.models.montecarlo import OUTCOMES

STATES = [(base, outs) for base in range(8) for outs in range(3)]


@pytest.mark.parametrize("base,outs", STATES)
def test_branches_are_a_distribution(base: int, outs: int) -> None:
    for oc in OUTCOMES:
        branches = B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE)
        assert branches, f"{base:03b} {outs} {oc} has no branches"
        total = sum(p for *_, p in branches)
        assert total == pytest.approx(1.0), f"{base:03b} {outs} {oc} sums to {total}"
        assert all(p > 0 for *_, p in branches)


@pytest.mark.parametrize("base,outs", STATES)
def test_every_runner_is_accounted_for(base: int, outs: int) -> None:
    """A man on base either scores, holds a base, or is retired -- never vanishes."""
    before = bin(base).count("1") + 1  # runners plus the batter
    for oc in OUTCOMES:
        for new_base, new_outs, runs, _ in B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE):
            after = bin(new_base).count("1") + runs + (new_outs - outs)
            assert after == before, f"{base:03b} {outs} {oc} -> {new_base:03b} {new_outs} {runs}"
            assert 0 <= new_base < 8
            assert outs <= new_outs <= 3


def test_a_strikeout_freezes_the_runners() -> None:
    for base in range(8):
        for outs in range(3):
            assert B.advance_dist(base, outs, "K", B.LEAGUE_DP_RATE) == (
                (base, outs + 1, 0, 1.0),
            )


def test_a_walk_advances_only_what_it_forces() -> None:
    dp = B.LEAGUE_DP_RATE
    assert B.advance_dist(0b010, 0, "BB", dp) == ((0b011, 0, 0, 1.0),)  # 2nd holds
    assert B.advance_dist(0b100, 1, "BB", dp) == ((0b101, 1, 0, 1.0),)  # 3rd holds
    assert B.advance_dist(0b001, 0, "BB", dp) == ((0b011, 0, 0, 1.0),)  # 1st forced
    assert B.advance_dist(0b101, 0, "BB", dp) == ((0b111, 0, 0, 1.0),)  # 3rd still holds
    assert B.advance_dist(0b111, 2, "BB", dp) == ((0b111, 2, 1, 1.0),)  # forced in


def test_a_home_run_clears_the_bases() -> None:
    assert B.advance_dist(0b111, 1, "HR", B.LEAGUE_DP_RATE) == ((0, 1, 4, 1.0),)
    assert B.advance_dist(0, 0, "HR", B.LEAGUE_DP_RATE) == ((0, 0, 1, 1.0),)


def test_the_double_play_needs_a_force_and_fewer_than_two_outs() -> None:
    def p_two_outs(base: int, outs: int) -> float:
        return sum(
            p for _, no, _, p in B.advance_dist(base, outs, "OUT", 0.25) if no - outs >= 2
        )

    assert p_two_outs(0b001, 0) == pytest.approx(0.25)
    assert p_two_outs(0b001, 1) == pytest.approx(0.25)
    assert p_two_outs(0b001, 2) == 0.0  # the inning ends on the first out
    assert p_two_outs(0b110, 0) == 0.0  # nobody to force
    assert p_two_outs(0b000, 0) == 0.0


def test_a_man_on_third_scores_on_an_out_but_not_with_two_down() -> None:
    def p_run(base: int, outs: int) -> float:
        return sum(p for _, _, r, p in B.advance_dist(base, outs, "OUT", B.LEAGUE_DP_RATE) if r)

    assert p_run(0b100, 0) == pytest.approx(B.SCORE_FROM_3B_ON_OUT[0])
    assert p_run(0b100, 1) == pytest.approx(B.SCORE_FROM_3B_ON_OUT[1])
    assert p_run(0b100, 2) == 0.0
    assert p_run(0b010, 0) == 0.0  # no sacrifice fly from second


def test_runners_go_on_contact_with_two_outs() -> None:
    """The out count has to change the advancement, or the pooled rate is wrong twice."""

    def p_run(oc: str, base: int, outs: int) -> float:
        return sum(p for _, _, r, p in B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE) if r)

    assert p_run("1B", 0b010, 0) < p_run("1B", 0b010, 1) < p_run("1B", 0b010, 2)
    assert p_run("2B", 0b001, 0) < p_run("2B", 0b001, 1) < p_run("2B", 0b001, 2)


def test_a_runner_never_passes_the_man_ahead_of_him() -> None:
    """First to third off first-and-second is only legal once second has scored."""
    branches = B.advance_dist(0b011, 0, "1B", B.LEAGUE_DP_RATE)
    assert {b[0] for b in branches} == {0b101, 0b011, 0b111}
    for new_base, _, runs, _ in branches:
        assert new_base & 1, "the batter is always on first after a single"
        if new_base == 0b101:
            assert runs == 1, "the man on second has to have scored to vacate the base"
        if new_base == 0b111:
            assert runs == 0, "nobody scored, so second is blocked and first stops there"


def test_a_double_play_retires_the_man_forced_at_second() -> None:
    """Identity recovery has to erase the trail runner, not the man on third."""
    branches = B.advance_dist(0b101, 0, "OUT", 0.5)
    dp = [b for b in branches if b[1] >= 2]
    assert dp, "a man on first with a force should sometimes be doubled up"
    for new_base, _, runs, _ in dp:
        bases, scored = B.assign([30, 10, 99], new_base, runs)  # 3rd, 1st, batter
        assert 10 not in bases, "the man forced at second is out, not on a base"
        assert 99 not in bases, "the batter is out on a double play"
        assert 10 not in scored and 99 not in scored


@pytest.mark.parametrize("base,outs", STATES)
def test_sampling_recovers_identity_for_every_branch(base: int, outs: int) -> None:
    """Whoever is on base after the play is determined by the state, since nobody passes."""
    runners = [30, 20, 10]  # third, second, first
    occupancy = (base >> 2 & 1, base >> 1 & 1, base & 1)
    order = [r for r, on in zip(runners, occupancy, strict=True) if on]
    for oc in OUTCOMES:
        for new_base, _, runs, _ in B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE):
            bases, scored = B.assign([*order, 99], new_base, runs)
            on_base = [r for r in bases if r >= 0]
            assert len(on_base) == bin(new_base).count("1")
            assert len(set(on_base) | set(scored)) == len(on_base) + len(scored)
            assert set(on_base) | set(scored) <= {*order, 99}


def test_sample_reproduces_the_distribution() -> None:
    rng = np.random.default_rng(0)
    branches = B.advance_dist(0b011, 1, "1B", B.LEAGUE_DP_RATE)
    counts = {b[:3]: 0 for b in branches}
    n = 40_000
    for _ in range(n):
        drawn = B.sample(0b011, 1, "1B", B.LEAGUE_DP_RATE, float(rng.random()))
        counts[drawn[:3]] += 1
    for base, outs, runs, prob in branches:
        assert counts[(base, outs, runs)] / n == pytest.approx(prob, abs=0.01)


def test_the_double_play_rate_follows_ground_ball_rate() -> None:
    assert B.dp_rate_from_gb(0.30) < B.dp_rate_from_gb(0.45) < B.dp_rate_from_gb(0.60)
    # A league-average ground-ball rate has to reproduce the league DP rate.
    assert B.dp_rate_from_gb(0.421) == pytest.approx(B.LEAGUE_DP_RATE, abs=0.01)


def test_double_plays_and_advancement_cost_runs() -> None:
    """The chain scores less than it did, which is the point of the change."""
    rates = {"1B": 0.142, "2B": 0.041, "3B": 0.004, "HR": 0.035, "BB": 0.097, "K": 0.222}
    rates["OUT"] = 1.0 - sum(rates.values())
    mean = lambda dist: float((dist * np.arange(len(dist))).sum())  # noqa: E731
    with_dp = mean(_inning_distribution(rates, B.LEAGUE_DP_RATE))
    without = mean(_inning_distribution(rates, 0.0))
    assert without > with_dp
    assert 0.40 < with_dp < 0.52  # a plausible half-inning


def test_the_f5_chain_and_the_run_environment_stay_sane() -> None:
    rates = {"1B": 0.142, "2B": 0.041, "3B": 0.004, "HR": 0.035, "BB": 0.097, "K": 0.222}
    rates["OUT"] = 1.0 - sum(rates.values())
    dist = team_f5_distribution([rates] * 9, tto_factors=(1.0, 1.0, 1.0, 1.0))
    assert dist.sum() == pytest.approx(1.0)
    assert 1.9 < float((dist * np.arange(len(dist))).sum()) < 2.7
