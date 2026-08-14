"""The conversion of hits into runs, which prices R, RBI, H+R+RBI and both totals."""

from __future__ import annotations

import numpy as np
import pytest

from mlb_engine.models import baserunning as B
from mlb_engine.models.markov_f5 import _inning_distribution
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig

LEAGUE = {
    "1B": 0.1422,
    "2B": 0.0411,
    "3B": 0.0038,
    "HR": 0.0338,
    "BB": 0.0967,
    "K": 0.2221,
    "OUT": 0.4603,
}


def _sim(n: int = 2000, seed: int = 3, rates: dict[str, float] | None = None):
    r = dict(LEAGUE if rates is None else rates)
    cfg = TeamSimConfig(
        bat_vs_starter=[dict(r) for _ in range(9)],
        bat_vs_pen=[dict(r) for _ in range(9)],
        # The league's own double-play rate, since these tests ask whether the sim
        # reproduces the league. 0.10 stood in for the old 12% anchor.
        gb_dp_rate=B.LEAGUE_DP_RATE,
    )
    return MonteCarlo(n, seed=seed).simulate(cfg, cfg)


def test_runs_and_rbi_are_not_identical() -> None:
    """A run can arrive without a batter driving it in: steal, wild pitch, error.

    The old model credited an RBI to someone for literally every run scored,
    which no league has ever done.
    """
    res = _sim()
    runs = res.bat["away"]["R"].sum()
    rbi = res.bat["away"]["RBI"].sum()
    assert 0.90 < rbi / runs < 0.99


def test_league_scoring_is_reproduced() -> None:
    """Fed the league's own plate-appearance rates, the sim scores like the league.

    Measured over 760 games (2026-06-01..07-27): 9.18 runs per game between the
    two teams. The sim sits a little under, since it models no error-assisted
    reaching, only error-assisted advancement.
    """
    res = _sim(n=3000)
    total = (res.home_runs_full + res.away_runs_full).mean()
    assert 8.3 < total < 9.4


def test_lineup_order_drives_runs_and_rbi() -> None:
    """The slot a hitter occupies decides his run and RBI chances, not just his bat.

    Nine identical hitters still produce a gradient: the top of the order comes to
    the plate more often and bats behind the weakest hitters.
    """
    res = _sim(n=3000)
    bat = res.bat["away"]
    r_by_slot = bat["R"].mean(axis=0)
    assert r_by_slot[0] > r_by_slot[8]
    rbi_by_slot = bat["RBI"].mean(axis=0)
    assert rbi_by_slot[3] > rbi_by_slot[8]


def test_an_out_can_drive_in_a_run() -> None:
    """The sac fly exists: about a tenth of real RBI come on outs.

    A lineup that never homers still has to produce RBI on outs, so with hits and
    outs only the RBI total must exceed what the hits alone could drive in.
    """
    rates = dict(LEAGUE)
    rates["OUT"] += rates["HR"]
    rates["HR"] = 0.0
    res = _sim(n=2000, rates=rates)
    bat = res.bat["away"]
    assert bat["RBI"].sum() > 0
    assert np.all(bat["HR"] == 0)


def test_a_single_does_not_always_score_the_runner_from_second() -> None:
    """Certainty on the bases is what inflated every run-scoring prop.

    With singles as the only hit, runs must fall short of the number the old
    always-scores rule produced -- the check is that the run distribution has
    spread rather than tracking hits one for one.
    """
    rates = {"1B": 0.30, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.35, "OUT": 0.35}
    res = _sim(n=2000, rates=rates)
    bat = res.bat["away"]
    hits = bat["H"].sum(axis=1)
    runs = bat["R"].sum(axis=1)
    assert runs.mean() < hits.mean() * 0.75


# --- the shared table ------------------------------------------------------
# Below the sim: the transition distribution itself, which the F5 Markov chain
# enumerates exactly and the sim samples. Both models read it, so a violation
# here is a violation in both.

STATES = [(base, outs) for base in range(8) for outs in range(3)]


@pytest.mark.parametrize(("base", "outs"), STATES)
def test_branches_are_a_distribution(base: int, outs: int) -> None:
    for oc in B.OUTCOMES:
        for name, branches in (
            ("advance", B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE)),
            ("transitions", B.transitions(base, outs, oc, B.LEAGUE_DP_RATE)),
        ):
            assert branches, f"{name} {base:03b} {outs} {oc} has no branches"
            total = sum(p for p, *_ in branches)
            assert total == pytest.approx(1.0), f"{name} {base:03b} {outs} {oc} sums to {total}"
            assert all(p > 0 for p, *_ in branches)


@pytest.mark.parametrize(("base", "outs"), STATES)
def test_every_runner_is_accounted_for(base: int, outs: int) -> None:
    """A man on base scores, holds a base, or is retired -- he never vanishes."""
    before = bin(base).count("1") + 1  # the runners plus the batter
    for oc in B.OUTCOMES:
        for _, new_base, new_outs, runs in B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE):
            after = bin(new_base).count("1") + runs + (new_outs - outs)
            assert after == before, f"{base:03b} {outs} {oc} -> {new_base:03b} {new_outs} {runs}"
            assert 0 <= new_base < 8
            assert outs <= new_outs <= 3


def test_a_strikeout_freezes_the_runners() -> None:
    for base in range(8):
        for outs in range(3):
            assert B.advance_dist(base, outs, "K", B.LEAGUE_DP_RATE) == ((1.0, base, outs + 1, 0),)


def test_a_walk_advances_only_what_it_forces() -> None:
    dp = B.LEAGUE_DP_RATE
    assert B.advance_dist(0b010, 0, "BB", dp) == ((1.0, 0b011, 0, 0),)  # second holds
    assert B.advance_dist(0b100, 1, "BB", dp) == ((1.0, 0b101, 1, 0),)  # third holds
    assert B.advance_dist(0b001, 0, "BB", dp) == ((1.0, 0b011, 0, 0),)  # first forced
    assert B.advance_dist(0b101, 0, "BB", dp) == ((1.0, 0b111, 0, 0),)  # third still holds
    assert B.advance_dist(0b111, 2, "BB", dp) == ((1.0, 0b111, 2, 1),)  # forced in


def test_the_double_play_needs_a_force_and_an_out_to_spare() -> None:
    def p_two_outs(base: int, outs: int) -> float:
        return sum(
            p for p, _, no, _ in B.advance_dist(base, outs, "OUT", 0.25) if no - outs >= 2
        )

    assert p_two_outs(0b001, 0) == pytest.approx(0.25)
    assert p_two_outs(0b001, 1) == pytest.approx(0.25)
    assert p_two_outs(0b001, 2) == 0.0  # the inning ends on the first out
    assert p_two_outs(0b110, 0) == 0.0  # nobody to force
    assert p_two_outs(0b000, 0) == 0.0


def test_a_man_on_third_scores_on_an_out_but_not_with_two_down() -> None:
    def p_run(base: int, outs: int) -> float:
        return sum(p for p, _, _, r in B.advance_dist(base, outs, "OUT", B.LEAGUE_DP_RATE) if r)

    assert p_run(0b100, 0) == pytest.approx(B.SCORE_FROM_3B_ON_OUT[0])
    assert p_run(0b100, 1) == pytest.approx(B.SCORE_FROM_3B_ON_OUT[1])
    assert p_run(0b100, 2) == 0.0
    assert p_run(0b010, 0) == 0.0  # there is no sacrifice fly from second


def test_runners_go_on_contact_with_two_outs() -> None:
    """The out count has to move the rate, or one pooled number is wrong twice."""

    def p_run(oc: str, base: int, outs: int) -> float:
        return sum(p for p, _, _, r in B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE) if r)

    assert p_run("1B", 0b010, 0) < p_run("1B", 0b010, 1) < p_run("1B", 0b010, 2)
    assert p_run("2B", 0b001, 0) < p_run("2B", 0b001, 1) < p_run("2B", 0b001, 2)


def test_a_runner_never_passes_the_man_ahead_of_him() -> None:
    """First to third off first-and-second is only legal once second has scored."""
    branches = B.advance_dist(0b011, 0, "1B", B.LEAGUE_DP_RATE)
    assert {b[1] for b in branches} == {0b101, 0b011, 0b111}
    for _, new_base, _, runs in branches:
        assert new_base & 1, "the batter is always on first after a single"
        if new_base == 0b101:
            assert runs == 1, "the man on second has to have scored to vacate it"
        if new_base == 0b111:
            assert runs == 0, "nobody scored, so second is blocked and first stops there"


def test_a_double_play_retires_the_man_forced_at_second() -> None:
    """Identity has to erase the trail runner, not the man standing on third."""
    dp = [b for b in B.advance_dist(0b101, 0, "OUT", 0.5) if b[2] >= 2]
    assert dp, "a man on first with a force should sometimes be doubled up"
    for _, new_base, _, runs in dp:
        bases, scored = B.assign([30, 10, 99], new_base, runs)  # third, first, batter
        assert 10 not in bases, "the man forced at second is out, not on a base"
        assert 99 not in bases, "the batter is out on a double play"
        assert 10 not in scored and 99 not in scored


@pytest.mark.parametrize(("base", "outs"), STATES)
def test_identity_is_recoverable_for_every_branch(base: int, outs: int) -> None:
    """Who is where after the play follows from the state, since nobody passes."""
    occupancy = (base >> 2 & 1, base >> 1 & 1, base & 1)
    order = [r for r, on in zip([30, 20, 10], occupancy, strict=True) if on]
    for oc in B.OUTCOMES:
        for _, new_base, _, runs in B.advance_dist(base, outs, oc, B.LEAGUE_DP_RATE):
            bases, scored = B.assign([*order, 99], new_base, runs)
            on_base = [r for r in bases if r >= 0]
            assert len(on_base) == bin(new_base).count("1")
            assert len(set(on_base) | set(scored)) == len(on_base) + len(scored)
            assert set(on_base) | set(scored) <= {*order, 99}


def test_sampling_reproduces_the_distribution() -> None:
    rng = np.random.default_rng(0)
    branches = B.advance_dist(0b011, 1, "1B", B.LEAGUE_DP_RATE)
    counts = {b[1:]: 0 for b in branches}
    n = 40_000
    for _ in range(n):
        counts[B.sample(0b011, 1, "1B", B.LEAGUE_DP_RATE, float(rng.random()))[1:]] += 1
    for prob, *state in branches:
        assert counts[tuple(state)] / n == pytest.approx(prob, abs=0.01)


def test_the_double_play_rate_follows_the_ground_ball_rate() -> None:
    assert B.dp_rate_from_gb(0.30) < B.dp_rate_from_gb(0.45) < B.dp_rate_from_gb(0.60)
    # A league-average ground-ball rate has to return the league's own DP rate.
    assert B.dp_rate_from_gb(0.43) == pytest.approx(B.LEAGUE_DP_RATE, abs=0.015)


def test_the_chain_and_the_sim_agree_on_the_run_environment() -> None:
    """The regression this change exists to prevent.

    The double play lived in the sim and not in the chain, so F5 totals were
    priced 4% hotter than full-game totals off the same lineup. Both now read one
    table; if they ever diverge again, this is what says so. The starter's hook is
    lifted so the two face identical rates.
    """
    half = _inning_distribution(LEAGUE, B.LEAGUE_DP_RATE)
    chain = 5 * float((half * np.arange(len(half))).sum())
    cfg = TeamSimConfig(
        bat_vs_starter=[dict(LEAGUE) for _ in range(9)],
        bat_vs_pen=[dict(LEAGUE) for _ in range(9)],
        gb_dp_rate=B.LEAGUE_DP_RATE,
        starter_bf_cap=99,
        starter_pitch_cap=999,
    )
    res = MonteCarlo(20_000, seed=5).simulate(cfg, cfg)
    sim = float(np.concatenate([res.home_runs_f5, res.away_runs_f5]).mean())
    assert sim == pytest.approx(chain, rel=0.02), f"chain {chain:.4f} vs sim {sim:.4f}"


def test_the_double_play_costs_the_chain_runs() -> None:
    def f5(dp: float) -> float:
        d = _inning_distribution(LEAGUE, dp)
        return 5 * float((d * np.arange(len(d))).sum())

    assert f5(0.0) > f5(0.12) > f5(B.LEAGUE_DP_RATE)
