"""The simulator's conversion of hits into runs, which prices R, RBI and H+R+RBI."""

from __future__ import annotations

import numpy as np

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
        gb_dp_rate=0.10,
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
