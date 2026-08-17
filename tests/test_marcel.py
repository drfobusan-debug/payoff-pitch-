"""The projection the batter prior shrinks toward, built from free season lines."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from mlb_engine.config import Config, default_ros_prior_path
from mlb_engine.features.marcel import (
    MARCEL_WEIGHTS,
    PROJECTED_PA,
    REGRESSION_PA,
    age_factor,
    marcel_projection,
)
from mlb_engine.features.rolling import OUTCOMES_ORDER, ros_rates_from_projection


def _line(pid: int, season: int, pa: int, **over: int) -> dict[str, int]:
    row = {
        "mlbam_id": pid,
        "season": season,
        "PA": pa,
        "H": int(pa * 0.24),
        "2B": int(pa * 0.045),
        "3B": int(pa * 0.004),
        "HR": int(pa * 0.03),
        "BB": int(pa * 0.09),
        "SO": int(pa * 0.22),
        "HBP": int(pa * 0.01),
    }
    row.update(over)
    return row


def _league(seasons: tuple[int, ...] = (2026, 2025, 2024)) -> list[dict[str, int]]:
    """Enough average hitters for the league rates to be well defined."""
    return [_line(900 + i, s, 600) for s in seasons for i in range(30)]


def test_a_thin_line_regresses_almost_all_the_way_to_the_league() -> None:
    """A 40-PA call-up who homered ten times is not a 150-homer hitter."""
    lines = pd.DataFrame(_league() + [_line(1, 2026, 40, HR=10), _line(2, 2026, 600)])
    proj = marcel_projection(lines, {}, 2026, min_weighted_pa=0.0).set_index("MLBAMID")
    hot = proj.loc[1, "HR"] / PROJECTED_PA
    average = proj.loc[2, "HR"] / PROJECTED_PA
    # He homered in a quarter of his plate appearances -- eight times league --
    # and 200 weighted PA against 1200 of regression leaves him at twice it.
    assert average < hot < average * 2.5


def test_a_full_season_keeps_most_of_its_edge() -> None:
    lines = pd.DataFrame(_league() + [_line(1, s, 600, HR=60) for s in (2026, 2025)])
    proj = marcel_projection(lines, {}, 2026, min_weighted_pa=0.0).set_index("MLBAMID")
    # 60 HR a year over two seasons is 5400 weighted PA against 1200 of
    # regression: the projection keeps roughly three quarters of the observed
    # rate, where the thin line above keeps almost none of its.
    assert 0.06 < proj.loc[1, "HR"] / PROJECTED_PA < 0.09


def test_the_current_season_outweighs_the_two_before_it() -> None:
    recent, old = 1, 2
    lines = pd.DataFrame(
        _league()
        + [_line(recent, 2026, 600, HR=60), _line(recent, 2024, 600, HR=0)]
        + [_line(old, 2026, 600, HR=0), _line(old, 2024, 600, HR=60)]
    )
    proj = marcel_projection(lines, {}, 2026, min_weighted_pa=0.0).set_index("MLBAMID")
    assert proj.loc[recent, "HR"] > proj.loc[old, "HR"]
    assert MARCEL_WEIGHTS[0] > MARCEL_WEIGHTS[-1]


def test_seasons_outside_the_window_are_ignored() -> None:
    lines = pd.DataFrame(_league() + [_line(1, 2019, 600, HR=200)])
    proj = marcel_projection(lines, {}, 2026, min_weighted_pa=0.0)
    assert 1 not in set(proj["MLBAMID"])


def test_a_young_hitter_is_aged_up_and_an_old_one_down() -> None:
    lines = pd.DataFrame(_league() + [_line(1, 2026, 600), _line(2, 2026, 600)])
    proj = marcel_projection(lines, {1: 23.0, 2: 38.0}, 2026, min_weighted_pa=0.0)
    young, old = proj.set_index("MLBAMID").loc[1], proj.set_index("MLBAMID").loc[2]
    assert young["HR"] > old["HR"]
    assert age_factor(23.0) > 1.0 > age_factor(38.0)
    # Aging must not break the hit hierarchy the singles residual depends on.
    assert young["H"] >= young["2B"] + young["3B"] + young["HR"]


def test_hitters_with_no_meaningful_history_are_left_to_the_league_prior() -> None:
    lines = pd.DataFrame(_league() + [_line(1, 2026, 12)])
    proj = marcel_projection(lines, {}, 2026, min_weighted_pa=100.0)
    assert 1 not in set(proj["MLBAMID"])


def test_the_output_is_what_the_prior_reads() -> None:
    lines = pd.DataFrame(_league() + [_line(1, 2026, 600, HR=45)])
    ros = ros_rates_from_projection(marcel_projection(lines, {1: 27.0}, 2026))
    assert set(ros["mlbam_id"]) >= {1}
    vec = ros[ros["mlbam_id"] == 1].iloc[0]
    assert pytest.approx(sum(vec[oc] for oc in OUTCOMES_ORDER), abs=1e-9) == 1.0
    assert all(vec[oc] > 0 for oc in OUTCOMES_ORDER)


def test_missing_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="missing"):
        marcel_projection(pd.DataFrame([{"mlbam_id": 1, "season": 2026}]), {}, 2026)


def test_regression_constant_is_marcels() -> None:
    assert REGRESSION_PA == 1200.0


def test_the_prior_is_on_by_default_and_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.delenv("MLBE_ROS_PRIOR", raising=False)
    assert Config().ros_prior_path == default_ros_prior_path()
    monkeypatch.setenv("MLBE_ROS_PRIOR", "")
    assert Config().ros_prior_path is None
    monkeypatch.setenv("MLBE_ROS_PRIOR", os.path.join("elsewhere", "ros.csv"))
    assert Config().ros_prior_path == os.path.join("elsewhere", "ros.csv")
