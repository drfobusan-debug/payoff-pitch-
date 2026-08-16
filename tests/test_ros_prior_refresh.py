"""The projection file is derived, so every machine has to be able to build it."""

from __future__ import annotations

import os
import time
from datetime import date as Date

import pandas as pd
import pytest

from mlb_engine.data import ros_prior
from mlb_engine.features.rolling import load_ros_priors

TODAY = Date(2026, 8, 17)


class _Client:
    """A Stats API stand-in: 40 average hitters a season, one slugger."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.seasons: list[int] = []

    def season_hitting(self, season: int, limit: int = 2000) -> list[dict[str, int]]:
        if self.fail:
            raise ConnectionError("statsapi unreachable")
        self.seasons.append(season)
        rows = [
            {
                "mlbam_id": 900 + i,
                "season": season,
                "PA": 600,
                "H": 144,
                "2B": 27,
                "3B": 2,
                "HR": 18,
                "BB": 54,
                "SO": 132,
                "HBP": 6,
            }
            for i in range(40)
        ]
        rows.append({**rows[0], "mlbam_id": 1, "HR": 50, "SO": 100})
        return rows

    def player_ages(self, ids: set[int]) -> dict[int, float]:
        return dict.fromkeys(ids, 28.0)


def test_a_fresh_machine_builds_the_file_the_prior_reads(tmp_path) -> None:
    path = tmp_path / "ros_hitters.csv"
    client = _Client()
    ros_prior.refresh_if_stale(path, TODAY, client)
    assert client.seasons == [2026, 2025, 2024]
    priors = load_ros_priors(path)
    assert 1 in priors
    assert priors[1]["HR"] > priors[900]["HR"]
    assert sum(priors[1].values()) == pytest.approx(1.0, abs=1e-9)


def test_a_fresh_file_is_left_alone(tmp_path) -> None:
    path = tmp_path / "ros_hitters.csv"
    pd.DataFrame([{"mlbam_id": 1}]).to_csv(path, index=False)
    client = _Client()
    ros_prior.refresh_if_stale(path, TODAY, client)
    assert client.seasons == []


def test_a_week_old_file_is_rebuilt(tmp_path) -> None:
    """The current season is the heaviest of Marcel's three and grows nightly."""
    path = tmp_path / "ros_hitters.csv"
    pd.DataFrame([{"mlbam_id": 1}]).to_csv(path, index=False)
    stale = time.time() - (ros_prior.MAX_AGE_DAYS + 1) * 86400
    os.utime(path, (stale, stale))
    assert ros_prior.is_stale(path, Date.today())
    client = _Client()
    ros_prior.refresh_if_stale(path, Date.today(), client)
    assert client.seasons


def test_an_unreachable_api_does_not_stop_the_slate(tmp_path, caplog) -> None:
    """Failing here costs the projection, not the card: it prices on the league."""
    path = tmp_path / "ros_hitters.csv"
    ros_prior.refresh_if_stale(path, TODAY, _Client(fail=True))
    assert not path.exists()
    assert load_ros_priors(path) == {}
    assert "league prior" in caplog.text


def test_a_failed_rebuild_keeps_the_old_file(tmp_path) -> None:
    """A week-old projection beats no projection, so a bad night changes nothing."""
    path = tmp_path / "ros_hitters.csv"
    ros_prior.refresh_if_stale(path, TODAY, _Client())
    before = path.read_text()
    stale = time.time() - (ros_prior.MAX_AGE_DAYS + 1) * 86400
    os.utime(path, (stale, stale))
    ros_prior.refresh_if_stale(path, TODAY, _Client(fail=True))
    assert path.read_text() == before


def test_no_configured_path_means_no_network_call() -> None:
    client = _Client()
    ros_prior.refresh_if_stale(None, TODAY, client)
    assert client.seasons == []
