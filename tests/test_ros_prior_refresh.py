"""The projection file is derived, so every machine has to be able to build it."""

from __future__ import annotations

import os
import time
from datetime import date as Date
from pathlib import Path

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


def _export(folder: Path, name: str, hr: int) -> Path:
    """A FanGraphs-shaped rest-of-season export for one hitter."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    pd.DataFrame(
        [
            {
                "MLBAMID": 1,
                "PA": 200,
                "H": 50,
                "2B": 10,
                "3B": 1,
                "HR": hr,
                "BB": 20,
                "SO": 40,
                "HBP": 2,
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_a_dropped_in_projection_prices_the_hitters_it_lists(tmp_path) -> None:
    """The subscriber's projection beats the Marcel, hitter by hitter."""
    path = tmp_path / "ros_hitters.csv"
    proj = tmp_path / "projections"
    _export(proj, "atc_ros.csv", hr=40)
    ros_prior.refresh_if_stale(path, TODAY, _Client(), projections=proj)
    priors = load_ros_priors(path)
    assert priors[1]["HR"] == pytest.approx(40 / 200)
    # ... and the Marcel still covers the 40 hitters it does not list.
    assert len(priors) == 41


def test_the_preferred_system_wins_over_a_newer_export(tmp_path) -> None:
    """A folder holding every system resolves to the one asked for, not the last download."""
    proj = tmp_path / "projections"
    wanted = _export(proj, "atc_ros.csv", hr=40)
    newer = _export(proj, "batx_ros.csv", hr=10)
    later = time.time() + 60
    os.utime(newer, (later, later))
    assert ros_prior.newest_export(proj, "atc") == wanted
    assert ros_prior.newest_export(proj, "batx") == newer
    assert ros_prior.newest_export(proj, "steamer") == newer  # nothing matched: newest
    assert ros_prior.newest_export(proj, "") == newer


def test_a_new_export_refreshes_a_file_that_is_not_yet_a_week_old(tmp_path) -> None:
    """The export is dropped by hand, so it cannot wait for the weekly clock."""
    path = tmp_path / "ros_hitters.csv"
    proj = tmp_path / "projections"
    ros_prior.refresh_if_stale(path, TODAY, _Client(), projections=proj)
    assert not ros_prior.is_stale(path, TODAY, projections=proj)
    export = _export(proj, "atc_ros.csv", hr=40)
    later = time.time() + 60
    os.utime(export, (later, later))
    assert ros_prior.is_stale(path, TODAY, projections=proj)
    ros_prior.refresh_if_stale(path, TODAY, _Client(), projections=proj)
    assert load_ros_priors(path)[1]["HR"] == pytest.approx(40 / 200)


def test_an_unreadable_export_leaves_the_marcel_in_charge(tmp_path, caplog) -> None:
    path = tmp_path / "ros_hitters.csv"
    proj = tmp_path / "projections"
    proj.mkdir()
    (proj / "atc_ros.csv").write_text("Name,Team\nnot a projection,KCR\n")
    ros_prior.refresh_if_stale(path, TODAY, _Client(), projections=proj)
    priors = load_ros_priors(path)
    assert len(priors) == 41
    assert priors[1]["HR"] < 0.2  # the Marcel's slugger, not the export's
    assert "using the Marcel" in caplog.text


def test_a_rounded_bench_line_is_left_to_the_marcel(tmp_path) -> None:
    """Two projected PA rounded to integers is a .000 hitter, not a projection."""
    path = tmp_path / "ros_hitters.csv"
    proj = tmp_path / "projections"
    proj.mkdir()
    pd.DataFrame(
        [
            {"MLBAMID": 1, "PA": 2, "H": 0, "2B": 0, "3B": 0, "HR": 0, "BB": 0, "SO": 0, "HBP": 0},
            {"MLBAMID": 2, "PA": 200, "H": 50, "2B": 10, "3B": 1, "HR": 40, "BB": 20, "SO": 40,
             "HBP": 2},
        ]
    ).to_csv(proj / "atc_ros.csv", index=False)
    ros_prior.refresh_if_stale(path, TODAY, _Client(), projections=proj)
    priors = load_ros_priors(path)
    assert priors[2]["HR"] == pytest.approx(40 / 200)
    assert priors[1]["HR"] > 0.05  # the Marcel's slugger, not a .000 bench line


def test_an_export_prices_the_slate_when_the_api_is_down(tmp_path) -> None:
    """Losing the Marcel costs the bench, not the lineup."""
    path = tmp_path / "ros_hitters.csv"
    proj = tmp_path / "projections"
    _export(proj, "atc_ros.csv", hr=40)
    ros_prior.refresh_if_stale(path, TODAY, _Client(fail=True), projections=proj)
    priors = load_ros_priors(path)
    assert list(priors) == [1]
    assert priors[1]["HR"] == pytest.approx(40 / 200)
