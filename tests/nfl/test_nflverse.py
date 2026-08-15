"""The data layer's two promises: cache the immutable, survive the missing."""

from __future__ import annotations

import io
from datetime import date
from typing import NamedTuple

import pandas as pd
import pytest
import requests

from nfl_engine.data import nflverse


class _Resp(NamedTuple):
    content: bytes

    def raise_for_status(self) -> None:
        return None


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(nflverse, "cache_dir", lambda: tmp_path)


def _games_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2024_01_C_D", "2026_01_E_F"],
            "season": [2024, 2024, 2026],
            "week": [1, 1, 1],
            "home_score": [24.0, 17.0, None],
            "away_score": [21.0, 20.0, None],
            "result": [3.0, -3.0, None],
            "total": [45.0, 37.0, None],
            "spread_line": [2.5, -1.0, 3.0],
            "total_line": [44.5, 41.0, 46.5],
        }
    )


def test_a_season_in_progress_is_refetched_and_a_finished_one_is_not():
    """A 2019 parquet will never change; this week's will."""
    assert nflverse._season_ttl(2019) is None
    live = nflverse._current_season(date(2026, 10, 1))
    assert nflverse._season_ttl(live + 1) == nflverse.LIVE_TTL
    # A season is named for its September, so January belongs to the year before.
    assert nflverse._current_season(date(2027, 1, 15)) == 2026
    assert nflverse._current_season(date(2026, 9, 15)) == 2026


def test_the_second_call_is_served_from_disk(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_get(url: str, **kw: object) -> _Resp:
        calls.append(url)
        return _Resp(_csv_bytes(_games_frame()))

    monkeypatch.setattr(nflverse.http, "get", fake_get)
    first = nflverse.games()
    second = nflverse.games()
    assert len(calls) == 1
    assert len(first) == len(second) == 3


def test_a_failed_fetch_returns_empty_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """One lost optional feed must not take the slate with it."""

    def boom(url: str, **kw: object) -> _Resp:
        raise requests.RequestException("connection reset")

    monkeypatch.setattr(nflverse.http, "get", boom)
    with caplog.at_level("WARNING"):
        assert nflverse.play_by_play(2024).empty
        assert nflverse.games().empty
    assert "fetch failed" in caplog.text


def test_a_failed_fetch_falls_back_to_a_stale_cache(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_get(url: str, **kw: object) -> _Resp:
        calls.append(url)
        return _Resp(_csv_bytes(_games_frame()))

    monkeypatch.setattr(nflverse.http, "get", fake_get)
    nflverse.games()
    # Expire the cache, then lose the network: yesterday's file beats nothing.
    monkeypatch.setattr(nflverse, "LIVE_TTL", -1)

    def boom(url: str, **kw: object) -> _Resp:
        raise requests.RequestException("timeout")

    monkeypatch.setattr(nflverse.http, "get", boom)
    assert len(nflverse.games()) == 3


def test_graded_games_drop_the_ungraded_and_keep_the_closing_line(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(nflverse.http, "get", lambda url, **kw: _Resp(_csv_bytes(_games_frame())))
    graded = nflverse.graded_games()
    assert list(graded.game_id) == ["2024_01_A_B", "2024_01_C_D"]
    assert {"spread_line", "total_line", "result", "total"} <= set(graded.columns)
    assert nflverse.graded_games(first_season=2026).empty


def test_a_game_file_without_closing_lines_is_refused(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Silently backtesting without a closing line would measure nothing."""
    frame = _games_frame().drop(columns=["spread_line"])
    monkeypatch.setattr(nflverse.http, "get", lambda url, **kw: _Resp(_csv_bytes(frame)))
    with caplog.at_level("WARNING"):
        assert nflverse.graded_games().empty
    assert "missing columns" in caplog.text


def test_seasons_before_a_feed_exists_are_empty_without_a_request(
    monkeypatch: pytest.MonkeyPatch,
):
    def boom(url: str, **kw: object) -> _Resp:
        raise AssertionError(f"should not have been requested: {url}")

    monkeypatch.setattr(nflverse.http, "get", boom)
    assert nflverse.participation(2012).empty
    assert nflverse.snap_counts(2005).empty
    assert nflverse.next_gen_stats(2010).empty


def test_seasonal_assets_use_the_release_paths_that_exist(monkeypatch: pytest.MonkeyPatch):
    """The release folder and the file name differ for half these assets."""
    seen: list[str] = []
    frame = pd.DataFrame({"season": [2024], "player_id": ["00-1"]})

    def fake_get(url: str, **kw: object) -> _Resp:
        seen.append(url)
        return _Resp(_parquet_bytes(frame))

    monkeypatch.setattr(nflverse.http, "get", fake_get)
    nflverse.play_by_play(2024)
    nflverse.player_week(2024)
    nflverse.team_week(2024)
    nflverse.rosters(2024)
    nflverse.snap_counts(2024)
    nflverse.participation(2024)
    tails = [url.rsplit("/download/", 1)[-1] for url in seen]
    assert tails == [
        "pbp/play_by_play_2024.parquet",
        "stats_player/stats_player_week_2024.parquet",
        "stats_team/stats_team_week_2024.parquet",
        "rosters/roster_2024.parquet",
        "snap_counts/snap_counts_2024.parquet",
        "pbp_participation/pbp_participation_2024.parquet",
    ]


def test_next_gen_and_pfr_files_are_filtered_to_the_season(monkeypatch: pytest.MonkeyPatch):
    """Both ship as one file for all seasons, so the filter is ours to apply."""
    frame = pd.DataFrame({"season": [2023, 2024, 2024], "player": ["a", "b", "c"]})
    monkeypatch.setattr(
        nflverse.http, "get", lambda url, **kw: _Resp(_parquet_bytes(frame))
    )
    assert len(nflverse.next_gen_stats(2024, "receiving")) == 2
    assert len(nflverse.pfr_advstats(2023, "pass")) == 1


def test_multi_season_loads_skip_the_seasons_that_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_get(url: str, **kw: object) -> _Resp:
        if "2023" in url:
            raise requests.RequestException("no such release")
        return _Resp(_parquet_bytes(pd.DataFrame({"season": [2024], "epa": [0.1]})))

    monkeypatch.setattr(nflverse.http, "get", fake_get)
    out = nflverse.load_seasons("play_by_play", [2023, 2024])
    assert len(out) == 1
    with pytest.raises(KeyError):
        nflverse.load_seasons("not_a_loader", [2024])
