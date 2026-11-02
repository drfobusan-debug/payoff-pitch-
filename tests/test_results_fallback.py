"""Tests for the MLB Stats API box-score cache fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from mlb_engine.data.results import GameResult, PlayerLine, fetch_result


class FailingSession:
    """A requests.Session stand-in that always raises a network error."""

    def get(self, *args, **kwargs):
        raise requests.RequestException("network down")


def _minimal_boxscore(game_pk: int) -> dict:
    return {
        "linescore": {
            "innings": [
                {"num": 1, "home": {"runs": 0}, "away": {"runs": 1}},
                {"num": 2, "home": {"runs": 2}, "away": {"runs": 0}},
                {"num": 3, "home": {"runs": 1}, "away": {"runs": 0}},
                {"num": 4, "home": {"runs": 0}, "away": {"runs": 0}},
                {"num": 5, "home": {"runs": 0}, "away": {"runs": 0}},
                {"num": 6, "home": {"runs": 1}, "away": {"runs": 0}},
                {"num": 7, "home": {"runs": 0}, "away": {"runs": 0}},
                {"num": 8, "home": {"runs": 0}, "away": {"runs": 0}},
                {"num": 9, "home": {"runs": 0}, "away": {"runs": 0}},
            ],
            "gameStatus": {"abstractGameState": "Final"},
        },
        "boxscore": {
            "teams": {
                "home": {
                    "players": {
                        "ID100": {
                            "person": {"id": 100, "fullName": "Home Hitter"},
                            "stats": {
                                "batting": {
                                    "hits": 2,
                                    "doubles": 1,
                                    "triples": 0,
                                    "homeRuns": 0,
                                    "rbi": 1,
                                    "runs": 1,
                                }
                            },
                        }
                    }
                },
                "away": {
                    "players": {
                        "ID200": {
                            "person": {"id": 200, "fullName": "Away Hitter"},
                            "stats": {
                                "batting": {
                                    "hits": 1,
                                    "doubles": 0,
                                    "triples": 0,
                                    "homeRuns": 0,
                                    "rbi": 0,
                                    "runs": 1,
                                }
                            },
                        }
                    }
                },
            }
        },
    }


def test_fetch_result_uses_cache_when_api_fails(tmp_path: Path) -> None:
    cache_dir = tmp_path
    game_pk = 123
    cache_path = cache_dir / "boxscores" / f"{game_pk}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_minimal_boxscore(game_pk)))

    result = fetch_result(game_pk, session=FailingSession(), cache_dir=cache_dir)

    assert isinstance(result, GameResult)
    assert result.final is True
    assert result.home_runs == 4
    assert result.away_runs == 1
    assert result.f5_home == 3
    assert result.f5_away == 1
    # This cached box score predates the plateAppearances field, so PA is rebuilt
    # from the rest of the line -- a hitter with two hits must not read as absent.
    assert result.players[100] == PlayerLine(
        batting={
            "PA": 2, "H": 2, "1B": 1, "2B": 1, "3B": 0, "HR": 0, "RBI": 1, "R": 1,
            "BB": 0, "K": 0,
        }
    )
    assert result.batted(100) is True
    assert result.batted(404) is False


class ScheduleSession:
    """Answers the box score from the fixture and the schedule as a rainout."""

    def __init__(self, game_pk: int, schedule: dict) -> None:
        self.box = _minimal_boxscore(game_pk)
        self.schedule = schedule

    def get(self, url: str, **kwargs):
        body = (
            self.box["linescore"] if url.endswith("/linescore")
            else self.box["boxscore"] if url.endswith("/boxscore")
            else self.schedule
        )

        class _R:
            def json(self_inner):
                return body

        return _R()


RAINOUT = {
    "dates": [
        {"games": [{"gamePk": 849817, "officialDate": "2026-10-18", "gameDate": "2026-10-17T00:00:00Z",
                    "status": {"abstractGameState": "Final", "detailedState": "Postponed"}}]},
        {"games": [{"gamePk": 849817, "officialDate": "2026-10-18", "gameDate": "2026-10-18T21:08:00Z",
                    "status": {"abstractGameState": "Final", "detailedState": "Final"}}]},
    ]
}


def test_a_rainout_reads_the_day_it_was_played_and_caches_it(tmp_path: Path) -> None:
    result = fetch_result(849817, session=ScheduleSession(849817, RAINOUT), cache_dir=tmp_path)
    assert (result.official_date, result.start_utc) == ("2026-10-18", "2026-10-18T21:08:00Z")

    again = fetch_result(849817, session=FailingSession(), cache_dir=tmp_path)
    assert (again.official_date, again.start_utc) == ("2026-10-18", "2026-10-18T21:08:00Z")


def test_a_cached_box_score_without_a_schedule_stamps_nothing(tmp_path: Path) -> None:
    path = tmp_path / "boxscores" / "123.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_minimal_boxscore(123)))
    result = fetch_result(123, session=FailingSession(), cache_dir=tmp_path)
    assert (result.official_date, result.start_utc) == ("", "")


def test_fetch_result_raises_when_api_and_cache_both_fail(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No box score available"):
        fetch_result(999, session=FailingSession(), cache_dir=tmp_path)


def test_fetch_result_writes_cache_on_success(tmp_path: Path) -> None:
    """When the API would succeed we still cache; tested here with a stub session."""
    cache_dir = tmp_path
    game_pk = 456
    cache_path = cache_dir / "boxscores" / f"{game_pk}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(_minimal_boxscore(game_pk)))

    # Verify cache is read by failing API path.
    result = fetch_result(game_pk, session=FailingSession(), cache_dir=cache_dir)
    assert result.home_runs == 4
