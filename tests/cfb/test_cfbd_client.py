"""What the CFBD client actually asks for."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import NamedTuple

import pytest

from cfb_engine.data import cfbd
from cfb_engine.data.cfbd import CFBDClient


class _Resp(NamedTuple):
    payload: list[dict[str, object]]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict[str, object]]:
        return self.payload


def test_advanced_stats_exclude_garbage_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mop-up snaps dilute the per-play rates every threshold is judged against.

    Over 2024's 134 teams, dropping them moves net PPA by .014/play on average,
    flips 1.0% of pairwise net-PPA matchups and moves 33 teams ten or more ranks
    in explosiveness -- small, but free to get right, and the league means the
    deadbands compare against are computed on the same scale.
    """
    seen: list[dict[str, object]] = []

    def fake_get(url: str, **kw: object) -> _Resp:
        params = kw.get("params")
        seen.append({"url": url, "params": params})
        row = {
            "team": "Georgia",
            "offense": {"ppa": 0.3, "successRate": 0.5},
            "defense": {"ppa": 0.1, "successRate": 0.4},
        }
        return _Resp([row] if "advanced" in url else [])

    monkeypatch.setattr(cfbd.http, "get", fake_get)
    CFBDClient("key", cache_dir=None).fetch_advanced(2026)

    advanced = [c for c in seen if "advanced" in str(c["url"])]
    assert advanced, "advanced stats were never requested"
    for call in advanced:
        params = call["params"]
        assert isinstance(params, dict)
        assert params.get("excludeGarbageTime") == "true"


def _box(team: str, athletes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "teams": [
            {
                "team": team,
                "categories": [
                    {
                        "name": "passing",
                        "types": [
                            {"name": "YDS", "athletes": athletes},
                            {"name": "C/ATT", "athletes": athletes},
                        ],
                    }
                ],
            }
        ]
    }


def test_starters_read_only_weeks_before_the_slate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage counted from the slate week onward would be the game's own box score."""
    weeks: list[object] = []

    def fake_get(url: str, **kw: object) -> _Resp:
        params = kw.get("params")
        assert isinstance(params, dict)
        weeks.append(params.get("week"))
        athletes = [
            {"id": "1", "name": "Cade McNamara", "stat": "18/25"},
            {"id": "2", "name": "Deacon Hill", "stat": "1/2"},
            {"id": "3", "name": "Kaleb Johnson", "stat": "no attempts"},
        ]
        return _Resp([_box("Iowa", athletes)])

    monkeypatch.setattr(cfbd.http, "get", fake_get)
    book = CFBDClient("key", cache_dir=None).fetch_starters(2026, 4)

    assert weeks == [1, 2, 3]
    starter = book["iowa"]
    assert starter.name == "Cade McNamara"
    assert starter.attempts == 75  # 25 a week for three weeks
    assert starter.share == pytest.approx(75 / 81)


def test_week_one_has_no_usage_to_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kw: object) -> _Resp:
        raise AssertionError("week 1 should not ask for box scores")

    monkeypatch.setattr(cfbd.http, "get", fake_get)
    assert CFBDClient("key", cache_dir=None).fetch_starters(2026, 1) == {}
    assert CFBDClient(None, cache_dir=None).fetch_starters(2026, 9) == {}


# -- the call budget -------------------------------------------------------
# CFBD bills by the call and the free key allows 1,000 a month. The first key
# was exhausted at 1,851 calls in 90 days, 44% of them one endpoint
# (/games/players) re-walked week by week on every run.


def _counting_get(calls: list[str]) -> Callable[..., _Resp]:
    def fake_get(url: str, **kw: object) -> _Resp:
        calls.append(url)
        return _Resp([{"id": 1, "latitude": 1.0, "longitude": 2.0}])

    return fake_get


def test_a_second_run_spends_no_calls_on_the_same_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cfbd.http, "get", _counting_get(calls))

    for _ in range(3):
        CFBDClient("key", cache_dir=tmp_path).fetch_venues()

    assert len(calls) == 1


def test_an_expired_entry_is_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cfbd.http, "get", _counting_get(calls))

    client = CFBDClient("key", cache_dir=tmp_path, cache_ttl=0)
    client.fetch_ratings(2026)
    client.fetch_ratings(2026)

    assert len(calls) > 1


def test_a_failed_call_falls_back_to_a_stale_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An exhausted quota should cost freshness, not the whole card."""
    monkeypatch.setattr(cfbd.http, "get", _counting_get([]))
    assert CFBDClient("key", cache_dir=tmp_path).fetch_venues()

    def quota_exceeded(url: str, **kw: object) -> _Resp:
        raise cfbd.requests.RequestException("429 Monthly call quota exceeded")

    monkeypatch.setattr(cfbd.http, "get", quota_exceeded)
    # Expired, and the network is refusing: the cache is all there is.
    assert CFBDClient("key", cache_dir=tmp_path, cache_ttl=0).fetch_venues()


def test_a_finished_season_is_never_refetched(tmp_path: Path) -> None:
    """Past seasons are immutable, so a backtest buys their history once."""
    client = CFBDClient("key", cache_dir=tmp_path, cache_ttl=0)
    past = cfbd.current_season() - 1

    assert client._ttl("/games", {"year": past}) == cfbd.LONG_TTL
    assert client._ttl("/games", {"year": cfbd.current_season()}) == cfbd.SHORT_TTL


def test_january_bowls_belong_to_the_previous_season() -> None:
    assert cfbd.current_season(date(2026, 1, 9)) == 2025
    assert cfbd.current_season(date(2026, 9, 5)) == 2026


def test_the_cache_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFBE_CFBD_CACHE", "0")
    assert cfbd.default_cache_dir() is None


def test_the_cache_lives_under_the_engine_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CFBE_CFBD_CACHE", raising=False)
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    assert cfbd.default_cache_dir() == tmp_path / "cache" / "cfbd"
