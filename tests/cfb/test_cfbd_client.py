"""What the CFBD client actually asks for."""

from __future__ import annotations

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
    CFBDClient("key").fetch_advanced(2026)

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
    book = CFBDClient("key").fetch_starters(2026, 4)

    assert weeks == [1, 2, 3]
    starter = book["iowa"]
    assert starter.name == "Cade McNamara"
    assert starter.attempts == 75  # 25 a week for three weeks
    assert starter.share == pytest.approx(75 / 81)


def test_week_one_has_no_usage_to_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kw: object) -> _Resp:
        raise AssertionError("week 1 should not ask for box scores")

    monkeypatch.setattr(cfbd.http, "get", fake_get)
    assert CFBDClient("key").fetch_starters(2026, 1) == {}
    assert CFBDClient(None).fetch_starters(2026, 9) == {}
