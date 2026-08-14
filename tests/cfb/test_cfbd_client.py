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
