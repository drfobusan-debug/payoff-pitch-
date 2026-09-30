from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from mlb_engine.data import fangraphs


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_leaderboard_caches_a_good_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"TeamName": "NYY", "SIERA": 3.5}]
    monkeypatch.setattr(fangraphs.http, "get", lambda url, **kw: _Resp({"data": rows}))
    assert fangraphs.leaderboard("https://fg/x", "rel_2026", cache_dir=tmp_path) == rows
    saved = json.loads((tmp_path / "rel_2026.json").read_text())
    assert saved["data"] == rows and saved["fetched"]


def test_leaderboard_falls_back_to_the_last_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "rel_2026.json").write_text(json.dumps({"fetched": "2026-09-25", "data": [{"TeamName": "BOS"}]}))

    def down(url: str, **kw: object) -> _Resp:
        raise requests.ConnectionError("fangraphs unreachable")

    monkeypatch.setattr(fangraphs.http, "get", down)
    assert fangraphs.leaderboard("https://fg/x", "rel_2026", cache_dir=tmp_path) == [{"TeamName": "BOS"}]

    monkeypatch.setattr(fangraphs.http, "get", lambda url, **kw: _Resp({"data": []}))
    assert fangraphs.leaderboard("https://fg/x", "rel_2026", cache_dir=tmp_path) == [{"TeamName": "BOS"}]


def test_leaderboard_raises_without_a_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def down(url: str, **kw: object) -> _Resp:
        raise requests.ConnectionError("fangraphs unreachable")

    monkeypatch.setattr(fangraphs.http, "get", down)
    with pytest.raises(requests.ConnectionError):
        fangraphs.leaderboard("https://fg/x", "sta_2026", cache_dir=tmp_path)
