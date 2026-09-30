"""Open-Meteo kickoff weather: hour matching, batching, and failing soft."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlb_engine.data import openmeteo as weather


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _hourly(day: str, wind, precip=None, temp=None):
    hours = [f"{day}T{h:02d}:00" for h in range(24)]
    return {
        "hourly": {
            "time": hours,
            "wind_speed_10m": [wind] * 24,
            "wind_gusts_10m": [wind * 1.5 for _ in hours],
            "precipitation": [precip if precip is not None else 0.0] * 24,
            "temperature_2m": [temp if temp is not None else 60.0] * 24,
        }
    }


def test_reads_the_hour_closest_to_kickoff(monkeypatch):
    day = "2026-09-05"
    hours = [f"{day}T{h:02d}:00" for h in range(24)]
    payload = {
        "hourly": {
            "time": hours,
            "wind_speed_10m": [float(h) for h in range(24)],
            "wind_gusts_10m": [0.0] * 24,
            "precipitation": [0.0] * 24,
            "temperature_2m": [60.0] * 24,
        }
    }
    monkeypatch.setattr(weather.http, "get", lambda *a, **k: _Resp(payload))
    kick = datetime(2026, 9, 5, 19, 10, tzinfo=timezone.utc)
    out = weather.fetch_venue_weather({"g1": (30.0, -97.0, kick)})
    assert out["g1"].wind_mph == 19.0


def test_one_call_covers_every_venue_playing_that_day(monkeypatch):
    day = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    calls: list[dict] = []

    def fake_get(url, params=None, **kwargs):
        calls.append(params or {})
        return _Resp([_hourly(day, 5.0), _hourly(day, 20.0)])

    monkeypatch.setattr(weather.http, "get", fake_get)
    kick = datetime.fromisoformat(f"{day}T18:00").replace(tzinfo=timezone.utc)
    out = weather.fetch_venue_weather(
        {"a": (30.0, -97.0, kick), "b": (40.0, -83.0, kick)}
    )
    assert len(calls) == 1
    assert calls[0]["latitude"].count(",") == 1
    assert out["a"].wind_mph == 5.0 and out["b"].wind_mph == 20.0


def test_past_dates_use_the_archive_endpoint(monkeypatch):
    urls: list[str] = []

    def fake_get(url, params=None, **kwargs):
        urls.append(url)
        return _Resp(_hourly("2024-11-02", 8.0))

    monkeypatch.setattr(weather.http, "get", fake_get)
    kick = datetime(2024, 11, 2, 17, 0, tzinfo=timezone.utc)
    weather.fetch_venue_weather({"g": (30.0, -97.0, kick)})
    assert urls == [weather._ARCHIVE]


def test_upcoming_dates_use_the_forecast_endpoint(monkeypatch):
    urls: list[str] = []
    day = (datetime.now(timezone.utc) + timedelta(days=2)).date()

    def fake_get(url, params=None, **kwargs):
        urls.append(url)
        return _Resp(_hourly(day.isoformat(), 8.0))

    monkeypatch.setattr(weather.http, "get", fake_get)
    kick = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=18)
    weather.fetch_venue_weather({"g": (30.0, -97.0, kick)})
    assert urls == [weather._FORECAST]


def test_a_reading_far_from_kickoff_is_discarded(monkeypatch):
    payload = {
        "hourly": {
            "time": ["2026-09-05T02:00"],
            "wind_speed_10m": [9.0],
            "wind_gusts_10m": [9.0],
            "precipitation": [0.0],
            "temperature_2m": [60.0],
        }
    }
    monkeypatch.setattr(weather.http, "get", lambda *a, **k: _Resp(payload))
    kick = datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc)
    assert weather.fetch_venue_weather({"g": (30.0, -97.0, kick)}) == {}


@pytest.mark.parametrize(
    "payload",
    [_Resp({}, status=503), _Resp({"hourly": "nonsense"}), _Resp([])],
)
def test_a_bad_response_leaves_the_game_without_weather(monkeypatch, payload):
    monkeypatch.setattr(weather.http, "get", lambda *a, **k: payload)
    kick = datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc)
    assert weather.fetch_venue_weather({"g": (30.0, -97.0, kick)}) == {}


def test_a_network_error_is_not_fatal(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("reset by peer")

    monkeypatch.setattr(weather.http, "get", boom)
    kick = datetime(2026, 9, 5, 19, 0, tzinfo=timezone.utc)
    assert weather.fetch_venue_weather({"g": (30.0, -97.0, kick)}) == {}
