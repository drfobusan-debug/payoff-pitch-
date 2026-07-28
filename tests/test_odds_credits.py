"""The Odds API bills markets x regions, so the client has to budget."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlb_engine.data.oddsapi import DEFAULT_PROP_MARKETS, OddsAPIClient
from mlb_engine.schemas import Game, Slate, TeamGameInfo, Venue


def _slate(n: int = 2) -> Slate:
    names = [
        ("Cleveland Guardians", "CLE", "Minnesota Twins", "MIN"),
        ("Cincinnati Reds", "CIN", "Chicago Cubs", "CHC"),
        ("Detroit Tigers", "DET", "Kansas City Royals", "KC"),
    ]
    games = []
    for i, (home, hab, away, aab) in enumerate(names[:n]):
        games.append(
            Game(
                game_pk=i + 1,
                game_date=datetime.date(2026, 7, 28),
                status="Preview",
                venue=Venue(venue_id=1, name="x"),
                home=TeamGameInfo(team_id=1 + i, name=home, abbrev=hab, is_home=True),
                away=TeamGameInfo(team_id=90 + i, name=away, abbrev=aab, is_home=False),
            )
        )
    return Slate(slate_date=datetime.date(2026, 7, 28), games=games)


def _bulk(slate: Slate) -> list[dict]:
    return [
        {"id": f"evt{i}", "home_team": g.home.name, "away_team": g.away.name, "bookmakers": []}
        for i, g in enumerate(slate.games)
    ]


def test_default_props_exclude_the_markets_the_engine_never_bets() -> None:
    """HR/doubles/runs/RBI produced zero favored picks in 54 slates; singles and
    ER priced below break-even. Paying a credit each buys prices for no bet."""
    client = OddsAPIClient("k")
    assert client.prop_markets == DEFAULT_PROP_MARKETS
    for skipped in (
        "batter_home_runs", "batter_doubles", "batter_runs_scored",
        "batter_rbis", "batter_singles", "pitcher_earned_runs",
    ):
        assert skipped not in client.prop_markets
    # 3 F5 + 5 props = 8 credits per event, down from 14.
    assert len(client.event_markets()) == 8


def test_prop_override_is_filtered_to_known_markets() -> None:
    client = OddsAPIClient("k", prop_markets=("batter_hits", "batter_smiles"), include_f5=False)
    assert client.event_markets() == ["batter_hits"]


def test_per_event_loop_stops_before_draining_the_plan() -> None:
    """Reserve is checked before each event, so a slate cannot overrun it."""
    slate = _slate(3)
    client = OddsAPIClient("k", min_credits=200)
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        if "/events/" in url:
            client.credits_remaining = 210 - 8 * len([c for c in calls if "/events/" in c])
            return {"id": "e", "bookmakers": []}
        client.credits_remaining = 210
        return _bulk(slate)

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    # One event is affordable (210 - 8 = 202); the next would break the reserve.
    assert len([c for c in calls if "/events/" in c]) == 1


def test_unknown_credit_count_does_not_block_fetching() -> None:
    slate = _slate(2)
    client = OddsAPIClient("k")
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        return {"id": "e", "bookmakers": []} if "/events/" in url else _bulk(slate)

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)
    assert len([c for c in calls if "/events/" in c]) == 2


def test_cache_hit_costs_no_request(tmp_path: Path) -> None:
    client = OddsAPIClient("k", cache_dir=tmp_path, cache_ttl=1800)
    resp = MagicMock(status_code=200, headers={"x-requests-remaining": "8000"})
    resp.json.return_value = [{"id": "evt"}]

    with patch("requests.get", return_value=resp) as get:
        first = client._get_json("https://x/odds", markets="h2h")
        second = client._get_json("https://x/odds", markets="h2h")

    assert get.call_count == 1
    assert first == second == [{"id": "evt"}]
    assert client.credits_remaining == 8000


def test_cache_key_ignores_the_api_key_and_is_not_written_to_disk(tmp_path: Path) -> None:
    """The cached filename and payload must never carry the credential."""
    client = OddsAPIClient("SEKRIT", cache_dir=tmp_path)
    resp = MagicMock(status_code=200, headers={})
    resp.json.return_value = {"ok": True}

    with patch("requests.get", return_value=resp):
        client._get_json("https://x/odds", markets="h2h")

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1
    assert "SEKRIT" not in written[0].name
    assert "SEKRIT" not in written[0].read_text()
    assert json.loads(written[0].read_text()) == {"ok": True}


def test_stale_cache_is_refetched(tmp_path: Path) -> None:
    client = OddsAPIClient("k", cache_dir=tmp_path, cache_ttl=0)
    resp = MagicMock(status_code=200, headers={})
    resp.json.return_value = {"ok": True}

    with patch("requests.get", return_value=resp) as get:
        client._get_json("https://x/odds", markets="h2h")
        client._get_json("https://x/odds", markets="h2h")

    assert get.call_count == 2


def test_f5_can_be_dropped_from_the_request() -> None:
    client = OddsAPIClient("k", include_f5=False)
    assert client.event_markets() == list(DEFAULT_PROP_MARKETS)
