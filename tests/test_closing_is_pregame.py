"""A close is the last price before first pitch, not the price during the game.

On 2026-08-08 no scheduled capture ran, and the snapshot got taken after the
games finished instead. The vendor keeps quoting a live game, so the file
recorded ATH ML at -2000 and BOS at +1600 in the same matchup -- a settled
score, not a market opinion. Scored as CLV that reads as a huge edge or a huge
miss depending only on who won, which is worse than having no CLV at all.
"""

from __future__ import annotations

import datetime

from mlb_engine.data.oddsapi import OddsAPIClient, _Event
from mlb_engine.schemas import Game, Slate, TeamGameInfo, Venue

_NOW = datetime.datetime(2026, 8, 8, 23, 30, tzinfo=datetime.timezone.utc)


def _iso(offset_minutes: int) -> str:
    """A first pitch that many minutes from the wall clock the test runs on."""
    at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=offset_minutes
    )
    return at.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slate() -> Slate:
    names = [
        ("Boston Red Sox", "BOS", "Oakland Athletics", "ATH"),
        ("Detroit Tigers", "DET", "Kansas City Royals", "KC"),
    ]
    games = [
        Game(
            game_pk=i + 1,
            game_date=datetime.date(2026, 8, 8),
            status="Preview",
            venue=Venue(venue_id=1, name="x"),
            home=TeamGameInfo(team_id=1 + i, name=home, abbrev=hab, is_home=True),
            away=TeamGameInfo(team_id=90 + i, name=away, abbrev=aab, is_home=False),
        )
        for i, (home, hab, away, aab) in enumerate(names)
    ]
    return Slate(slate_date=datetime.date(2026, 8, 8), games=games)


def _bulk(slate: Slate, commence: list[str]) -> list[dict]:
    return [
        {
            "id": f"evt{i}",
            "home_team": g.home.name,
            "away_team": g.away.name,
            "commence_time": commence[i],
            "bookmakers": [],
        }
        for i, g in enumerate(slate.games)
    ]


def _client(slate: Slate, commence: list[str]) -> tuple[OddsAPIClient, list[str]]:
    client = OddsAPIClient("k")
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        if "/events/" in url:
            return {"id": "e", "bookmakers": []}
        return _bulk(slate, commence)

    client._get_json = fake_get  # type: ignore[method-assign]
    return client, calls


def test_a_game_under_way_is_dropped_from_the_close() -> None:
    slate = _slate()
    # The first game went off an hour ago; the second is 40 minutes out.
    client, calls = _client(slate, [_iso(-60), _iso(40)])
    client.fetch(slate, pregame_only=True)

    # Only the unstarted game is worth a per-event prop credit.
    assert len([c for c in calls if "/events/" in c]) == 1
    assert "evt1" in "".join(c for c in calls if "/events/" in c)


def test_a_pregame_run_and_a_slate_replay_still_see_every_game() -> None:
    """The default must not change: the audit re-prices a finished slate to
    rebuild yesterday's picks, and dropping started games would price nothing.
    """
    slate = _slate()
    client, calls = _client(slate, [_iso(-60), _iso(40)])
    client.fetch(slate)
    assert len([c for c in calls if "/events/" in c]) == 2


def test_an_event_with_no_start_time_is_treated_as_pregame() -> None:
    """Missing information is not evidence the game is over -- capture it and
    let the merge sort it out, rather than silently losing the whole slate."""
    ev = _Event("e", "BOS", "ATH", "boston", "oakland", None)
    assert not ev.started(_NOW)


def test_first_pitch_exactly_now_counts_as_started() -> None:
    at = datetime.datetime(2026, 8, 8, 23, 30, tzinfo=datetime.timezone.utc)
    assert _Event("e", "BOS", "ATH", "b", "o", at).started(_NOW)
    assert not _Event(
        "e", "BOS", "ATH", "b", "o", at + datetime.timedelta(minutes=1)
    ).started(_NOW)
