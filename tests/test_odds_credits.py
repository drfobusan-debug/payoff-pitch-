"""The Odds API bills markets x regions, so the client has to budget."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlb_engine.data.oddsapi import (
    _BATTER_MARKETS,
    _PITCHER_MARKETS,
    DEFAULT_PROP_MARKETS,
    PRICE_ONLY_MARKETS,
    OddsAPIClient,
)
from mlb_engine.schemas import Game, Slate, TeamGameInfo, Venue

# Markets the engine bought a price for *and* bet, before the fetch list was
# widened. Buying more prices must not add to this set.
_BETTABLE_BEFORE = {
    "batter_h", "pitcher_k", "pitcher_outs", "pitcher_h", "pitcher_bb",
    "game_ml", "game_rl", "game_total", "f5_ml", "f5_rl", "f5_total",
}


def test_price_only_markets_are_fetched_but_never_bet() -> None:
    """Runs is priced to capture the quote, not to buy the over; it must be in
    the fetched set AND in the price-only set that pipeline hard-passes."""
    assert "batter_runs_scored" in DEFAULT_PROP_MARKETS  # its odds are fetched
    assert "batter_r" in PRICE_ONLY_MARKETS  # ...but the over is never recommended
    assert not PRICE_ONLY_MARKETS & _BETTABLE_BEFORE


def test_buying_a_price_still_does_not_make_a_market_bettable() -> None:
    """Reopening is a separate decision from pricing, and it is made per market.

    Six markets were promoted out of the price-only set once the capture had
    graded them. The two still in it are there because nothing in their record
    argues for reopening: runs lost 31.8% with no profitable slice, and pitcher
    ER has never been examined at all. Fetching their prices must not quietly do
    the promoting.
    """
    for vendor in ("batter_runs_scored", "pitcher_earned_runs"):
        assert vendor in DEFAULT_PROP_MARKETS
        engine, _ = {**_BATTER_MARKETS, **_PITCHER_MARKETS}[vendor]
        assert engine in PRICE_ONLY_MARKETS, engine
        assert engine not in _BETTABLE_BEFORE


def test_total_bases_and_hrr_are_mapped_at_all() -> None:
    """Both were absent from the vendor map, so 23.5k ledger rows across them
    were graded against a price that was never fetched."""
    assert _BATTER_MARKETS["batter_total_bases"] == ("batter_tb", "TB")
    assert _BATTER_MARKETS["batter_hits_runs_rbis"] == ("batter_hrr", "H+R+RBI")


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


def test_every_parseable_prop_market_is_priced() -> None:
    """The old list dropped five markets because the engine faded 100% of them and
    scored a high NPV doing so -- but a market faded outright scores its own base
    rate for free, so that was arithmetic defending its own blind spot. Buying
    the price is what lets the audit find out."""
    client = OddsAPIClient("k")
    assert client.prop_markets == DEFAULT_PROP_MARKETS
    assert set(client.prop_markets) == set(_BATTER_MARKETS) | set(_PITCHER_MARKETS)
    # 3 F5 + 16 props = 19 credits per event, ~285 a slate. Stolen bases is the
    # sixteenth: one more credit per event, ~450 a month on a 100k plan.
    assert len(client.event_markets()) == 19


def test_prop_override_is_filtered_to_known_markets() -> None:
    client = OddsAPIClient("k", prop_markets=("batter_hits", "batter_smiles"), include_f5=False)
    assert client.event_markets() == ["batter_hits"]


def test_per_event_loop_stops_before_draining_the_plan() -> None:
    """Reserve is checked before each event, so a slate cannot overrun it."""
    slate = _slate(3)
    cost = len(OddsAPIClient("k").event_markets())
    client = OddsAPIClient("k", min_credits=200)
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        if "/events/" in url:
            n = len([c for c in calls if "/events/" in c])
            client.credits_remaining = 200 + cost - cost * n
            return {"id": "e", "bookmakers": []}
        client.credits_remaining = 200 + cost
        return _bulk(slate)

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    # Exactly one event fits above the reserve; the next would break it.
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

    with patch("mlb_engine.data.http.get", return_value=resp) as get:
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

    with patch("mlb_engine.data.http.get", return_value=resp):
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

    with patch("mlb_engine.data.http.get", return_value=resp) as get:
        client._get_json("https://x/odds", markets="h2h")
        client._get_json("https://x/odds", markets="h2h")

    assert get.call_count == 2


def _dated_slate() -> Slate:
    """Two games, both with a scheduled first pitch, unlike ``_slate``."""
    slate = _slate(2)
    for i, g in enumerate(slate.games):
        g.game_datetime_utc = f"2026-07-28T2{i}:10:00Z"
    return slate


def test_a_later_meeting_of_the_same_clubs_is_not_priced() -> None:
    """A series puts the same two teams on the vendor's board three nights
    running, and every one of them matches the slate's name map. Those extra
    events bill per market and come back empty of props, because the books have
    not posted tomorrow's yet."""
    slate = _dated_slate()
    board = _bulk(slate)
    for i, g in enumerate(slate.games):
        board[i]["commence_time"] = g.game_datetime_utc
        board.append(
            {
                "id": f"tomorrow{i}",
                "home_team": g.home.name,
                "away_team": g.away.name,
                "commence_time": f"2026-07-29T2{i}:10:00Z",
                "bookmakers": [],
            }
        )

    client = OddsAPIClient("k")
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        return {"id": "e", "bookmakers": []} if "/events/" in url else board

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    per_event = [c for c in calls if "/events/" in c]
    assert len(per_event) == 2
    assert not any("tomorrow" in c for c in per_event)


def test_a_doubleheader_nightcap_is_still_priced() -> None:
    """Both ends are on the slate in their own right, hours rather than a day
    apart, so the date filter must not mistake the second for tomorrow."""
    slate = _slate(1)
    slate.games[0].game_datetime_utc = "2026-07-28T17:10:00Z"
    nightcap = Game(
        game_pk=99,
        game_date=datetime.date(2026, 7, 28),
        game_datetime_utc="2026-07-28T21:40:00Z",
        status="Preview",
        venue=Venue(venue_id=1, name="x"),
        home=slate.games[0].home,
        away=slate.games[0].away,
    )
    slate.games.append(nightcap)

    board = [
        {
            "id": f"evt{i}",
            "home_team": g.home.name,
            "away_team": g.away.name,
            "commence_time": g.game_datetime_utc,
            "bookmakers": [],
        }
        for i, g in enumerate(slate.games)
    ]

    client = OddsAPIClient("k")
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        return {"id": "e", "bookmakers": []} if "/events/" in url else board

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    assert len([c for c in calls if "/events/" in c]) == 2


def test_an_event_without_a_start_time_is_kept() -> None:
    """The vendor omitting a commence stamp should cost a wasted credit at worst,
    never a missing price."""
    slate = _dated_slate()
    board = _bulk(slate)  # no commence_time on any row

    client = OddsAPIClient("k")
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        return {"id": "e", "bookmakers": []} if "/events/" in url else board

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    assert len([c for c in calls if "/events/" in c]) == 2


def test_the_free_event_fallback_also_drops_later_dates() -> None:
    """The props path must not re-import tomorrow's games when the bulk board
    fails and event ids come from /events instead."""
    slate = _dated_slate()
    listing = [
        {
            "id": f"evt{i}",
            "home_team": g.home.name,
            "away_team": g.away.name,
            "commence_time": g.game_datetime_utc,
        }
        for i, g in enumerate(slate.games)
    ]
    listing.append(
        {
            "id": "tomorrow",
            "home_team": slate.games[0].home.name,
            "away_team": slate.games[0].away.name,
            "commence_time": "2026-07-29T20:10:00Z",
        }
    )

    client = OddsAPIClient("k")
    calls: list[str] = []

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        if "/events/" in url:
            return {"id": "e", "bookmakers": []}
        if url.endswith("/events"):
            return listing
        return None  # the bulk board fails

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    per_event = [c for c in calls if "/events/" in c]
    assert len(per_event) == 2
    assert not any("tomorrow" in c for c in per_event)


def test_f5_can_be_dropped_from_the_request() -> None:
    client = OddsAPIClient("k", include_f5=False)
    assert client.event_markets() == list(DEFAULT_PROP_MARKETS)


def test_a_failed_game_board_still_prices_the_props() -> None:
    """Ten of twenty audited slates lost every prop because one failed bulk call
    returned {} for the whole fetch. Event ids cost nothing, so the per-event
    loop has to survive a dead board."""
    slate = _slate(2)
    calls: list[str] = []
    client = OddsAPIClient("k")

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        if url.endswith("/odds"):
            return None  # the bulk board is down
        if url.endswith("/events"):
            return _bulk(slate)
        return {"id": "e", "bookmakers": []}

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    assert any(u.endswith("/events") for u in calls)
    assert len([c for c in calls if "/events/" in c]) == 2


def test_the_free_event_list_is_only_a_fallback() -> None:
    """It costs nothing but it is still a request; a healthy board must not
    trigger it."""
    slate = _slate(2)
    calls: list[str] = []
    client = OddsAPIClient("k")

    def fake_get(url: str, **params: str) -> object:
        calls.append(url)
        return {"id": "e", "bookmakers": []} if "/events/" in url else _bulk(slate)

    client._get_json = fake_get  # type: ignore[method-assign]
    client.fetch(slate)

    assert not any(u.endswith("/events") for u in calls)


def test_a_total_prop_failure_is_logged_not_swallowed() -> None:
    """A silent zero-prop slate is what let this run for twenty slates: the card
    still builds, so nobody notices until the audit grades at an assumed price."""
    slate = _slate(2)
    client = OddsAPIClient("k")

    def fake_get(url: str, **params: str) -> object:
        return None if "/events/" in url else _bulk(slate)

    client._get_json = fake_get  # type: ignore[method-assign]
    with patch("mlb_engine.data.oddsapi.log") as logger:
        client.fetch(slate)

    assert logger.error.called


def test_total_bases_and_hrr_prices_reach_the_quote_keys() -> None:
    """Mapping the market is only half of it; the selection string has to match
    the one the ledger already grades (``Name TB o1.5``)."""
    slate = _slate(1)
    client = OddsAPIClient("k", include_f5=False)
    board = {
        "id": "evt0",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "outcomes": [
                            {"name": "Over", "description": "Cam Smith",
                             "point": 1.5, "price": -115},
                            {"name": "Under", "description": "Cam Smith",
                             "point": 1.5, "price": -105},
                        ],
                    },
                    {
                        "key": "batter_hits_runs_rbis",
                        "outcomes": [
                            {"name": "Over", "description": "Cam Smith",
                             "point": 1.5, "price": 120},
                            {"name": "Under", "description": "Cam Smith",
                             "point": 1.5, "price": -140},
                        ],
                    },
                ],
            }
        ],
    }

    def fake_get(url: str, **params: str) -> object:
        return board if "/events/" in url else _bulk(slate)

    client._get_json = fake_get  # type: ignore[method-assign]
    quotes = client.fetch(slate)

    matchup = next(iter(quotes))[0]
    tb = quotes[(matchup, "batter_tb", "Cam Smith TB o1.5")]
    hrr = quotes[(matchup, "batter_hrr", "Cam Smith H+R+RBI o1.5")]
    assert tb[0].american == -115
    # The under is what makes the over's no-vig probability computable, which is
    # the whole point of capturing a market we never bet.
    assert tb[0].opposite_american == -105
    assert hrr[0].american == 120
    assert hrr[0].opposite_american == -140
