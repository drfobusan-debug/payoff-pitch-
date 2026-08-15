"""What the NFL odds client asks for, and what it does with the answer."""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import pytest

from nfl_engine.data import oddsapi
from nfl_engine.data.oddsapi import OddsAPIClient


class _Resp(NamedTuple):
    payload: object
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


def _event(home: str = "Kansas City Chiefs", away: str = "Buffalo Bills") -> dict:
    return {
        "id": "abc123",
        "commence_time": "2026-09-13T17:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": -160},
                            {"name": away, "price": 140},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home, "price": -105, "point": -3.0},
                            {"name": away, "price": -115, "point": 3.0},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 47.5},
                            {"name": "Under", "price": -110, "point": 47.5},
                        ],
                    },
                ],
            },
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home, "price": -120, "point": -2.5},
                            {"name": away, "price": 100, "point": 2.5},
                        ],
                    }
                ],
            },
        ],
    }


def _fetch(monkeypatch: pytest.MonkeyPatch, payload: object) -> tuple:
    calls: list[dict] = []

    def fake_get(url: str, **kw: object) -> _Resp:
        calls.append({"url": url, "params": kw.get("params")})
        return _Resp(payload, {"x-requests-remaining": "87000"})

    monkeypatch.setattr(oddsapi.http, "get", fake_get)
    client = OddsAPIClient("secret-key")
    slate, board = client.fetch_board(season=2026, week=2, first_day=date(2026, 9, 10))
    return slate, board, calls, client


def test_no_key_means_an_empty_board_not_an_exception():
    """A missing credential degrades to "no prices", never a dead slate."""
    slate, board = OddsAPIClient(None).fetch_board(
        season=2026, week=1, first_day=date(2026, 9, 10)
    )
    assert board == {}
    assert slate.games == []


def test_one_bulk_request_covers_all_three_markets(monkeypatch: pytest.MonkeyPatch):
    _, _, calls, client = _fetch(monkeypatch, [_event()])
    assert len(calls) == 1
    params = calls[0]["params"]
    assert isinstance(params, dict)
    assert params["markets"] == "h2h,spreads,totals"
    assert params["oddsFormat"] == "american"
    assert client.credits_remaining == 87000


def test_the_board_doubles_as_the_schedule(monkeypatch: pytest.MonkeyPatch):
    slate, board, _, _ = _fetch(monkeypatch, [_event()])
    assert [game.matchup() for game in slate.games] == ["BUF @ KC"]
    game = slate.games[0]
    assert game.season == 2026 and game.week == 2
    # Kickoff, not the fetch window's first day: 17:00 UTC is Sunday afternoon.
    assert game.game_date == date(2026, 9, 13)
    assert "BUF @ KC" in board


def test_every_rung_of_the_spread_ladder_is_kept(monkeypatch: pytest.MonkeyPatch):
    """Which side of a 3 the bet is struck on is worth more than most ratings."""
    _, board, _, _ = _fetch(monkeypatch, [_event()])
    odds = board["BUF @ KC"]
    assert odds.spread_ladder() == [-2.5, -3.0]
    assert odds.main_spread() == -2.5
    best = odds.best_spread("KC")
    assert best is not None
    point, quote = best
    # -2.5 is the better number even though -3.0 is the better price.
    assert point == -2.5 and quote.book == "draftkings"


def test_spreads_are_stored_on_the_home_axis(monkeypatch: pytest.MonkeyPatch):
    _, board, _, _ = _fetch(monkeypatch, [_event()])
    sides = board["BUF @ KC"].spreads[-3.0]
    assert set(sides) == {"KC", "BUF"}
    assert sides["KC"][0].american == -105


def test_the_opposite_price_comes_from_the_same_book_and_line(
    monkeypatch: pytest.MonkeyPatch,
):
    """De-vigging needs the real other side; it is never assumed to be -110."""
    _, board, _, _ = _fetch(monkeypatch, [_event()])
    odds = board["BUF @ KC"]
    home_ml = odds.ml["KC"][0]
    assert home_ml.opposite_american == 140
    assert 0.55 < home_ml.no_vig_prob < 0.63
    over = odds.totals[47.5]["over"][0]
    assert over.opposite_american == -110
    assert abs(over.no_vig_prob - 0.5) < 1e-9


def test_an_unmappable_team_is_skipped_not_guessed(monkeypatch: pytest.MonkeyPatch):
    slate, board, _, _ = _fetch(monkeypatch, [_event(away="Toronto Argonauts")])
    assert slate.games == [] and board == {}


def test_a_failed_request_returns_empty_and_hides_the_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import requests

    def boom(url: str, **kw: object) -> _Resp:
        raise requests.RequestException(f"401 for {url}?apiKey=secret-key")

    monkeypatch.setattr(oddsapi.http, "get", boom)
    with caplog.at_level("WARNING"):
        slate, board = OddsAPIClient("secret-key").fetch_board(
            season=2026, week=1, first_day=date(2026, 9, 10)
        )
    assert board == {} and slate.games == []
    assert "secret-key" not in caplog.text


# -- player props: archived, never priced ---------------------------------
def _prop_payload() -> dict:
    return {
        "id": "abc123",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_receptions",
                        "outcomes": [
                            {"name": "Over", "description": "Rashee Rice", "price": -130, "point": 5.5},
                            {"name": "Under", "description": "Rashee Rice", "price": 105, "point": 5.5},
                            {"name": "Over", "description": "Travis Kelce", "price": -115, "point": 4.5},
                            # Kelce's under is missing: it must not borrow Rice's.
                        ],
                    },
                    {
                        "key": "player_anytime_td",
                        "outcomes": [{"name": "Rashee Rice", "price": 145}],
                    },
                ],
            }
        ],
    }


def _prop_rows(monkeypatch: pytest.MonkeyPatch) -> list:
    slate, _, _, client = _fetch(monkeypatch, [_event()])

    def fake_get(url: str, **kw: object) -> _Resp:
        assert "/events/abc123/odds" in url
        return _Resp(_prop_payload(), {})

    monkeypatch.setattr(oddsapi.http, "get", fake_get)
    return client.fetch_props(list(slate.games), captured_at="2026-09-10T18:00:00Z")


def test_prop_quotes_pair_within_a_player_and_a_line(monkeypatch: pytest.MonkeyPatch):
    rows = _prop_rows(monkeypatch)
    paired = {
        (row.player, row.side): row.opposite_american
        for row in rows
        if row.market == "player_receptions"
    }
    assert paired[("Rashee Rice", "over")] == 105
    assert paired[("Rashee Rice", "under")] == -130
    # The unpaired side stays unpaired rather than inheriting another player's.
    assert paired[("Travis Kelce", "over")] is None


def test_prop_rows_carry_the_line_the_player_and_the_moment(monkeypatch: pytest.MonkeyPatch):
    rows = _prop_rows(monkeypatch)
    row = next(r for r in rows if r.player == "Rashee Rice" and r.side == "over")
    assert (row.line, row.book, row.american) == (5.5, "draftkings", -130)
    assert (row.matchup, row.event_id) == ("BUF @ KC", "abc123")
    assert row.captured_at == "2026-09-10T18:00:00Z"
    assert (row.season, row.week) == (2026, 2)


def test_an_anytime_scorer_names_the_player_not_the_side(monkeypatch: pytest.MonkeyPatch):
    rows = [r for r in _prop_rows(monkeypatch) if r.market == "player_anytime_td"]
    assert [(r.player, r.side, r.american, r.line) for r in rows] == [
        ("Rashee Rice", "yes", 145, None)
    ]


def test_props_without_a_key_are_no_request_and_no_rows():
    assert OddsAPIClient(None).fetch_props([]) == []
