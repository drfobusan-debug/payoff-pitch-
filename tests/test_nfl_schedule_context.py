"""The context the odds feed does not carry, and what happens without it.

The two situational terms with evidence behind them -- wind on the total,
divisional on the total -- read fields that a live game had no way of having:
The Odds API sells prices. They fired in historical replay and nowhere else.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone

import pandas as pd
import pytest

from mlb_engine.data.openmeteo import VenueWeather
from nfl_engine.data import schedule, weather
from nfl_engine.features.adjustments import adjust
from nfl_engine.pipeline import situation_of
from nfl_engine.schemas import Game, GameEnvironment, TeamGameInfo

SEASON, WEEK = 2026, 1


def _game(home: str = "BUF", away: str = "NYJ", **kwargs: object) -> Game:
    return Game(
        game_id="evt1",
        season=SEASON,
        week=WEEK,
        game_date=Date(2026, 9, 13),
        kickoff_utc="2026-09-13T17:00:00Z",
        home=TeamGameInfo(name="Buffalo Bills", abbrev=home, is_home=True),
        away=TeamGameInfo(name="New York Jets", abbrev=away, is_home=False),
        **kwargs,
    )


def _schedule_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": SEASON,
        "week": WEEK,
        "home_team": "BUF",
        "away_team": "NYJ",
        "location": "Home",
        "roof": "outdoors",
        "surface": "grass",
        "div_game": 1,
        "home_rest": 7,
        "away_rest": 4,
        "stadium_id": "BUF00",
    }
    row.update(overrides)
    return row


@pytest.fixture
def no_weather(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weather is a separate feed; these tests are about the schedule."""
    monkeypatch.setattr(weather, "fetch_venue_weather", lambda points, **kw: {})


def test_a_live_game_gets_the_divisional_flag_and_rest_off_the_schedule(
    monkeypatch: pytest.MonkeyPatch, no_weather: None
) -> None:
    monkeypatch.setattr(schedule.nflverse, "games", lambda: _schedule_frame([_row()]))

    game = schedule.enrich([_game()])[0]

    assert game.div_game is True
    assert (game.home_rest, game.away_rest) == (7, 4)
    assert game.env.roof == "outdoors"
    # And the flag reaches the number: a divisional game runs under.
    assert adjust(situation_of(game)).total_points == pytest.approx(-0.66, abs=0.01)


def test_a_game_the_schedule_does_not_have_is_priced_as_before(
    monkeypatch: pytest.MonkeyPatch, no_weather: None
) -> None:
    monkeypatch.setattr(schedule.nflverse, "games", lambda: _schedule_frame([_row(home_team="KC")]))

    game = schedule.enrich([_game()])[0]

    assert game.div_game is None
    assert adjust(situation_of(game)).total_points == 0.0


def test_an_unreachable_schedule_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, no_weather: None
) -> None:
    monkeypatch.setattr(schedule.nflverse, "games", lambda: pd.DataFrame())

    game = schedule.enrich([_game()])[0]

    assert game.div_game is None and game.home_rest is None


def test_a_neutral_site_is_read_off_the_schedule(
    monkeypatch: pytest.MonkeyPatch, no_weather: None
) -> None:
    monkeypatch.setattr(
        schedule.nflverse, "games", lambda: _schedule_frame([_row(location="Neutral")])
    )

    game = schedule.enrich([_game()])[0]

    assert game.env.neutral_site is True


def test_a_retractable_roof_is_filled_from_what_that_venue_has_done(
    monkeypatch: pytest.MonkeyPatch, no_weather: None
) -> None:
    """Blank is a game-day decision not yet taken, and is not "outdoors"."""
    played = [
        {**_row(season=2025, week=w, roof="closed", stadium_id="PHO00"), "home_team": "ARI"}
        for w in (1, 2)
    ]
    upcoming = _row(roof="", stadium_id="PHO00", home_team="ARI")
    monkeypatch.setattr(schedule.nflverse, "games", lambda: _schedule_frame([*played, upcoming]))

    game = schedule.enrich([_game(home="ARI")])[0]

    assert game.env.roof == "closed"
    assert game.env.is_indoors()


def test_a_venue_with_no_history_stays_unknown_rather_than_outdoors(
    monkeypatch: pytest.MonkeyPatch, no_weather: None
) -> None:
    monkeypatch.setattr(
        schedule.nflverse,
        "games",
        lambda: _schedule_frame([_row(roof="", stadium_id="NEW00")]),
    )

    game = schedule.enrich([_game()])[0]

    assert game.env.roof is None
    # Unknown means the wind term declines to fire, not that it assumes calm.
    assert adjust(situation_of(game)).total_points == pytest.approx(-0.66, abs=0.01)


def test_the_kickoff_forecast_reaches_the_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schedule.nflverse, "games", lambda: _schedule_frame([_row()]))
    asked: dict[str, tuple[float, float, datetime]] = {}

    def fake_fetch(points, **kwargs):
        asked.update(points)
        return {
            key: VenueWeather(wind_mph=18.5, gust_mph=None, precipitation=0.0, temperature_f=44.0)
            for key in points
        }

    monkeypatch.setattr(weather, "fetch_venue_weather", fake_fetch)

    game = schedule.enrich([_game()])[0]

    assert asked["NYJ @ BUF"] == (
        *weather.VENUE_COORDS["BUF00"],
        datetime(2026, 9, 13, 17, tzinfo=timezone.utc),
    )
    assert game.env.wind_mph == 18.5
    # 10 mph over the league's average outdoor game, at -0.20/mph, plus divisional.
    assert adjust(situation_of(game)).total_points == pytest.approx(-2.66, abs=0.01)


def test_indoor_games_are_never_asked_about(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        schedule.nflverse,
        "games",
        lambda: _schedule_frame([_row(roof="dome", stadium_id="DET00")]),
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        weather, "fetch_venue_weather", lambda points, **kw: calls.append(points) or {}
    )

    game = schedule.enrich([_game()])[0]

    assert calls == []
    assert game.env.wind_mph is None


def test_weather_failing_leaves_the_slate_priceable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schedule.nflverse, "games", lambda: _schedule_frame([_row()]))
    monkeypatch.setattr(weather, "fetch_venue_weather", lambda points, **kw: {})

    game = schedule.enrich([_game()])[0]

    assert game.env.wind_mph is None
    assert game.div_game is True


def test_a_reported_wind_is_not_overwritten_by_a_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(schedule.nflverse, "games", lambda: _schedule_frame([_row()]))
    monkeypatch.setattr(
        weather,
        "fetch_venue_weather",
        lambda points, **kw: pytest.fail("asked about a game that already had a reading"),
    )

    known = _game(env=GameEnvironment(roof="outdoors", wind_mph=6.0))
    assert schedule.enrich([known])[0].env.wind_mph == 6.0
