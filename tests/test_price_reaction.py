from datetime import datetime, timedelta, timezone
from pathlib import Path

from mlb_engine.audit.reaction import (
    BASELINE,
    LINEUP,
    ReactionRow,
    append_rows,
    implied,
    load_rows,
    render_summary,
    rows_from_quotes,
    summarize,
)
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.market.ev import MarketQuote
from mlb_engine.schemas import Game, Slate, TeamGameInfo, Venue

T0 = datetime(2026, 9, 22, 20, 0, tzinfo=timezone.utc)
POSTED = (T0 + timedelta(minutes=50)).isoformat()


def _quotes(dk: float, mgm: float | None = -110.0) -> dict:
    q = {("MIN @ CLE", "batter_h", "Joe Hitter o0.5"): [MarketQuote(book="draftkings", american=dk)]}
    if mgm is not None:
        q[("MIN @ CLE", "batter_h", "Joe Hitter o0.5")].append(MarketQuote(book="betmgm", american=mgm))
    return q


def _rows(quotes: dict, event: str, minutes: float, side: str = "") -> list[ReactionRow]:
    now = T0 + timedelta(minutes=minutes)
    return rows_from_quotes(
        quotes, slate_date="2026-09-22", game_pk=700, event=event,
        side=side, event_at=POSTED if event == LINEUP else "", now=now,
    )


def test_rows_carry_minutes_since_posting_and_round_trip_the_csv(tmp_path: Path) -> None:
    rows = _rows(_quotes(-120), LINEUP, 55.4, side="home")
    assert rows[0].minutes_since == 5.4 and rows[0].event_at == POSTED
    base = _rows(_quotes(-120), BASELINE, 10)
    assert base[0].minutes_since is None and base[0].event_at == ""
    path = tmp_path / "price_reaction_2026-09-22.csv"
    append_rows(path, base)
    append_rows(path, rows)
    back = load_rows(path)
    assert back == base + rows


def test_move_is_read_against_the_last_baseline_before_the_posting() -> None:
    rows = (
        _rows(_quotes(-110), BASELINE, 0)  # 52.4%
        + _rows(_quotes(-130), BASELINE, 40)  # 56.5%, the last one before the post at 50
        + _rows(_quotes(-130), LINEUP, 50, "home")  # +0m: unmoved
        + _rows(_quotes(-160, mgm=None), LINEUP, 65, "home")  # +15m: DK moved, MGM pulled
        + _rows(_quotes(-110), BASELINE, 70)  # after the post; must not be the reference
    )
    stats = summarize(rows, threshold_pts=1.0)
    at0 = stats[("draftkings", "batter_h", 0)]
    assert (at0.n, at0.moved, at0.pulled) == (1, 0, 0) and at0.mean_abs == 0.0
    at15 = stats[("draftkings", "batter_h", 15)]
    assert (at15.n, at15.moved, at15.pulled) == (1, 1, 0)
    assert abs(at15.mean_abs - 100 * (implied(-160) - implied(-130))) < 1e-9
    mgm15 = stats[("betmgm", "batter_h", 15)]
    assert (mgm15.n, mgm15.pulled, mgm15.mean_abs) == (1, 1, None)
    assert ("draftkings", "batter_h", 30) not in stats  # never looked at
    text = render_summary(stats, threshold_pts=1.0)
    assert "all books +15m: 2 lines, 1 pulled, 100% moved" in text


def test_capture_off_every_mark_is_not_read_as_one() -> None:
    rows = _rows(_quotes(-110), BASELINE, 0) + _rows(_quotes(-150), LINEUP, 60, "home")  # +10m
    assert summarize(rows) == {}


def test_no_baseline_before_the_posting_means_no_reading() -> None:
    rows = _rows(_quotes(-110), BASELINE, 55) + _rows(_quotes(-150), LINEUP, 50, "home")
    assert summarize(rows) == {}
    assert "No lineup-posting captures" in render_summary({}, threshold_pts=1.0)


def _slate() -> Slate:
    def team(ab: str, name: str, home: bool) -> TeamGameInfo:
        return TeamGameInfo(team_id=1, name=name, abbrev=ab, is_home=home, probable_pitcher=None, lineup=[])

    g = Game(
        game_pk=700, game_date=T0.date(), game_datetime_utc="2026-09-22T23:10:00Z",
        status="Scheduled", venue=Venue(venue_id=1, name="Park"),
        home=team("CLE", "Cleveland Guardians", True), away=team("MIN", "Minnesota Twins", False),
    )
    g2 = g.model_copy(update={"game_pk": 701, "home": team("NYY", "New York Yankees", True)})
    return Slate(slate_date=T0.date(), games=[g, g2])


def test_fetch_game_props_prices_only_the_asked_game(monkeypatch) -> None:
    client = OddsAPIClient("key", prop_markets=("batter_hits",))
    calls: list[tuple[str, dict]] = []

    def fake(url: str, **params: str) -> object:
        calls.append((url, params))
        if url.endswith("/events"):
            return [
                {"id": "E1", "home_team": "Cleveland Guardians", "away_team": "Minnesota Twins",
                 "commence_time": "2026-09-22T23:10:00Z"},
                {"id": "E2", "home_team": "New York Yankees", "away_team": "Minnesota Twins",
                 "commence_time": "2026-09-22T23:10:00Z"},
            ]
        assert url.endswith("/events/E1/odds")
        return {"bookmakers": [{"key": "draftkings", "markets": [{"key": "batter_hits", "outcomes": [
            {"name": "Over", "description": "Joe Hitter", "point": 0.5, "price": -120},
            {"name": "Under", "description": "Joe Hitter", "point": 0.5, "price": -105},
        ]}]}]}

    monkeypatch.setattr(client, "_get_json", fake)
    monkeypatch.setattr("mlb_engine.data.oddsapi.datetime", _Frozen)
    quotes = client.fetch_game_props(_slate(), _slate().games[0])
    assert quotes is not None and len(quotes) == 2
    assert {k[0] for k in quotes} == {"MIN @ CLE"}
    assert {q.book for qs in quotes.values() for q in qs} == {"draftkings"}
    assert [u for u, _ in calls] == [
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/events",
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/E1/odds",
    ]
    assert calls[1][1] == {"markets": "batter_hits"}


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return T0 if tz is None else T0.astimezone(tz)


def test_fetch_game_props_says_none_when_the_game_is_off_the_board(monkeypatch) -> None:
    client = OddsAPIClient("key", prop_markets=("batter_hits",))
    monkeypatch.setattr(client, "_get_json", lambda url, **p: [])
    assert client.fetch_game_props(_slate(), _slate().games[0]) is None
