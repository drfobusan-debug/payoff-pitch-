import json
from datetime import date
from pathlib import Path

from nba_engine.data import teamnames
from nba_engine.data.capture import ALL_MARKETS, EVENT_MARKETS, GAME_MARKETS, MARKET_MAP
from nba_engine.data.oddsapi import event_rows, games_from, redact, slate_date_of

FIX = Path(__file__).parent / "fixtures"


def _event() -> dict:
    return json.loads((FIX / "oddsapi_hist_event_20251210.json").read_text())["data"]


def test_market_map_covers_the_eleven_requested_keys():
    assert len(ALL_MARKETS) == 11
    assert set(GAME_MARKETS) | set(EVENT_MARKETS) == set(ALL_MARKETS)
    assert all(MARKET_MAP[k][1] == "reg_only" for k in ("h2h_h1", "spreads_h1", "totals_h1"))
    assert all(MARKET_MAP[k][1] == "incl_ot" for k in GAME_MARKETS)


def test_team_names_map_and_unknowns_are_refused():
    assert teamnames.code_for("Los Angeles Lakers") == "LAL"
    assert teamnames.code_for("LA Clippers") == "LAC"
    assert teamnames.code_for("Seattle SuperSonics") is None
    assert teamnames.canonical("SA") == "SAS"
    assert teamnames.canonical("GS") == "GSW"


def test_late_tip_belongs_to_the_eastern_slate_date():
    assert slate_date_of("2025-12-11T03:00:00Z") == date(2025, 12, 10)


def test_event_rows_pair_both_sides_of_every_market():
    event = _event()
    games = games_from([event], date(2025, 12, 10))
    assert len(games) == 1
    game = games[0]
    assert (game.away, game.home, game.event_id) == ("SAS", "LAL", event["id"])
    rows = event_rows(event, game, "2025-12-11T02:50:38Z")
    assert {r.market for r in rows} == {MARKET_MAP[k][0] for k in ALL_MARKETS}
    assert all(r.opposite_american is not None for r in rows)

    ats = {r.side: r for r in rows if r.market == "game_ats" and r.book == "fanduel"}
    home_line, away_line = ats["LAL"].line, ats["SAS"].line
    assert home_line is not None and away_line is not None and home_line == -away_line
    assert ats["LAL"].opposite_american == ats["SAS"].american

    pts = [r for r in rows if r.market == "pl_pts" and r.book == "fanduel"]
    assert {r.side for r in pts} == {"over", "under"}
    assert all(r.entity for r in pts)
    for r in pts:
        partner = next(
            p for p in pts if p.entity == r.entity and p.side != r.side and p.line == r.line
        )
        assert r.opposite_american == partner.american


def test_a_missing_partner_is_not_invented():
    event = _event()
    book = event["bookmakers"][0]
    market = next(m for m in book["markets"] if m["key"] == "totals")
    market["outcomes"] = [o for o in market["outcomes"] if o["name"] == "Over"]
    game = games_from([event], date(2025, 12, 10))[0]
    rows = [
        r
        for r in event_rows(event, game, "t")
        if r.market == "game_total" and r.book == book["key"]
    ]
    assert len(rows) == 1 and rows[0].opposite_american is None


def test_redact_hides_the_key():
    assert "SECRET" not in redact("https://x/?apiKey=SECRET&a=1", "SECRET")
