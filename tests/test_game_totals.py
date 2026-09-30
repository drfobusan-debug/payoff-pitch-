from __future__ import annotations

from datetime import date as Date
from types import SimpleNamespace

import requests

from mlb_engine.data import game_totals
from mlb_engine.data.game_totals import board_totals, parse_espn, posted_totals
from mlb_engine.market.ev import MarketQuote

CFG = SimpleNamespace()
SLATE = SimpleNamespace(slate_date=Date(2026, 9, 19), games=[])


def test_espn_scoreboard_totals_use_engine_team_codes() -> None:
    payload = {
        "events": [
            {"competitions": [{
                "competitors": [
                    {"homeAway": "away", "team": {"abbreviation": "NYY"}},
                    {"homeAway": "home", "team": {"abbreviation": "ARI"}},
                ],
                "odds": [{"provider": {"name": "DraftKings"}, "overUnder": 7.5}],
            }]},
            {"competitions": [{
                "competitors": [
                    {"homeAway": "away", "team": {"abbreviation": "DET"}},
                    {"homeAway": "home", "team": {"abbreviation": "CHW"}},
                ],
                "odds": [{"provider": {"name": "DraftKings"}}],
            }]},
        ]
    }
    assert parse_espn(payload) == {"NYY @ AZ": 7.5}
    assert parse_espn("not json") == {}


def test_odds_api_board_total_is_the_over_nearest_minus_110() -> None:
    quotes = {
        ("SEA @ COL", "game_total", "Over 10.5"): [MarketQuote("draftkings", -108.0)],
        ("SEA @ COL", "game_total", "Over 11.5"): [MarketQuote("draftkings", 135.0)],
        ("SEA @ COL", "game_total", "Under 10.5"): [MarketQuote("draftkings", -112.0)],
        ("SEA @ COL", "game_ml", "SEA"): [MarketQuote("draftkings", -130.0)],
    }
    assert board_totals(quotes) == {"SEA @ COL": 10.5}


def test_sources_fill_only_what_earlier_ones_left_and_outages_are_skipped() -> None:
    calls: list[str] = []

    def first(cfg: object, slate: object) -> dict[str, float]:
        calls.append("first")
        raise requests.ConnectionError("down")

    def second(cfg: object, slate: object) -> dict[str, float]:
        calls.append("second")
        return {"KC @ PIT": 8.0, "DET @ CWS": 8.0}

    def third(cfg: object, slate: object) -> dict[str, float]:
        calls.append("third")
        return {"KC @ PIT": 8.5, "SF @ LAD": 7.5}

    got = posted_totals(
        CFG, SLATE, {"KC @ PIT", "SF @ LAD"},
        sources=(("a", first), ("b", second), ("c", third)),
    )
    assert got == {"KC @ PIT": 8.0, "SF @ LAD": 7.5}
    assert calls == ["first", "second", "third"]


def test_no_source_is_asked_once_nothing_is_missing() -> None:
    def never(cfg: object, slate: object) -> dict[str, float]:
        raise AssertionError("asked with nothing missing")

    assert posted_totals(CFG, SLATE, set(), sources=(("a", never),)) == {}
    assert game_totals.SOURCES[0][0] == "Odds API"
