"""Marking layer: metric confidence bumps + NPV veto gates."""

from __future__ import annotations

from cfb_engine.config import MarkingParams
from cfb_engine.data.advanced import AdvancedBook, TeamAdvanced, parse_advanced
from cfb_engine.market.confidence import (
    MatchupSignal,
    build_signal,
    confidence_adjustment,
    market_veto,
)


def _team(key: str, **over: float) -> TeamAdvanced:
    base = dict(
        team=key,
        games=12,
        off_ppa=0.2,
        def_ppa=0.1,
        off_success=0.45,
        def_success=0.42,
        off_explosive=1.2,
        def_explosive=1.2,
        off_finishing=4.3,
        def_finishing=4.3,
        havoc=0.15,
        plays_per_game=68.0,
        drives_per_game=12.0,
        turnover_margin_pg=0.0,
    )
    base.update(over)
    return TeamAdvanced(**base)  # type: ignore[arg-type]


def _book(home: TeamAdvanced, away: TeamAdvanced) -> AdvancedBook:
    from cfb_engine.data.advanced import _finalize

    return _finalize({"home": home, "away": away})


PARAMS = MarkingParams()


def test_missing_stats_is_a_noop():
    steps, reasons = confidence_adjustment("game_ml", "home", "win", MatchupSignal(), PARAMS)
    assert steps == 0 and reasons == []
    assert market_veto("game_ats", "home", "cover", -7.0, MatchupSignal(), PARAMS).dropped is False


def test_efficiency_edge_bumps_the_backed_side_and_fades_the_other():
    home = _team("home", off_ppa=0.35, off_success=0.52, havoc=0.22)
    away = _team("away", off_ppa=0.05, off_success=0.38, havoc=0.09)
    sig = build_signal(_book(home, away), "home", "away")
    up, _ = confidence_adjustment("game_ml", "home", "win", sig, PARAMS)
    down, _ = confidence_adjustment("game_ml", "away", "win", sig, PARAMS)
    assert up == 1
    assert down == -1


def test_turnover_luck_vetoes_backing_the_lucky_favorite():
    home = _team("home", turnover_margin_pg=1.2)
    away = _team("away")
    sig = build_signal(_book(home, away), "home", "away")
    # Backing the +TO home team as an ATS favorite (line -7) is a fade candidate.
    veto = market_veto("game_ats", "home", "cover", -7.0, sig, PARAMS)
    assert veto.dropped and veto.gate is not None
    # The underdog side (laying no points) is not vetoed.
    assert market_veto("game_ats", "away", "cover", 7.0, sig, PARAMS).dropped is False


def test_low_scoring_environment_vetoes_the_over():
    home = _team("home", off_ppa=0.02, plays_per_game=58.0)
    away = _team("away", off_ppa=0.01, plays_per_game=57.0)
    # Force league means high so this game reads below average.
    book = AdvancedBook(
        teams={"home": home, "away": away},
        mean_off_ppa=0.25,
        mean_off_explosive=1.4,
        mean_off_finishing=4.3,
        mean_plays_per_game=70.0,
    )
    sig = build_signal(book, "home", "away")
    assert market_veto("game_total", None, "over", 62.0, sig, PARAMS).dropped is True
    assert market_veto("game_total", None, "under", 62.0, sig, PARAMS).dropped is False


def test_parse_advanced_computes_turnover_margin_and_pace():
    rows = [
        {
            "team": "Georgia",
            "offense": {
                "ppa": 0.23,
                "successRate": 0.46,
                "explosiveness": 1.22,
                "pointsPerOpportunity": 4.58,
                "plays": 936,
                "drives": 165,
            },
            "defense": {
                "ppa": 0.11,
                "successRate": 0.40,
                "explosiveness": 1.10,
                "pointsPerOpportunity": 3.9,
                "havoc": {"total": 0.157},
            },
        }
    ]
    season_stats = {
        "georgia": {
            "games": 12.0,
            "interceptions": 13.0,
            "fumblesRecovered": 11.0,
            "passesIntercepted": 8.0,
            "fumblesLost": 7.0,
        }
    }
    book = parse_advanced(rows, season_stats)
    ga = book.get("Georgia")
    assert ga is not None
    assert abs(ga.turnover_margin_pg - (24 - 15) / 12) < 1e-9
    assert abs(ga.plays_per_game - 936 / 12) < 1e-9
    assert abs(ga.net_ppa - (0.23 - 0.11)) < 1e-9
