"""Offline fixtures for the retrospective batting study."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_engine.data.statcast import StatcastRepository
from scripts.worksheet_batting_backtest import (
    bootstrap,
    fetch_inputs,
    opposite_quote,
    payout,
    quote,
    schedule_games,
)


def test_missing_inputs_are_cached_without_live_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Response:
        text = '{"dates": []}'

        def raise_for_status(self) -> None:
            pass

    def get(*args: object, **kwargs: object) -> Response:
        calls.append("schedule")
        return Response()

    def load_range(self: StatcastRepository, start: date, end: date) -> pd.DataFrame:
        assert start == end == date(2026, 3, 25)
        calls.append("statcast")
        return pd.DataFrame({"game_date": ["2026-03-25"], "pitcher": [123]})

    monkeypatch.setattr("scripts.worksheet_batting_backtest.requests.get", get)
    monkeypatch.setattr(StatcastRepository, "load_range", load_range)
    audit = tmp_path / "audit"
    fetch_inputs(date(2026, 3, 25), audit)
    assert calls == ["schedule", "statcast"]
    assert (audit / "schedule_2026.json").read_text() == '{"dates": []}'
    assert len(pd.read_csv(audit / "statcast_csv/2026-03-25.csv")) == 1
    fetch_inputs(date(2026, 3, 25), audit)
    assert calls == ["schedule", "statcast"]


def test_total_quote_keeps_both_sides_on_the_same_line() -> None:
    rows = [
        {"market": "game_total", "selection": "Over 8.0",
         "american": -116, "no_vig_prob": 0.52},
        {"market": "game_total", "selection": "Under 8.0",
         "american": 101, "no_vig_prob": 0.48},
        {"market": "game_total", "selection": "Over 8.5",
         "american": 102, "no_vig_prob": 0.48},
        {"market": "game_total", "selection": "Under 8.5",
         "american": -112, "no_vig_prob": 0.52},
        {"market": "game_total", "selection": "Over 9.0",
         "american": 110, "no_vig_prob": 0.45},
    ]
    over = quote(rows, "game_total", "Over ")
    assert over is not None
    under = opposite_quote(rows, "game_total", over["line"])
    assert under is not None and under["line"] == over["line"] == 8.5
    assert round(over["prob"] + under["prob"], 6) == 1
    assert payout(over["american"], 0.5) == 0
    assert payout(under["american"], 1) > 0


def test_doubleheader_not_duplicated_on_matchup_label() -> None:
    game = {"gameType": "R", "status": {"abstractGameState": "Final"},
            "gamePk": 1, "teams": {
                "away": {"team": {"abbreviation": "NYY"}, "score": 3},
                "home": {"team": {"abbreviation": "BOS"}, "score": 2},
            }, "linescore": {"innings": [
                {"away": {"runs": 0}, "home": {"runs": 0}} for _ in range(9)
            ]}}
    schedule = {"dates": [{"date": "2026-08-04", "games": [
        game, game | {"gamePk": 2},
    ]}]}
    assert schedule_games(schedule, date(2026, 8, 4), date(2026, 8, 4)) == []


def test_pitched_partial_removes_confounding_and_is_reproducible() -> None:
    rng = np.random.default_rng(71)
    sp = rng.normal(size=120)
    bp = rng.normal(size=120)
    bat = sp + rng.normal(scale=0.1, size=120)
    runs = sp + bp + rng.normal(scale=0.2, size=120)
    raw = bootstrap(bat, runs, draws=2000)
    controlled = bootstrap(bat, runs, np.column_stack([sp, bp]), draws=2000)
    assert raw["effect"] > 0.5
    assert abs(controlled["effect"]) < 0.2
    assert controlled == bootstrap(bat, runs, np.column_stack([sp, bp]), draws=2000)
