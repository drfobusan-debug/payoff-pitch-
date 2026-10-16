"""Historical close puller: grid, cost plan before spend, resumable pull, offline export."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from engine_runtime.odds_client import HIST_MULT
from nhl_engine import history
from nhl_engine.cli import main
from nhl_engine.schemas import GameResult, PeriodScore

DAY = Date(2025, 1, 11)
STARTS = ["2025-01-11T23:00:00Z", "2025-01-11T23:04:00Z", "2025-01-12T01:00:00Z"]


def _event(eid: str, start: str, home: str, away: str, home_ml: float = -150) -> dict[str, object]:
    def book(key: str) -> dict[str, object]:
        return {
            "key": key,
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": home, "price": home_ml},
                        {"name": away, "price": 130},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 6.5, "price": -105},
                        {"name": "Under", "point": 6.5, "price": -115},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": home, "point": -1.5, "price": 170},
                        {"name": away, "point": 1.5, "price": -200},
                    ],
                },
            ],
        }

    return {
        "id": eid,
        "commence_time": start,
        "home_team": home,
        "away_team": away,
        "bookmakers": [book("draftkings"), book("fanduel")],
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def historical_board(
        self, markets: tuple[str, ...], *, at: str, pass_id: str = "", **params: str
    ) -> tuple[str, list[dict[str, object]]]:
        self.calls.append(at)
        evs = [
            _event("e1", STARTS[0], "Buffalo Sabres", "Boston Bruins"),
            _event("e2", STARTS[1], "Toronto Maple Leafs", "Ottawa Senators"),
            _event("e3", STARTS[2], "Los Angeles Kings", "Anaheim Ducks"),
        ]
        return at, evs


def test_close_times_one_per_grid_cell() -> None:
    assert history.close_times(STARTS) == ["2025-01-11T22:50:00Z", "2025-01-12T00:50:00Z"]


def test_plan_costs_before_spend() -> None:
    p = history.plan({DAY.isoformat(): STARTS})
    assert p["snapshots"] == 2
    assert p["credits_per_snapshot"] == HIST_MULT * 3
    assert p["credits"] == 2 * HIST_MULT * 3


def test_pull_filters_window_and_resumes(tmp_path: Path) -> None:
    client = FakeClient()
    c = history.pull_day(client, tmp_path, DAY, STARTS)
    assert c["snapshots"] == 2 and c["games"] == 3 and len(client.calls) == 2
    man = history.read_manifest(tmp_path, DAY)
    assert [m["games"] for m in man] == [["BOS @ BUF", "OTT @ TOR"], ["ANA @ LAK"]]
    again = history.pull_day(client, tmp_path, DAY, STARTS)
    assert again["skipped"] == 2 and again["snapshots"] == 0 and len(client.calls) == 2


def test_export_joins_close_to_final(tmp_path: Path) -> None:
    history.pull_day(FakeClient(), tmp_path, DAY, STARTS)
    starts = {"BOS @ BUF": STARTS[0], "OTT @ TOR": STARTS[1], "ANA @ LAK": STARTS[2]}
    res = GameResult(
        nhl_game_id=1,
        game_date=DAY,
        away="BOS",
        home="BUF",
        state="OFF",
        periods=(PeriodScore(1, "REG", 2, 4),),
        decided="REG",
    )
    rows = history.export_day(tmp_path, DAY, starts, {"BOS @ BUF": res})
    assert len(rows) == 3
    buf = next(r for r in rows if r["home"] == "BUF")
    assert buf["close_at"] == "2025-01-11T22:50:00Z"
    assert 0.5 < float(str(buf["close_home_ml"])) < 0.65
    assert buf["close_total"] == 6.5 and buf["final_home"] == 4
    lak = next(r for r in rows if r["home"] == "LAK")
    assert lak["final_home"] == "" and lak["close_at"] == "2025-01-12T00:50:00Z"
    assert history.write_csv(rows, tmp_path / "out.csv") == 3


def test_cli_plan_and_pull_without_yes_spend_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from nhl_engine import cli
    from nhl_engine.schemas import Game

    games = [Game(game_date=DAY, home="BUF", away="BOS", start_utc=STARTS[0], nhl_game_id=1)]
    monkeypatch.setattr(cli, "_history_schedule", lambda c, s, e: {DAY: games})
    monkeypatch.setattr(
        cli.OddsClient, "historical_board", lambda *a, **k: pytest.fail("spent credits")
    )
    for action in ("plan", "pull"):
        assert main(["history", action, "--start", "2025-01-11", "--end", "2025-01-11"]) == 0
        out = capsys.readouterr()
        assert f"1 snapshots x {HIST_MULT * 3} = {HIST_MULT * 3} credits" in out.out
    assert "re-run with --yes" in out.err
