"""Backfill the per-game close for graded buys that have none, from Odds API history."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pytest

from cfb_engine import cli
from cfb_engine.audit import closefill
from cfb_engine.audit.clv import ClosingQuote, load_closing, save_closing
from cfb_engine.audit.ledger import LedgerEntry, load_ledger, update_ledger
from cfb_engine.config import Config
from cfb_engine.data.oddsapi import OddsAPIClient
from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote
from cfb_engine.market.tiers import Tier
from cfb_engine.schemas import Game, Slate, TeamGameInfo

DAY = date(2026, 9, 5)
KICK = "2026-09-05T16:00:00Z"
MATCHUP = "UGA @ ALA"


def _entry(tier: str = Tier.MODERATE.value, result: str = "win") -> LedgerEntry:
    return LedgerEntry(
        date=DAY.isoformat(),
        matchup=MATCHUP,
        category="Moneyline",
        market="game_ml",
        selection="ALA ML",
        line=None,
        book="b",
        odds=-120.0,
        tier=tier,
        model_prob=0.6,
        ev=0.05,
        result=result,
        pnl=0.83,
        fair_prob=0.53,
    )


def _q(captured: str, american: float = -150.0) -> ClosingQuote:
    return ClosingQuote(american, 0.58, None, captured, KICK)


def test_only_graded_buys_without_a_timely_close_are_missing() -> None:
    entries = [_entry(), _entry(tier=Tier.PASS.value), _entry(result="")]
    assert closefill.missing(entries, lambda _: {}) == {DAY.isoformat(): {MATCHUP}}
    stale = {f"{MATCHUP}|game_ml|ALA": ClosingQuote(-150.0, 0.58, None, "", "")}
    assert closefill.missing(entries, lambda _: stale) == {DAY.isoformat(): {MATCHUP}}
    timely = {f"{MATCHUP}|game_ml|ALA": _q("2026-09-05T15:50:00Z")}
    assert closefill.missing(entries, lambda _: timely) == {}


def test_games_sharing_a_kickoff_share_one_snapshot_ten_minutes_before() -> None:
    kicks = {"A @ B": KICK, "C @ D": KICK, "E @ F": "2026-09-05T19:30:00Z", "G @ H": ""}
    assert closefill.windows(kicks) == {
        "2026-09-05T15:50:00Z": ["A @ B", "C @ D"],
        "2026-09-05T19:20:00Z": ["E @ F"],
    }
    assert closefill.cost(2, "us", lookups=1) == 61


def test_a_timely_close_on_file_is_never_overwritten() -> None:
    key = f"{MATCHUP}|game_ml|ALA"
    kept = {key: _q("2026-09-05T15:55:00Z", -140.0)}
    assert closefill.fill(kept, {key: _q("2026-09-05T15:50:00Z")})[key].american == -140.0
    stale = {key: ClosingQuote(-110.0, 0.5, None, "", "")}
    assert closefill.fill(stale, {key: _q("2026-09-05T15:50:00Z")})[key].american == -150.0
    early = {key: _q("2026-09-05T13:00:00Z")}
    assert closefill.fill({}, early) == {}


def test_the_historical_board_keeps_the_slate_day_and_its_snapshot_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int | None, dict[str, str]]] = []
    client = OddsAPIClient("key")

    def fake(url: str, credit_cost: int | None = None, **params: str) -> object:
        calls.append((url, credit_cost, params))
        return {
            "timestamp": "2026-09-05T15:45:00Z",
            "data": [
                {
                    "id": "1",
                    "home_team": "Alabama Crimson Tide",
                    "away_team": "Georgia Bulldogs",
                    "commence_time": KICK,
                    "bookmakers": [],
                },
                {
                    "id": "2",
                    "home_team": "Texas Longhorns",
                    "away_team": "Oklahoma Sooners",
                    "commence_time": "2026-09-12T19:30:00Z",
                    "bookmakers": [],
                },
            ],
        }

    monkeypatch.setattr(client, "_get_json", fake)
    stamp, slate, board = client.fetch_historical_board(DAY, "2026-09-05T15:50:00Z")
    assert stamp == "2026-09-05T15:45:00Z"
    assert [g.game_id for g in slate.games] == ["1"] and len(board) == 1
    url, cost, params = calls[0]
    assert "/historical/" in url and cost == 30 and params["date"] == "2026-09-05T15:50:00Z"


def _board() -> tuple[Slate, dict[str, GameOdds]]:
    game = Game(
        game_id="1",
        game_date=DAY,
        commence_time_utc=KICK,
        home=TeamGameInfo(name="Alabama", abbrev="ALA", is_home=True),
        away=TeamGameInfo(name="Georgia", abbrev="UGA", is_home=False),
    )
    odds = GameOdds(matchup=game.matchup())
    odds.ml["ALA"] = [MarketQuote(book="b", american=-150, opposite_american=130)]
    odds.ml["UGA"] = [MarketQuote(book="b", american=130, opposite_american=-150)]
    return Slate(slate_date=DAY, games=[game]), {game.matchup(): odds}


class _FakeOdds:
    bought: list[str] = []

    def __init__(self, *a: object, **k: object) -> None:
        self.credits_remaining = 4_000_000

    def available(self) -> bool:
        return True

    def historical_kickoffs(self, day: date) -> dict[str, str]:
        return {MATCHUP: KICK}

    def fetch_historical_board(self, day: date, at: str) -> tuple[str, Slate, dict[str, GameOdds]]:
        self.bought.append(at)
        slate, board = _board()
        return "2026-09-05T15:45:00Z", slate, board


def test_plan_spends_nothing_and_buy_fills_the_close_and_restamps_clv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CFBE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CFBE_STATE_SYNC", "0")
    monkeypatch.setenv("CFBE_LEDGER_FILE", str(tmp_path / "ledger.csv"))
    monkeypatch.setattr(cli, "OddsAPIClient", _FakeOdds)
    cfg = Config()
    update_ledger(cfg.ledger_file, [_entry()], DAY)
    save_closing({"OTHER @ GAME|game_ml|GAME": _q("2026-09-05T15:50:00Z")}, cfg.closing_file(DAY))
    _FakeOdds.bought = []

    plan = argparse.Namespace(date=None, buy=False, lead=10.0)
    assert cli.cmd_close_backfill(cfg, plan) == 0
    assert _FakeOdds.bought == [] and "Plan only" in capsys.readouterr().out

    assert cli.cmd_close_backfill(cfg, argparse.Namespace(date=None, buy=True, lead=10.0)) == 0
    assert _FakeOdds.bought == ["2026-09-05T15:50:00Z"]
    closing = load_closing(cfg.closing_file(DAY))
    assert "OTHER @ GAME|game_ml|GAME" in closing
    ala = closing[f"{MATCHUP}|game_ml|ALA"]
    assert ala.captured_at == "2026-09-05T15:45:00Z" and ala.kickoff_utc == KICK
    assert load_ledger(cfg.ledger_file)[0].clv is not None
