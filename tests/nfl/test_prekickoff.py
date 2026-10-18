"""The pre-kickoff pass: a second look at the board once the inactives are out.

`nfl-engine price --within N` must spend nothing when no game is near, price only
the games kicking off inside the window, write only buys, and never displace a
price already held -- the first price of a buy stays its price of record.
"""

from __future__ import annotations

import argparse
from datetime import date as Date
from datetime import datetime, timezone

import pytest

from nfl_engine import cli
from nfl_engine.audit.ledger import LedgerEntry, load_ledger, merge_ledger, save_ledger
from nfl_engine.features import books as books_mod
from nfl_engine.schemas import Game, TeamGameInfo

MORNING = "2026-09-13T12:05:00Z"
EARLY = "2026-09-13T17:00:00Z"
LATE = "2026-09-13T20:25:00Z"
AT_1130 = datetime(2026, 9, 13, 15, 30, tzinfo=timezone.utc)  # 11:30 ET


def row(**overrides: object) -> LedgerEntry:
    entry = LedgerEntry(
        season=2026,
        week=1,
        date="2026-09-13",
        matchup="BUF @ KC",
        market="moneyline",
        side="KC",
        line=None,
        book="dk",
        odds=110.0,
        opposite_odds=-130.0,
        tier="Pass",
        model_prob=0.50,
        fair_prob=0.52,
        ev_model=0.05,
        ev_fair=-0.03,
        paired_books=3,
        captured_at=MORNING,
        kickoff_utc=EARLY,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def game(home: str, away: str, kickoff: str) -> Game:
    return Game(
        game_id=f"{away}@{home}",
        season=2026,
        week=1,
        game_date=Date(2026, 9, 13),
        kickoff_utc=kickoff,
        home=TeamGameInfo(name=home, abbrev=home, is_home=True),
        away=TeamGameInfo(name=away, abbrev=away, is_home=False),
    )


def test_a_later_buy_sits_beside_the_pass_it_replaces(tmp_path) -> None:
    path = tmp_path / "l.csv"
    save_ledger(path, [row()])
    later = row(tier="Moderate buy", odds=125.0, ev_fair=0.02, captured_at="2026-09-13T15:30:00Z")
    assert merge_ledger(path, [later]) == []  # the default keeps first-seen, Pass or not
    assert merge_ledger(path, [later], over_pass=True) == [later]
    held = load_ledger(path)
    assert [(e.tier, e.odds) for e in held] == [("Pass", 110.0), ("Moderate buy", 125.0)]


def test_a_held_buy_keeps_its_first_price(tmp_path) -> None:
    path = tmp_path / "l.csv"
    first = row(tier="Moderate buy", odds=118.0, ev_fair=0.02)
    save_ledger(path, [first])
    later = row(tier="Strong buy", odds=130.0, ev_fair=0.04, captured_at="2026-09-13T15:30:00Z")
    assert merge_ledger(path, [later], over_pass=True) == []
    assert [e.odds for e in load_ledger(path)] == [118.0]


def _pre_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    held: list[LedgerEntry],
    priced: list[LedgerEntry],
    *,
    email: bool = False,
) -> tuple[list[LedgerEntry], list[int], list[list[str]], list[argparse.Namespace]]:
    """`nfl-engine price --within 120` at 11:30 ET, with the model stubbed out."""
    path = tmp_path / "nfl_ledger.csv"
    save_ledger(path, held)
    fetches: list[int] = []
    priced_games: list[list[str]] = []
    cards: list[argparse.Namespace] = []

    def fetch(days, kind=None):
        fetches.append(days)
        games = [game("KC", "BUF", EARLY), game("SF", "DAL", LATE)]
        return cli.Fetched(2026, 1, "2026-09-13T15:30:00Z", games, {})

    def price_slate(games, board, **kwargs):
        priced_games.append([g.game_id for g in games])
        return []

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003 - the point is to ignore the wall clock
            return AT_1130

    monkeypatch.setattr(cli, "ledger_path", lambda root=None: path)
    monkeypatch.setattr(cli, "_fetch", fetch)
    monkeypatch.setattr(cli, "datetime", _Clock)
    monkeypatch.setattr(cli, "_books", lambda *a, **k: books_mod.Books())
    monkeypatch.setattr(cli.books_mod, "attach_qbs", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "price_slate", price_slate)
    monkeypatch.setattr(cli, "_ledger_rows", lambda pricings, taken: list(priced))
    monkeypatch.setattr(cli, "_stamp_open", lambda *a, **k: "open: stubbed")
    monkeypatch.setattr(cli, "cmd_card", lambda a: cards.append(a) or 0)
    args = argparse.Namespace(
        days=8, sims=100, top=5, write=True, ratings=False, within=120, email=email
    )
    assert cli.cmd_price(args) == 0
    return load_ledger(path), fetches, priced_games, cards


def test_no_game_near_kickoff_spends_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _, fetches, _, _ = _pre_pass(monkeypatch, tmp_path, [row(kickoff_utc=LATE)], [])
    assert fetches == []


def test_it_prices_only_the_games_inside_the_window_and_writes_only_buys(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    buy = row(tier="Moderate buy", odds=125.0, ev_fair=0.02, captured_at="2026-09-13T15:30:00Z")
    still_pass = row(side="BUF", odds=-140.0, captured_at="2026-09-13T15:30:00Z")
    held, fetches, priced_games, cards = _pre_pass(
        monkeypatch, tmp_path, [row()], [buy, still_pass], email=True
    )
    assert fetches == [8]
    assert priced_games == [["BUF@KC"]]  # the 4:25 game waits for its own pass
    assert [(e.side, e.tier) for e in held] == [("KC", "Pass"), ("KC", "Moderate buy")]
    assert len(cards) == 1 and cards[0].email and cards[0].week == 1


def test_no_new_buy_sends_no_email(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    still_pass = row(captured_at="2026-09-13T15:30:00Z")
    held, _, _, cards = _pre_pass(monkeypatch, tmp_path, [row()], [still_pass], email=True)
    assert len(held) == 1
    assert cards == []
