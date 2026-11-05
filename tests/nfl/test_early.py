"""Tuesday's early card: record only, write-once, graded and closed like the card."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest

from nfl_engine import cli
from nfl_engine.audit import early
from nfl_engine.audit.ledger import LedgerEntry, load_ledger, save_ledger
from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.schemas import Game, TeamGameInfo

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"
TUESDAY = "2026-09-08T12:05:00Z"
THURSDAY = "2026-09-10T12:05:00Z"
KICKOFF = "2026-09-13T17:00:00Z"


def row(**overrides: object) -> LedgerEntry:
    entry = LedgerEntry(
        season=2026,
        week=1,
        date="2026-09-13",
        matchup=MATCHUP,
        market="spread",
        side=HOME,
        line=-7.5,
        book="dk",
        odds=-110.0,
        opposite_odds=-110.0,
        tier="Pass",
        model_prob=0.53,
        fair_prob=0.5,
        ev_model=0.01,
        ev_fair=-0.04,
        paired_books=4,
        captured_at=TUESDAY,
        kickoff_utc=KICKOFF,
        clv_ev=0.02,
        close_captured_at="2026-09-13T16:50:00Z",
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def game() -> Game:
    return Game(
        game_id="g1",
        season=2026,
        week=1,
        game_date=datetime(2026, 9, 13).date(),
        kickoff_utc=KICKOFF,
        home=TeamGameInfo(name=HOME, abbrev=HOME, is_home=True),
        away=TeamGameInfo(name=AWAY, abbrev=AWAY, is_home=False),
    )


def test_the_lean_is_the_side_rated_above_fair_on_the_first_pass() -> None:
    rows = [
        row(),
        row(side=AWAY, line=7.5, model_prob=0.47),
        row(model_prob=0.40, captured_at=THURSDAY),  # a later pass does not move the lean
    ]
    leans = early.leans(rows)
    assert [e.side for e in leans.values()] == [HOME]
    assert next(iter(leans.values())).captured_at == TUESDAY


def test_pairs_score_the_same_side_at_both_prices() -> None:
    tuesday = [row(clv_ev=0.03)]
    card = [row(side=HOME, model_prob=0.49, clv_ev=0.01, captured_at=THURSDAY)]
    pairs = early.pairs(tuesday, card, "spread")
    assert len(pairs) == 1  # paired even though the card no longer leans that way
    assert pairs[0].gain == pytest.approx(0.02)


def test_unclosed_or_in_play_rows_never_pair() -> None:
    card = [row(captured_at=THURSDAY)]
    assert early.pairs([row(close_captured_at="", clv_ev=None)], card, "spread") == []
    late = row(captured_at="2026-09-13T17:30:00Z")
    assert early.pairs([row()], [late], "spread") == []


def test_the_bar_waits_for_its_games_then_reads_one_se() -> None:
    few = early.bar("x", [early.Pair(row(clv_ev=0.05), row(clv_ev=0.0))], preregistered=True)
    assert few.verdict == f"needs {early.BAR_GAMES - 1} more"
    gains = [0.02 if i % 2 else 0.0 for i in range(early.BAR_GAMES)]
    many = [early.Pair(row(clv_ev=g), row(clv_ev=0.0)) for g in gains]
    assert early.bar("x", many, preregistered=True).verdict == "Tuesday better"
    flat = [early.Pair(row(clv_ev=g - 0.01), row(clv_ev=0.0)) for g in gains]
    assert early.bar("x", flat, preregistered=True).verdict == "not answerable at this n"


def _open_early(monkeypatch: pytest.MonkeyPatch, tmp_path) -> list[int]:
    monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "ledger_path", lambda root=None: tmp_path / "nfl_ledger.csv")
    monkeypatch.setattr(cli.capture, "capture_dir", lambda root=None: tmp_path / "captures")
    fetches: list[int] = []

    def fetch(days, *, kind=cli.capture.GAME_KIND, archive=True):
        fetches.append(days)
        return cli.Fetched(2026, 1, TUESDAY, [game()], {MATCHUP: GameOdds(matchup=MATCHUP)})

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return datetime(2026, 9, 8, 12, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "_fetch", fetch)
    monkeypatch.setattr(cli, "datetime", _Clock)
    monkeypatch.setattr(cli, "_price_fetched", lambda fetched, **k: ["priced"])
    monkeypatch.setattr(
        cli, "_ledger_rows", lambda pricings, taken: [row(captured_at=taken, tier="Moderate buy")]
    )
    return fetches


def test_open_early_writes_only_the_early_ledger_and_only_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fetches = _open_early(monkeypatch, tmp_path)
    args = argparse.Namespace(days=8, early=True, sims=100, ratings=False)
    assert cli.cmd_open(args) == 0
    assert cli.cmd_open(args) == 0
    assert fetches == [8, 8]  # the same single fetch `open` always made
    held = load_ledger(tmp_path / early.EARLY_LEDGER_NAME)
    assert [(e.week, e.captured_at) for e in held] == [(1, TUESDAY)]
    assert not (tmp_path / "nfl_ledger.csv").exists()


def test_plain_open_still_prices_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _open_early(monkeypatch, tmp_path)
    assert cli.cmd_open(argparse.Namespace(days=8)) == 0
    assert not (tmp_path / early.EARLY_LEDGER_NAME).exists()


def test_close_and_grade_carry_the_early_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
    card_path = tmp_path / "nfl_ledger.csv"
    monkeypatch.setattr(cli, "ledger_path", lambda root=None: card_path)
    fresh = {"clv_ev": None, "close_captured_at": ""}
    save_ledger(card_path, [row(captured_at=THURSDAY, **fresh)])
    save_ledger(tmp_path / early.EARLY_LEDGER_NAME, [row(**fresh)])
    odds = GameOdds(matchup=MATCHUP)
    odds.add_spread(-7.5, HOME, MarketQuote("dk", -120, 100))
    odds.add_spread(-7.5, AWAY, MarketQuote("dk", 100, -120))
    monkeypatch.setattr(
        cli,
        "_fetch",
        lambda days, kind=None: cli.Fetched(2026, 1, "2026-09-13T16:50:00Z", [], {MATCHUP: odds}),
    )

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003
            return datetime(2026, 9, 13, 16, 50, tzinfo=timezone.utc)

    monkeypatch.setattr(cli, "datetime", _Clock)
    assert cli.cmd_close(argparse.Namespace(days=8, write=True)) == 0
    monkeypatch.setattr(cli, "_final_scores", lambda season: {(MATCHUP, "2026-09-13"): (31, 20)})
    assert cli.cmd_grade(argparse.Namespace(season=2026, write=True)) == 0
    for path in (card_path, tmp_path / early.EARLY_LEDGER_NAME):
        (held,) = load_ledger(path)
        assert held.close_odds == -120 and held.clv_ev is not None
        assert held.result == "win"


def test_the_parser_knows_early(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "cmd_open", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(cli, "cmd_early_report", lambda args: seen.append(args) or 0)
    assert cli.main(["open", "--early"]) == 0
    assert cli.main(["early-report"]) == 0
    assert seen[0].early is True and seen[0].sims == 40000
