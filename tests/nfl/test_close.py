"""The closing stamp: provisional until kickoff, final afterwards.

The bug being nailed down: `job` runs `close` every time it runs, and the first
run of a week happens days before kickoff. Stamping that price as the close and
refusing to look again measures CLV against a Wednesday number -- which flatters
every bet the market later moved toward and hides every one it moved away from.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pytest

from nfl_engine import cli
from nfl_engine.audit.ledger import (
    LedgerEntry,
    apply_close,
    close_is_final,
    load_ledger,
    save_ledger,
)
from nfl_engine.market.board import GameOdds, MarketQuote

HOME, AWAY = "KC", "BUF"
MATCHUP = f"{AWAY} @ {HOME}"
KICKOFF = "2026-09-13T17:00:00Z"
DAY = "2026-09-13"


def row(**overrides: object) -> LedgerEntry:
    entry = LedgerEntry(
        season=2026,
        week=1,
        date=DAY,
        matchup=MATCHUP,
        market="moneyline",
        side=HOME,
        line=None,
        book="dk",
        odds=110.0,
        opposite_odds=-130.0,
        tier="Strong buy",
        model_prob=0.55,
        fair_prob=0.52,
        ev_model=0.155,
        ev_fair=0.092,
        paired_books=3,
        captured_at="2026-09-09T18:00:00Z",
        kickoff_utc=KICKOFF,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def board(home: float = -140, away: float | None = 120) -> dict[str, GameOdds]:
    odds = GameOdds(matchup=MATCHUP)
    odds.add_ml(HOME, MarketQuote("dk", home, away))
    if away is not None:
        odds.add_ml(AWAY, MarketQuote("dk", away, home))
    return {MATCHUP: odds}


def close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    entries: list[LedgerEntry],
    *,
    quotes: dict[str, GameOdds] | None = None,
    now: datetime,
    captured_at: str = "2026-09-13T16:55:00Z",
) -> list[LedgerEntry]:
    """Run `nfl-engine close` against a hand-built board and a frozen clock."""
    path = tmp_path / "nfl_ledger.csv"
    save_ledger(path, entries)
    monkeypatch.setattr(cli, "ledger_path", lambda root=None: path)
    monkeypatch.setattr(
        cli,
        "_fetch",
        lambda days, kind=None: cli.Fetched(
            season=2026,
            week=1,
            captured_at=captured_at,
            games=[],
            board=board() if quotes is None else quotes,
        ),
    )

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003 - the point is to ignore the wall clock
            return now

    monkeypatch.setattr(cli, "datetime", _Clock)
    assert cli.cmd_close(argparse.Namespace(days=8, write=True)) == 0
    return load_ledger(path)


def test_kickoff_decides_whether_the_close_is_final() -> None:
    before = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
    after = datetime(2026, 9, 13, 17, 0, 1, tzinfo=timezone.utc)
    assert not close_is_final(row(), now=before)
    assert close_is_final(row(), now=after)


def test_a_row_without_a_kickoff_time_freezes_on_the_day_after() -> None:
    """Rows priced before the board carried a kickoff time fall back to the date.

    Late is safe and early is not: freezing at midnight UTC after the game keeps a
    real close, while freezing at midnight before it would record a pregame price
    from the previous day as the close.
    """
    undated = row(kickoff_utc="")
    assert not close_is_final(undated, now=datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc))
    assert close_is_final(undated, now=datetime(2026, 9, 14, 0, 1, tzinfo=timezone.utc))
    # A kickoff string the board mangled is treated as absent, not as the epoch.
    assert not close_is_final(
        row(kickoff_utc="sunday afternoon"), now=datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc)
    )


def test_the_first_capture_of_the_week_stamps_a_provisional_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    stamped = close(
        monkeypatch,
        tmp_path,
        [row()],
        now=datetime(2026, 9, 9, 18, 0, tzinfo=timezone.utc),
        captured_at="2026-09-09T18:05:00Z",
    )[0]
    assert stamped.close_odds == -140
    assert stamped.close_captured_at == "2026-09-09T18:05:00Z"
    assert stamped.clv is not None and stamped.clv > 0


def test_a_later_price_before_kickoff_replaces_the_earlier_stamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The whole fix: Wednesday's number is not the close, so it gets overwritten.

    The board here moves back toward the +110 taken, which shrinks CLV -- the
    direction that matters, since a stamp frozen early can only ever be replaced
    by a *worse* number for us, and that is the number the record should carry.
    """
    early = apply_close(row(), -140, 120, captured_at="2026-09-09T18:05:00Z")
    early_clv = early.clv
    late = close(
        monkeypatch,
        tmp_path,
        [early],
        quotes=board(105, -125),
        now=datetime(2026, 9, 13, 16, 55, tzinfo=timezone.utc),
    )[0]
    assert late.close_odds == 105
    assert late.close_captured_at == "2026-09-13T16:55:00Z"
    assert late.clv is not None and early_clv is not None
    assert late.clv < early_clv


def test_kickoff_freezes_the_stamp_against_in_play_prices(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    final = apply_close(row(), -140, 120, captured_at="2026-09-13T16:55:00Z")
    frozen = close(
        monkeypatch,
        tmp_path,
        [final],
        quotes=board(-400, 320),
        now=datetime(2026, 9, 13, 18, 30, tzinfo=timezone.utc),
    )[0]
    assert frozen.close_odds == -140
    assert frozen.close_captured_at == "2026-09-13T16:55:00Z"
    assert frozen.clv == final.clv


def test_a_graded_row_is_never_re_closed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    graded = apply_close(row(result="win", pnl=1.1), -140, 120)
    after = close(
        monkeypatch,
        tmp_path,
        [graded],
        quotes=board(-400, 320),
        now=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    )[0]
    assert after.close_odds == -140
    assert after.result == "win"


def test_a_missing_quote_leaves_the_row_alone(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No price for this position means no stamp -- not a stamp of the wrong side."""
    empty = close(
        monkeypatch,
        tmp_path,
        [row()],
        quotes={},
        now=datetime(2026, 9, 13, 16, 55, tzinfo=timezone.utc),
    )[0]
    assert empty.close_odds is None
    assert empty.clv is None


def test_an_unpaired_closing_price_is_still_stamped(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """One side at the close has no hold to remove, so the raw price is recorded.

    It reads about half the hold high, which is visible beside the paired stamp
    rather than hidden -- inventing the missing side would hide it.
    """
    unpaired = close(
        monkeypatch,
        tmp_path,
        [row()],
        quotes=board(-140, None),
        now=datetime(2026, 9, 13, 16, 55, tzinfo=timezone.utc),
    )[0]
    paired = apply_close(row(), -140, 120)
    assert unpaired.close_prob is not None and paired.close_prob is not None
    assert unpaired.close_prob > paired.close_prob


def test_the_ledger_carries_kickoff_and_the_stamp_time_through_csv(tmp_path) -> None:
    path = tmp_path / "ledger.csv"
    save_ledger(path, [apply_close(row(), -140, 120, captured_at="2026-09-13T16:55:00Z")])
    loaded = load_ledger(path)[0]
    assert loaded.kickoff_utc == KICKOFF
    assert loaded.close_captured_at == "2026-09-13T16:55:00Z"
    assert not close_is_final(loaded, now=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc))
