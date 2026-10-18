"""The display-only NFL graduation bar."""

from __future__ import annotations

from nfl_engine.audit.graduation import (
    MEETS_BAR,
    NOT_YET,
    WATCHING,
    eligible_buys,
    graduation,
    graduation_lines,
)
from nfl_engine.audit.ledger import LedgerEntry


def buy(i: int, *, result: str = "win", **overrides: object) -> LedgerEntry:
    day = 1 + i // 24
    hour = i % 24
    entry = LedgerEntry(
        season=2026,
        week=1 + i // 16,
        date=f"2026-10-{day:02d}",
        matchup=f"G{i} @ H{i}",
        market="spread",
        side=f"G{i}",
        line=3.5,
        book="dk",
        odds=100.0,
        opposite_odds=-120.0,
        tier="Strong buy",
        model_prob=0.56,
        fair_prob=0.50,
        ev_model=0.12,
        ev_fair=0.05,
        paired_books=3,
        result=result,
        pnl=1.0 if result == "win" else -1.0,
        captured_at=f"2026-10-{day:02d}T{hour:02d}:00:00Z",
        kickoff_utc=f"2026-10-{day:02d}T{hour:02d}:50:00Z",
        close_captured_at=f"2026-10-{day:02d}T{hour:02d}:40:00Z",
        clv=0.02 if i % 3 else 0.0,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def test_after_kickoff_price_and_second_card_do_not_count() -> None:
    first = buy(0)
    late = buy(1, captured_at="2026-10-01T01:55:00Z")
    second_card = buy(0, week=2, book="fd", captured_at="2026-10-01T00:10:00Z")
    unstamped = buy(2, captured_at="")
    pass_row = buy(3, tier="Pass")
    assert eligible_buys([first, late, second_card, unstamped, pass_row]) == [first]


def test_watching_below_volume() -> None:
    g = graduation([buy(i) for i in range(10)])
    assert g.status == WATCHING
    assert g.failed[0].startswith("volume (10 of 100)")


def test_meets_bar_when_winning_steadily_with_real_clv() -> None:
    rows = [buy(i, result="loss" if i % 4 == 0 else "win") for i in range(120)]
    g = graduation(rows)
    assert (g.status, g.failed) == (MEETS_BAR, ())


def test_an_early_close_is_not_clv_evidence() -> None:
    rows = [
        buy(i, result="loss" if i % 4 == 0 else "win", close_captured_at="2026-09-30T08:00:00Z")
        for i in range(120)
    ]
    g = graduation(rows)
    assert g.status == NOT_YET
    assert g.failed == ("clv",)
    assert g.clv is None
    assert "n/a on 0 buy(s)" in "\n".join(graduation_lines(rows))


def test_a_losing_half_blocks_it() -> None:
    rows = [buy(i, result="win" if i >= 60 or i % 3 == 0 else "loss") for i in range(120)]
    g = graduation(rows)
    assert g.status == NOT_YET
    assert "consistency" in g.failed
