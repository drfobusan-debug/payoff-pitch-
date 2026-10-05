from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone

from nhl_engine.audit.ledger import LedgerRow, grade_rows
from nhl_engine.data.capture import QuoteRow, pregame
from nhl_engine.pipeline import has_started, one_buy_per_side
from nhl_engine.schemas import GameResult, PeriodScore


def _q(side: str, american: float, opp: float, at: str, matchup: str = "A @ H") -> QuoteRow:
    return QuoteRow(
        at, "2026-10-01", matchup, "ev1", "game_ml", side, "", None, "dk", american, opp
    )


def _row(**kw) -> LedgerRow:
    base = dict(
        slate_date="2026-10-01",
        matchup="A @ H",
        home="H",
        away="A",
        market="game_total",
        side="under",
        entity="",
        line=6.5,
        ot_rule="incl_ot_so",
        book="dk",
        american=-110.0,
        books=3,
        consensus=0.5,
        model_prob=0.56,
        push_prob=0.0,
        edge=0.06,
        ev=0.07,
        tier="Strong buy",
        pass_gate=True,
        kelly=0.05,
    )
    base.update(kw)
    return LedgerRow(**base)


def test_pregame_drops_in_play_quotes_and_keeps_unknown_games():
    rows = [
        _q("H", -140, 120, "2026-10-01T22:30:00Z"),
        _q("H", -400, 300, "2026-10-01T23:30:00Z"),
        _q("H", -150, 130, "2026-10-01T23:30:00Z", matchup="B @ C"),
    ]
    kept = pregame(rows, {"A @ H": "2026-10-01T23:00:00Z"})
    assert [(r.matchup, r.american) for r in kept] == [("A @ H", -140), ("B @ C", -150)]


def test_grade_rows_closes_on_last_pre_drop_quote():
    quotes = [
        _q("H", -140, 120, "2026-10-01T22:30:00Z"),
        _q("A", 120, -140, "2026-10-01T22:30:00Z"),
        _q("H", -400, 300, "2026-10-01T23:30:00Z"),
        _q("A", 300, -400, "2026-10-01T23:30:00Z"),
    ]
    res = {
        "A @ H": GameResult(
            1,
            Date(2026, 10, 1),
            "A",
            "H",
            "OFF",
            tuple(PeriodScore(i + 1, "REG", 0, 1) for i in range(3)),
            "REG",
        )
    }

    def graded(starts):
        row = _row(market="game_ml", side="H", line=None)
        return grade_rows([row], res, quotes, graded_at="g", starts=starts)[0]

    assert graded({"A @ H": "2026-10-01T23:00:00Z"}).close_american == -140
    assert graded(None).close_american == -400


def test_has_started():
    now = datetime(2026, 10, 4, 21, 0, tzinfo=timezone.utc)
    assert has_started("2026-10-04T17:00:00Z", now)
    assert not has_started("2026-10-04T22:00:00Z", now)
    assert not has_started("", now)


def test_one_buy_per_side_keeps_highest_kelly():
    u60 = _row(line=6.0, kelly=0.04)
    u65 = _row(line=6.5, kelly=0.06)
    over = _row(side="over", line=5.5, kelly=0.01)
    passed = _row(line=7.0, tier="Pass", pass_gate=False, kelly=0.2)
    one_buy_per_side([u60, u65, over, passed])
    assert u65.is_buy and over.is_buy
    assert not u60.is_buy and "duplicate_side" in u60.gates and u60.tier == "Strong buy"
    assert passed.gates == []
