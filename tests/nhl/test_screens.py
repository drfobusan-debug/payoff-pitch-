"""Graded-only refusal screens: which buys each would refuse, saved units, halves, verdict."""

from __future__ import annotations

from datetime import date as Date

import pytest

from nhl_engine.audit import ledger_report, screens
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.data.availability import Availability


def _row(
    *,
    slate="2026-10-10",
    matchup="A @ H",
    market="game_ml",
    side="A",
    status="confirmed/confirmed",
    pnl=-1.0,
    tier="Strong buy",
    priced_at="2026-10-10T21:00:00Z",
) -> LedgerRow:
    return LedgerRow(
        slate_date=slate,
        matchup=matchup,
        home="H",
        away="A",
        market=market,
        side=side,
        entity="",
        line=None,
        ot_rule="incl_ot_so",
        book="b",
        american=100.0,
        books=3,
        consensus=0.5,
        model_prob=0.55,
        push_prob=0.0,
        edge=0.05,
        ev=0.1,
        tier=tier,
        pass_gate=True,
        goalie_status=status,
        priced_at=priced_at,
        outcome="win" if pnl > 0 else "loss",
        pnl=pnl,
    )


def _hits(rows, ctx=None):
    ctx = ctx or screens.Context()
    return {s.name: s.n for s in screens.read_screens(rows, ctx)}


def test_goalie_probable_hits_game_buys_only():
    rows = [
        _row(status="probable/confirmed"),
        _row(status="confirmed/confirmed"),
        _row(status="probable/confirmed", market="sk_points"),
        _row(status="probable/confirmed", tier="Pass"),
    ]
    assert _hits(rows)["goalie_probable"] == 1


def test_b2b_screens_read_the_context():
    ctx = screens.Context(b2b=frozenset({"2026-10-10|A"}))
    rows = [
        _row(status="probable/confirmed"),  # road B2B, unconfirmed, road side
        _row(status="confirmed/confirmed", side="H"),  # B2B team confirmed, home side
    ]
    hits = _hits(rows, ctx)
    assert hits["b2b_goalie_unconfirmed"] == 1
    assert hits["road_b2b_vs_rested_home"] == 1
    both = screens.Context(b2b=frozenset({"2026-10-10|A", "2026-10-10|H"}))
    assert _hits(rows, both)["road_b2b_vs_rested_home"] == 0


def test_saved_units_halves_and_verdict():
    rows = [
        _row(status="probable/confirmed", pnl=p, matchup=f"A @ H{i}")
        for i, p in enumerate((-1.0, -1.0, 0.9, -1.0))
    ]
    s = next(
        x for x in screens.read_screens(rows, screens.Context()) if x.name == "goalie_probable"
    )
    assert s.n == 4 and s.record == "1-3"
    assert s.saved == pytest.approx(2.1)
    assert s.halves == pytest.approx((2.0, 0.1))
    assert s.verdict == "probation (4/100)"
    many = [
        _row(status="probable/confirmed", pnl=-1.0 if i % 3 else 0.9, matchup=f"A @ H{i}")
        for i in range(120)
    ]
    s = next(
        x for x in screens.read_screens(many, screens.Context()) if x.name == "goalie_probable"
    )
    assert s.verdict == "clears the bar: promote explicitly"


def test_tickets_on_one_game_are_one_game_for_se_and_halves():
    rows = [
        _row(status="probable/confirmed", pnl=-1.0),
        _row(status="probable/confirmed", pnl=-1.0, market="game_pl"),
        _row(status="probable/confirmed", pnl=0.9, matchup="B @ C"),
    ]
    s = next(
        x for x in screens.read_screens(rows, screens.Context()) if x.name == "goalie_probable"
    )
    assert s.n == 3
    assert s.games == pytest.approx([-2.0, 0.9])
    assert s.halves == pytest.approx((2.0, -0.9))


def test_b2b_teams_off_the_schedule():
    games = {
        Date(2026, 10, 9): {"A", "X"},
        Date(2026, 10, 10): {"A", "H"},
    }
    got = screens.b2b_teams(["2026-10-10"], lambda d: games.get(d, set()))
    assert got == frozenset({"2026-10-10|A"})


def _av(slate, team, pid, status, seen, role="F"):
    return Availability(
        slate=slate,
        team=team,
        player_id=pid,
        name=f"p{pid}",
        status=status,
        source="rotowire",
        posted_at="",
        seen_at=seen,
        role=role,
    )


def test_late_news_is_first_sighting_after_pricing():
    rows = [_row()]
    recs = [
        _av("2026-10-10", "A", 1, "out", "2026-10-10T22:00:00Z"),  # late
        _av("2026-10-09", "H", 2, "out", "2026-10-09T15:00:00Z"),  # known since yesterday
        _av("2026-10-10", "H", 2, "out", "2026-10-10T22:00:00Z"),
        _av("2026-10-10", "H", 3, "questionable", "2026-10-10T22:00:00Z"),
        _av("2026-10-10", "H", 4, "scratched", "2026-10-10T22:00:00Z", role="G2"),
    ]
    assert screens.late_news_teams(rows, recs) == frozenset({"2026-10-10|A"})


def test_audit_report_renders_the_screens():
    audit = ledger_report.build([_row(status="probable/confirmed")])
    md = ledger_report.render_md(audit, as_of="t")
    assert "## Graded-only screens" in md and "| goalie_probable |" in md
    assert "goalie_probable" in ledger_report.render_html(audit, as_of="t")
