"""Ledger audit report: tallies, sign tests, one-side-per-game reads, goalie accuracy, files."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from nhl_engine.audit import ledger, ledger_report
from nhl_engine.audit.ledger import LedgerRow


def _row(
    *,
    slate="2026-10-01",
    matchup="A @ H",
    market="game_ml",
    side="H",
    line=None,
    consensus=0.5,
    model=0.55,
    tier="Strong buy",
    pass_gate=True,
    outcome="win",
    pnl=0.9,
    clv=-0.01,
    goalie_status="confirmed/projected",
) -> LedgerRow:
    return LedgerRow(
        slate_date=slate,
        matchup=matchup,
        home="H",
        away="A",
        market=market,
        side=side,
        entity="",
        line=line,
        ot_rule="incl_ot_so",
        book="b",
        american=-110,
        books=5,
        consensus=consensus,
        model_prob=model,
        push_prob=0.0,
        edge=model - consensus,
        ev=0.05,
        tier=tier,
        pass_gate=pass_gate,
        away_goalie="Jakub Dobes",
        home_goalie="Joseph Woll",
        goalie_status=goalie_status,
        outcome=outcome,
        pnl=pnl,
        clv=clv,
    )


def test_sign_test_and_bootstrap():
    assert ledger_report.sign_test_p(0, 0) == 1.0
    assert ledger_report.sign_test_p(5, 10) == pytest.approx(1.0)
    assert ledger_report.sign_test_p(28, 29) < 1e-5
    assert ledger_report.sign_test_p(1, 29) == ledger_report.sign_test_p(28, 29)
    lo, hi = ledger_report.bootstrap_roi([1.0, -1.0, 1.0, -1.0], n=200)
    assert -1 <= lo <= 0 <= hi <= 1
    assert ledger_report.bootstrap_roi([]) == (0.0, 0.0)


def test_build_reads_buys_sides_and_goalies():
    rows = []
    for i in range(24):  # model under the market on every total; overs win
        m = f"T{i} @ H{i}"
        rows.append(
            _row(
                matchup=m,
                market="game_total",
                side="over",
                line=6.0,
                consensus=0.52,
                model=0.45,
                outcome="win",
            )
        )
        rows.append(
            _row(
                matchup=m,
                market="game_total",
                side="under",
                line=6.0,
                consensus=0.48,
                model=0.55,
                outcome="loss",
                pnl=-1.0,
            )
        )
        rows.append(
            _row(
                matchup=m,
                market="game_total",
                side="over",
                line=7.5,
                consensus=0.3,
                model=0.2,
                outcome="loss",
                pnl=-1.0,
                pass_gate=False,
            )
        )
        rows.append(
            _row(
                matchup=m,
                market="game_ml",
                side="H",
                consensus=0.5,
                model=0.5,
                outcome="win",
                tier="Pass",
                pass_gate=False,
            )
        )
    rows.append(_row(matchup="X @ Y", outcome=None, pnl=None, clv=None))  # ungraded
    starters = {f"2026-10-01|T{i} @ H{i}": ("J. Dobes", "S. Bobrovsky") for i in range(24)}
    a = ledger_report.build(rows, starters)
    assert a.nights == ["2026-10-01"] and a.rows == len(rows)
    assert a.buys.n == 48 and a.buys.record == "24-24-0" and a.buys.clv_neg == 48
    assert a.clv_p < 1e-6
    assert set(a.buys_by_market) == {"game_total"} and set(a.buys_by_tier) == {"Strong buy"}
    over = next(s for s in a.sides if s.label == "Over")
    assert over.n == 24 and over.below == 24 and over.brier_model > over.brier_market
    assert over.won_above == (0, 0) and over.won_below == (24, 24)
    assert (
        over.outcomes.record == "24-0-0"
    )  # the 6.0 line (nearest 0.5) represents the game, not the 7.5 alt
    g = {x.status: x for x in a.goalies}
    assert g["confirmed"].wrong == 0 and g["projected"].wrong == 24
    text = " ".join(a.findings)
    assert (
        "market moves against our buys" in text and "Over: model under the market in 24/24" in text
    )
    assert "'projected' was wrong 24/24" in text and "not callable" in text


def test_build_empty_and_render(tmp_path: Path):
    a = ledger_report.build([])
    assert a.buys.n == 0 and a.sides == [] and a.findings == []
    md = ledger_report.render_md(a, as_of="t")
    assert "nothing graded yet" in md and "0 graded nights" in md
    rows = [_row(), _row(matchup="B @ H", outcome="loss", pnl=-1.0, clv=0.02)]
    a = ledger_report.build(rows)
    md = ledger_report.render_md(a, as_of="t")
    assert "**1-1-0, -0.10u, ROI -5.0%**" in md and "| game_ml | 1-1-0 |" in md
    html = ledger_report.render_html(a, as_of="t")
    assert "<h1>NHL ledger audit" in html and "home ML" in html
    assert ledger_report.latest(tmp_path) == {}
    paths = ledger_report.write(a, tmp_path, Date(2026, 10, 2), as_of="t", pdf=False)
    assert set(paths) == {"audit_md"} and paths["audit_md"].read_text().startswith(
        "# NHL ledger audit"
    )
    ledger_report.write(a, tmp_path, Date(2026, 10, 1), as_of="t", pdf=False)
    assert ledger_report.latest(tmp_path)["audit_md"].name == "ledger_audit_2026-10-02.md"


def test_cli_audit_writes_report_and_card_attaches_it(tmp_path: Path, monkeypatch):
    from nhl_engine import cli
    from nhl_engine.audit import ledger

    monkeypatch.setenv("NHLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NHLE_OUTPUT_DIR", str(tmp_path / "out"))
    ledger.save_rows(
        [_row(), _row(matchup="B @ H", outcome="loss", pnl=-1.0)],
        tmp_path / "ledger" / "graded_2026-10-01.json",
    )
    monkeypatch.setattr(cli, "_results_for", lambda day: ({}, {}))
    real_write = ledger_report.write
    monkeypatch.setattr(
        ledger_report,
        "write",
        lambda a, out, day, *, as_of, pdf=True: real_write(a, out, day, as_of=as_of, pdf=False),
    )
    assert cli.main(["audit", "--date", "2026-10-02"]) == 0
    got = ledger_report.latest(tmp_path / "out")
    assert got["audit_md"].name == "ledger_audit_2026-10-02.md"
    assert "1-1-0" in got["audit_md"].read_text()


def test_clv_sign_test_ignores_flat_closes():
    rows = [_row(matchup=f"T{i} @ H", clv=0.0) for i in range(30)]
    rows += [_row(matchup="U @ H", clv=-0.02), _row(matchup="V @ H", clv=0.01)]
    a = ledger_report.build(rows)
    assert (a.buys.clv_neg, a.buys.clv_pos, a.buys.clv_flat) == (1, 1, 30)
    assert a.clv_p == 1.0
    text = " ".join(a.findings)
    assert "against 1, toward 1, flat 30" in text and "no CLV verdict yet" in text


def test_a_night_without_closes_leaves_its_clv_blank():
    rows = [_row(slate="2026-10-01"), _row(slate="2026-10-02", matchup="B @ H", clv=None)]
    md = ledger_report.render_md(ledger_report.build(rows), as_of="t")
    assert "| 2026-10-01 | 1-0-0 | +90.0% | +0.90u | -1.0 |" in md
    assert "| 2026-10-02 | 1-0-0 | +90.0% | +0.90u |  |" in md
    assert "- - " not in ledger_report.render_md(ledger_report.build([]), as_of="t")


def test_unregradable_lists_graded_nights_without_predictions(tmp_path: Path):
    d = tmp_path / "ledger"
    d.mkdir()
    for name in ("graded_2026-10-01.json", "predictions_2026-10-01.json", "graded_2026-10-02.json"):
        (d / name).write_text("[]")
    assert ledger.unregradable(tmp_path) == [Date(2026, 10, 2)]
    assert ledger.unregradable(tmp_path / "missing") == []
