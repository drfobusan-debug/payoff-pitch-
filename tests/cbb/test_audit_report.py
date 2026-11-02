from __future__ import annotations

import argparse
from datetime import date as Date
from pathlib import Path

import pytest

from cbb_engine import audit, audit_report, cli
from engine_runtime import ledger

DAY = Date(2026, 11, 20)


def _row(day: str, eid: str, *, side: str = "over", outcome: str = "win", status: str = "live",
         market: str = "game_total", tier: str = "Moderate buy", gates: list[str] | None = None,
         clv: float | None = None, edge: float = 0.025, book: str = "draftkings") -> ledger.Row:
    r = ledger.Row(slate_date=day, event_id=eid, tip_utc=f"{day}T23:00:00Z", market=market, side=side,
                   line=140.5, book=book, pass_id="p", pass_kind="featured", bet_of_record=True,
                   status=status, american=-110, p_fair=0.5, p_model=0.52, tier=tier, edge=edge,
                   gates=list(gates or []))
    r.outcome, r.pnl, r.clv_consensus = outcome, audit.pnl(r, outcome), clv
    r.close_status = "pre_tip" if clv is not None else "pulled"
    return r


def _ledger() -> dict[str, list[ledger.Row]]:
    return {
        "2026-11-05": [_row("2026-11-05", "a", outcome="loss", clv=-1.0), _row("2026-11-05", "b", clv=2.0)],
        "2026-11-18": [
            _row("2026-11-18", "c", clv=1.0, tier="Strong buy", edge=0.06),
            _row("2026-11-18", "d", outcome="push"),
            _row("2026-11-18", "e", market="game_ats", side="home", status="research"),
            _row("2026-11-18", "f", tier="Pass", gates=["thin_market"], outcome="loss"),
        ],
        "2026-11-25": [_row("2026-11-25", "g")],  # after the audit day: never counted
    }


def test_segment_record_units_breakeven_and_clv() -> None:
    s = audit_report.segment("x", _ledger()["2026-11-05"])
    assert (s.record, s.n, s.units) == ("1-1", 2, round(100 / 110 - 1, 3))
    assert s.breakeven == pytest.approx(110 / 210)
    assert (s.clv_n, s.clv, s.beat) == (2, 0.5, 0.5)
    assert s.lo < s.roi < s.hi and s.verdict == "underpowered (2/100)"


def test_build_splits_last_seven_from_season_and_live_from_research_and_passes() -> None:
    rep = audit_report.build(_ledger(), DAY, checklist_lines=["LIVE CARD ALLOWED"])
    week, season, checks = rep.sections
    assert week.title.startswith("Last 7 days (2026-11-14") and season.title == "Season to date"
    live = {t.title: t for t in season.tables}["Live buys (full-game totals at DraftKings / BetMGM)"]
    assert live.rows[0][:3] == ("all live buys", "4", "2-1-1")  # the Pass and the research buy are out
    assert {r[0]: r[2] for r in live.rows}["Strong buy"] == "1-0"
    wk = {t.title: t for t in week.tables}["Live buys (full-game totals at DraftKings / BetMGM)"]
    assert wk.rows[0][2] == "1-0-1"
    paper = {t.title: t for t in season.tables}["Would-be buys in research markets (cleared every gate; not bet)"]
    assert [r[:3] for r in paper.rows] == [("game_ats", "1", "1-0")]
    assert checks.lines == ["LIVE CARD ALLOWED"]
    assert rep.subject == "CBB audit 2026-11-20 -- season buys 2-1-1, +0.82u [paper]"
    assert any("underpowered (3/100)" in f for f in rep.findings)
    text = audit_report.render_text(rep)
    assert "Grading coverage" in text and "2026-11-25" not in text


def test_empty_ledger_reports_nothing_graded_and_flags_ungraded_slates() -> None:
    r = _row("2026-11-01", "a", outcome="")
    r.pnl = None
    rep = audit_report.build({"2026-11-01": [r]}, DAY, bands=None)
    assert rep.findings[0] == "No graded live buys yet this season."
    assert any("nothing graded (2026-11-01)" in f for f in rep.findings)
    page = audit_report.render_html(rep)
    assert page.startswith("<!DOCTYPE html>") and "bands unavailable" in page


def test_html_renders_to_pdf() -> None:
    pytest.importorskip("weasyprint")
    from cbb_engine.output import pdf

    try:
        body = pdf.render_pdf(audit_report.render_html(audit_report.build(_ledger(), DAY)))
    except OSError as exc:  # native Pango/GLib missing on this box
        pytest.skip(str(exc))
    assert body[:4] == b"%PDF"


def _run_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pdf_ok: bool) -> dict[str, Path]:
    monkeypatch.setenv("ENGINE_RUNTIME_DIR", str(tmp_path / "rt"))
    monkeypatch.setenv("CBBE_DATA_DIR", str(tmp_path / "data"))
    day = "2026-11-18"
    ledger.write_once(tmp_path / "rt", "cbb", day, "p", [_row(day, "c", outcome="", clv=None)])
    sent: dict[str, Path] = {}
    monkeypatch.setattr(cli.checklist, "run", lambda *a, **k: [])
    monkeypatch.setattr(cli.checklist, "render", lambda *a, **k: "LIVE CARD ALLOWED")
    monkeypatch.setattr(cli, "_season_bands", lambda *a: [])

    def render(page: str) -> bytes:
        if not pdf_ok:
            raise OSError("no pango")
        return b"%PDF-fake"

    monkeypatch.setattr(cli.pdf, "render_pdf", render)
    monkeypatch.setattr(cli.email, "send", lambda cfg, **k: sent.update(k["paths"]) or "me@x")
    args = argparse.Namespace(date=day, offline=True, time_travel=False, email=True, to=None)
    assert cli.cmd_audit(args) == 0
    assert (tmp_path / "data" / "audit" / f"{day}.json").exists()
    assert (tmp_path / "data" / "audit" / f"cbb_audit_{day}.html").exists()
    return sent


def test_cmd_audit_emails_the_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _run_audit(tmp_path, monkeypatch, pdf_ok=True)
    assert list(sent) == ["audit_pdf"] and sent["audit_pdf"].read_bytes() == b"%PDF-fake"


def test_cmd_audit_falls_back_to_html_when_the_pdf_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _run_audit(tmp_path, monkeypatch, pdf_ok=False)
    assert list(sent) == ["audit_html"]
