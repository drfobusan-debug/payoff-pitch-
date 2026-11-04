from __future__ import annotations

import json
from datetime import date as Date
from datetime import timedelta

from nhl_engine import cli
from nhl_engine.audit import ledger_report
from nhl_engine.data.capture import QuoteRow
from nhl_engine.schemas import GameResult, PeriodScore
from tests.nhl.test_ledger_report import _row


def test_fast_start_reads_periods_and_writes_each_night_once(monkeypatch, tmp_path) -> None:
    nights = [Date(2026, 10, 1) + timedelta(days=i) for i in range(4)]
    rows, results, quotes, calls = [], {}, {}, []
    for d in nights:
        for k in range(10):
            m = f"A{k} @ H{k}"
            rows.append(_row(slate=d.isoformat(), matchup=m))
            p1 = k % 4
            results.setdefault(d, {})[m] = GameResult(
                nhl_game_id=k,
                game_date=d,
                away=f"A{k}",
                home=f"H{k}",
                state="OFF",
                periods=(
                    PeriodScore(1, "REG", p1, 0),
                    PeriodScore(2, "REG", 1, 1),
                    PeriodScore(3, "REG", 1, 2 * p1),
                ),
            )
            quotes.setdefault(d, []).append(
                QuoteRow(
                    f"{d}T22:40:00Z",
                    d.isoformat(),
                    m,
                    f"{d}-{k}",
                    "game_total",
                    "over",
                    "",
                    6.0,
                    "draftkings",
                    -110,
                    -110,
                )
            )

    def fake_results(day):
        calls.append(day)
        return results[day], {m: f"{day}T23:00:00Z" for m in results[day]}

    monkeypatch.setattr(cli, "_results_for", fake_results)
    monkeypatch.setattr(cli.capture, "read_day", lambda root, d: quotes.get(d, []))
    reads = {r.checkpoint: r for r in cli._fast_start(tmp_path, rows, nights[-1])}
    assert set(reads) == {"end of P1", "end of P2"}
    assert reads["end of P1"].n == 10 and abs(reads["end of P1"].b - 3.0) < 1e-6
    assert len(json.loads((tmp_path / "faststart" / f"{nights[0]}.json").read_text())) == 20
    cli._fast_start(tmp_path, rows, nights[-1])
    assert len(calls) == 4

    audit = ledger_report.build(rows, fast=list(reads.values()))
    md = ledger_report.render_md(audit, as_of="x")
    assert "## Fast starts" in md and "| end of P1 |" in md and "in-play totals" in md
    assert "Fast starts" in ledger_report.render_html(audit, as_of="x")


def test_fast_start_skips_a_night_not_yet_final(monkeypatch, tmp_path) -> None:
    d = Date(2026, 10, 1)
    live = GameResult(nhl_game_id=1, game_date=d, away="A", home="H", state="LIVE")
    monkeypatch.setattr(
        cli, "_results_for", lambda day: ({"A @ H": live}, {"A @ H": f"{d}T23:00:00Z"})
    )
    assert cli._fast_night(tmp_path, d) == []
    assert not (tmp_path / "faststart").exists()
