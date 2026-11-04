from __future__ import annotations

from dataclasses import replace
from datetime import date as Date
from datetime import timedelta

from nba_engine import cli
from nba_engine.data.capture import QuoteRow
from nba_engine.schemas import GameResult
from tests.nba.test_output import priced


def test_fast_start_reads_quarters_against_the_timely_close(monkeypatch, tmp_path) -> None:
    base = priced()[0]
    days = [Date(2026, 11, 1) + timedelta(days=i) for i in range(4)]
    rows, results, quotes = [], {}, {}
    for d in days:
        for k in range(12):
            eid = f"{d}-{k}"
            m = f"A{k} @ H{k}"
            rows.append(
                replace(
                    base,
                    event_id=eid,
                    matchup=m,
                    slate_date=d.isoformat(),
                    tip_utc=f"{d}T23:00:00Z",
                )
            )
            gap = (k % 5 - 2) * 4
            q1 = 55 + gap
            results.setdefault(d, []).append(
                GameResult(
                    espn_id=eid,
                    game_date=d,
                    away=f"A{k}",
                    home=f"H{k}",
                    state="STATUS_FINAL",
                    away_q=(q1 // 2, 28, 28, 28),
                    home_q=(q1 - q1 // 2, 27, 27, 27 + gap),
                )
            )
            quotes.setdefault(d, []).append(
                QuoteRow(
                    f"{d}T22:40:00Z",
                    d.isoformat(),
                    m,
                    eid,
                    "game_total",
                    "over",
                    "",
                    220.0,
                    "draftkings",
                    -110,
                    -110,
                )
            )
    monkeypatch.setattr(cli.boxes, "read_results", lambda root, d: results.get(d))
    monkeypatch.setattr(cli.capture, "read_day", lambda root, d: quotes.get(d, []))
    reads = {r.checkpoint: r for r in cli._fast_start(tmp_path, rows, days)}
    assert set(reads) == {"end of Q1", "half", "end of Q3"}
    assert reads["end of Q1"].n == 12 and abs(reads["end of Q1"].b - 2.0) < 1e-6
    lines = cli._fast_start_lines(tmp_path, rows, days)
    assert lines[0].startswith("[fast starts") and any(
        line.startswith("end of Q1 |") for line in lines
    )
