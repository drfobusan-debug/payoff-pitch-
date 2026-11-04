from __future__ import annotations

from datetime import date, timedelta

from cfb_engine.audit import faststart as cfb_fast
from cfb_engine.audit.clv import ClosingQuote
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.output.weekly_report import _fast_html


def _close(day: str, line: float, *, minutes: int = 20) -> ClosingQuote:
    kick = f"{day}T20:00:00Z"
    at = f"{day}T{(20 * 60 - minutes) // 60:02d}:{(20 * 60 - minutes) % 60:02d}:00Z"
    return ClosingQuote(-110.0, 0.5, line, captured_at=at, kickoff_utc=kick)


def _slate(n: int = 48):
    closings: dict[str, dict[str, ClosingQuote]] = {}
    games = []
    for i in range(n):
        day = (date(2026, 9, 5) + timedelta(days=7 * (i // 12))).isoformat()
        q1 = 7 + i % 9
        closings.setdefault(day, {})[f"Away{i} @ Home{i}|game_total|Over"] = _close(day, 50.0)
        closings[day][f"Away{i} @ Home{i}|game_total|Under"] = _close(day, 50.0)
        games.append(
            cfb_fast.LineScore(
                f"Home{i}", f"Away{i}", f"{day}T20:00:00.000Z", (q1, 7, 7, 10), (0, 7, 3, 3)
            )
        )
    return closings, games


def test_reads_use_timely_over_close_and_cfbd_quarters() -> None:
    closings, games = _slate()
    reads = {r.checkpoint: r for r in cfb_fast.reads(closings, games)}
    assert set(reads) == {"end of Q1", "half", "end of Q3"}
    assert reads["end of Q1"].n == 12 and abs(reads["end of Q1"].b - 1.0) < 1e-6


def test_stale_close_and_unmatched_games_are_skipped() -> None:
    closings, games = _slate()
    stale = {d: {k: _close(d, 50.0, minutes=90) for k in c} for d, c in closings.items()}
    assert cfb_fast.reads(stale, games) == []
    assert cfb_fast.reads(closings, []) == []


def test_line_scores_from_cfbd_games() -> None:
    client = CFBDClient(None)
    client._games_cache[2026] = [
        {
            "homeTeam": "Ohio",
            "awayTeam": "Akron",
            "startDate": "2026-09-05T20:00:00.000Z",
            "homeLineScores": [7, 0, 3, 7],
            "awayLineScores": [0, 3, 0, 0],
        },
        {"homeTeam": "Kent State", "awayTeam": "Toledo", "startDate": "2026-09-05T20:00:00.000Z"},
    ]
    assert client.fetch_line_scores(2026) == [
        cfb_fast.LineScore("Ohio", "Akron", "2026-09-05T20:00:00.000Z", (7, 0, 3, 7), (0, 3, 0, 0))
    ]


def test_weekly_article_section() -> None:
    closings, games = _slate()
    html = _fast_html(cfb_fast.reads(closings, games))
    assert "Fast starts" in html and "end of Q1" in html
    assert "Needs 30+" in _fast_html([])
