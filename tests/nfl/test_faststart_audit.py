from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from scripts.nfl import ledger_audit
from tests.nfl.test_ledger_audit import row


def _total(day: str, matchup: str, book: str, line: float, prob: float, **kw: object):
    fields: dict[str, object] = {
        "date": day,
        "matchup": matchup,
        "market": "total",
        "side": "over",
        "line": line,
        "book": book,
        "close_odds": -110.0,
        "close_prob": prob,
        "clv": 0.0,
        "kickoff_utc": f"{day}T17:00:00Z",
        "close_captured_at": f"{day}T16:40:00Z",
    }
    return row(**{**fields, **kw})


def test_close_total_is_the_coin_flip_rung_median_across_books() -> None:
    d = "2026-09-13"
    rows = [
        _total(d, "BUF @ KC", "dk", 47.5, 0.50),
        _total(d, "BUF @ KC", "dk", 44.5, 0.60),
        _total(d, "BUF @ KC", "mgm", 48.5, 0.51),
        _total(d, "BUF @ KC", "fd", 30.0, 0.50, close_captured_at=f"{d}T12:00:00Z"),  # stale
    ]
    assert ledger_audit.close_totals(rows) == {("BUF @ KC", d): 48.0}


def test_quarter_points_and_reads() -> None:
    pbp = pd.DataFrame(
        {
            "game_id": ["g"] * 4,
            "game_date": ["2026-09-13"] * 4,
            "home_team": ["KC"] * 4,
            "away_team": ["BUF"] * 4,
            "qtr": [1, 2, 3, 4],
            "total_home_score": [7, 14, 17, 24],
            "total_away_score": [3, 3, 10, 20],
        }
    )
    assert ledger_audit.quarter_points(pbp) == {("BUF @ KC", "2026-09-13"): [10, 7, 10, 17]}

    rows, quarters = [], {}
    for i in range(48):
        d = (date(2026, 9, 1) + timedelta(days=i // 12)).isoformat()
        m = f"A{i} @ H{i}"
        q1 = 6 + i % 7
        rows.append(_total(d, m, "dk", 44.0, 0.5, home_score=30 + q1, away_score=8))
        quarters[(m, d)] = [q1, 11, 11, 16]
    reads = {r.checkpoint: r for r in ledger_audit.fast_start_reads(rows, quarters)}
    assert reads["end of Q1"].n == 12 and abs(reads["end of Q1"].b - 1.0) < 1e-6
