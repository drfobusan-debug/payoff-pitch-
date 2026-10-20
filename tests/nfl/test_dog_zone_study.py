from __future__ import annotations

from pathlib import Path

from nfl_engine.audit.ledger import LOSS, LedgerEntry
from scripts.nfl.dog_zone_study import load_cache, summary, write_cache

SETTINGS: dict[str, int | bool] = {"first": 2006, "last": 2025, "sims": 5000, "ratings": True}


def _row(result: str = LOSS, pnl: float = -1.0) -> LedgerEntry:
    return LedgerEntry(
        season=2024,
        week=3,
        date="2024-09-22",
        matchup="ARI @ DET",
        market="moneyline",
        side="ARI",
        line=None,
        book="close",
        odds=350.0,
        opposite_odds=-450.0,
        tier="Pass",
        model_prob=0.25,
        fair_prob=0.2,
        ev_model=0.12,
        ev_fair=None,
        paired_books=1,
        result=result,
        pnl=pnl,
    )


def test_an_all_loss_sample_still_reports() -> None:
    line = summary("ML dog above +300", [_row(), _row(), _row()])
    assert "0-3" in line
    assert "ROI -100.0%" in line
    assert "t +nan" in line


def test_the_cache_round_trips_and_refuses_other_settings(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    rows = [_row(), _row()]
    write_cache(path, SETTINGS, rows)
    assert load_cache(path, SETTINGS) == rows
    assert load_cache(path, {**SETTINGS, "sims": 200}) is None
    assert load_cache(path, {**SETTINGS, "ratings": False}) is None
