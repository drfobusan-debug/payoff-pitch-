"""The season audit reads both ledgers, grades in memory, and files a record without touching the source."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from mlb_engine.output import daily_worksheet as ws
from mlb_engine.output import totals_audit as ta

audit = pytest.importorskip("scripts.season_audit")


def _total(day: str, pk: int, sum_pts: int, runs: tuple[int, int] | None, line: float = 8.5) -> ta.LedgerRow:
    r = ta.LedgerRow(date=day, game=f"A{pk} @ H{pk}", game_pk=pk, line=line, sum_pts=sum_pts, bands=ta.BANDS)
    if runs is not None:
        r.away_runs, r.home_runs = runs
        total = sum(runs)
        r.result = ta.OVER if total > line else ta.UNDER if total < line else ta.PUSH
    return r


def _sheet(day: str, pk: int, result: str = "", implied: float | None = 0.6, ml: float | None = -150) -> ws.LedgerRow:
    return ws.LedgerRow(
        date=day, game=f"AW{pk} @ HM{pk}", game_pk=pk, away=f"AW{pk}", home=f"HM{pk}", away_sp="a", home_sp="h",
        away_bat=1.0, away_bp=1.0, away_sp_pts=1.0, home_bat=0.0, home_bp=0.0, home_sp_pts=0.0,
        away_w=40.0, home_w=25.0, gap=15.0, fav=f"AW{pk}", complete=True,
        away_ml=ml, home_ml=130, fav_implied=implied, result=result,
    )


def _write(tmp: Path, totals: list[ta.LedgerRow], sheet: list[ws.LedgerRow]) -> Path:
    src = tmp / "src"
    ta.write_ledger(src / ta.LEDGER_NAME, totals)
    ws.save_ledger(src / ws.LEDGER_NAME, sheet)
    return src


def test_report_counts_the_season_and_leaves_the_source_alone(tmp_path: Path) -> None:
    totals = [
        _total("2026-09-01", 1, 6, (5, 5)),  # over call, 10 runs: hit
        _total("2026-09-01", 2, -6, (5, 5)),  # under call: miss
        _total("2026-09-02", 3, -3, (1, 2)),  # under call: hit
        _total("2025-09-02", 4, 9, (9, 9)),  # another season: not counted
    ]
    sheet = [_sheet("2026-09-25", 10, "fav"), _sheet("2026-09-26", 11, "dog"), _sheet("2025-09-26", 12, "fav")]
    src = _write(tmp_path, totals, sheet)
    before = {p.name: p.read_bytes() for p in src.iterdir()}

    path = audit.run(2026, src, tmp_path / "out", "test", grade=False)

    text = path.read_text()
    assert "3 graded games" in text
    assert "| SUM sign as the call | 2-1 (67%) |" in text
    assert "| SUM says Under | 1-1 (50%) |" in text
    assert "2 graded games with both starters scored" in text
    assert "| **All** | 1-1 |" in text
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    assert len(ta.read_ledger(tmp_path / "out" / ta.LEDGER_NAME)) == 3
    assert len(ws.load_ledger(tmp_path / "out" / ws.LEDGER_NAME)) == 2


def test_pending_rows_are_graded_in_the_copy_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _write(tmp_path, [_total("2026-10-06", 849819, -15, None, line=6.5)], [_sheet("2026-10-06", 849819)])
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    finals = {849819: ("AW849819 @ HM849819", 3, 1)}
    monkeypatch.setattr(ta, "finals", lambda day: finals)
    monkeypatch.setattr(ws, "finals", lambda day: finals)

    audit.run(2026, src, tmp_path / "out", "test", today=Date(2026, 10, 7))

    assert [r.result for r in ta.read_ledger(tmp_path / "out" / ta.LEDGER_NAME)] == [ta.UNDER]
    assert [r.result for r in ws.load_ledger(tmp_path / "out" / ws.LEDGER_NAME)] == ["fav"]
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before


def test_market_test_judges_wins_against_the_prices_given() -> None:
    m = audit.MarketTest()
    for won in (True,) * 17 + (False,) * 8:
        m.add(won, 0.557, -125)
    assert m.market == pytest.approx(0.557)
    assert m.z == pytest.approx((17 - 25 * 0.557) / (25 * 0.557 * 0.443) ** 0.5)
    assert m.verdict() == "unproven (n<30)"
    assert m.units == pytest.approx(17 * 0.8 - 8)
    assert m.games_needed() is not None

    strong = audit.MarketTest()
    for won in (True,) * 80 + (False,) * 20:
        strong.add(won, 0.55, None)
    assert strong.verdict() == "beats market"
    assert strong.staked == 0
