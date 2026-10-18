"""The re-score pass rebuilds old ledger rows under the current rule and re-grades from stored runs."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pandas as pd
import pytest

from mlb_engine.output import daily_worksheet as ws
from mlb_engine.output.daily_worksheet import LedgerRow, Ranking, StarterTable
from scripts import worksheet_rescore as rs


def _rank(pts: dict[str, float], deciles: bool = False) -> Ranking:
    return Ranking({t: {"m": v} for t, v in pts.items()}, {t: {"m": 0.0} for t in pts},
                   {t: set() for t in pts}, sorted(pts, key=lambda t: -pts[t]), 0, deciles=deciles)


def _row(**kw: object) -> LedgerRow:
    base = dict(
        date="2026-09-20", game="SD @ COL", game_pk=7, away="SD", home="COL", away_sp="Ace", home_sp="Scrub",
        away_bat=10.0, away_bp=12.0, away_sp_pts=4.0, home_bat=9.0, home_bp=-8.0, home_sp_pts=11.0,
        away_w=32.0, home_w=8.0, gap=24.0, fav="SD", complete=True, away_ml=-150.0, home_ml=130.0,
        fav_implied=ws.implied(-150, 130), weights="w1-1.5-1-dec-spz", away_runs=2, home_runs=5, result="dog",
    )
    base.update(kw)
    return LedgerRow(**base)  # type: ignore[arg-type]


def _tables() -> tuple[dict[str, Ranking], Ranking, StarterTable, dict[int, str]]:
    bats = {s: _rank({"SD": 60.5, "COL": 60.5}, deciles=True) for s in ("Overall", "vs LHP", "vs RHP")}
    pens = _rank({"SD": -3.0, "COL": 4.0})
    sps = StarterTable(_rank({"1": -6.0, "2": 5.0}), {1: "Ace", 2: "Scrub"}, {1: "SD", 2: "COL"},
                       {1: 100.0, 2: 80.0}, {"ace": 1, "scrub": 2})
    return bats, pens, sps, {1: "L", 2: "R"}


def test_rescore_row_recomputes_gap_flips_the_favourite_and_regrades_from_runs() -> None:
    bats, pens, sps, hands = _tables()
    new = rs.rescore_row(_row(), bats, pens, sps, hands)
    assert (new.away_bp, new.home_bp, new.away_sp_pts, new.home_sp_pts) == (-3.0, 4.0, -6.0, 5.0)
    assert new.away_w == round(8.8 - 4.5 - 6.0, 1) and new.home_w == round(8.8 + 6.0 + 5.0, 1)
    assert new.gap == round(new.away_w - new.home_w, 1) < 0 and new.fav == "COL"
    assert new.result == "fav" and new.weights == ws.WEIGHTS_VERSION
    assert new.fav_implied == ws.implied(130, -150)
    assert (new.away_ml, new.home_ml, new.away_runs, new.home_runs) == (-150.0, 130.0, 2, 5)


def test_rescore_row_drops_the_grade_when_a_starter_no_longer_scores() -> None:
    bats, pens, sps, hands = _tables()
    new = rs.rescore_row(_row(home_sp="Nobody"), bats, pens, sps, hands)
    assert new.home_sp_pts is None and not new.complete and new.result == ""


def test_rescore_leaves_current_rows_alone_and_rebuilds_each_stale_day_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bats, pens, sps, hands = _tables()
    built: list[Date] = []
    monkeypatch.setattr(ws, "season_pitches", lambda cache, day: built.append(day) or pd.DataFrame(
        {"pitcher": [1, 2], "p_throws": ["L", "R"]}))
    monkeypatch.setattr(ws, "_prepare", lambda df: df)
    monkeypatch.setattr(ws, "bullpen_ranking", lambda as_of: pens)
    monkeypatch.setattr(ws, "offense_rankings", lambda df, as_of: bats)
    monkeypatch.setattr(ws, "starter_table", lambda df, as_of: sps)
    current = _row(game_pk=8, weights=ws.WEIGHTS_VERSION, gap=1.0)
    rows = [_row(), _row(game_pk=9, game="COL @ SD", away="COL", home="SD", away_sp="Scrub", home_sp="Ace"), current]
    out, notes = rs.rescore(rows, tmp_path)
    assert built == [Date(2026, 9, 20)] and len(notes) == 2
    assert out[2] is current and all(r.weights == ws.WEIGHTS_VERSION for r in out)
    assert "fav SD->COL" in notes[0]


def test_fg_season_table_is_a_date_range_split_only_for_a_past_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ws, "leaderboard", lambda url, key: calls.append((url, key)) or [])
    today = Date.today()
    ws._fg("rel", today.year, None, today)
    ws._fg("rel", 2026, None, Date(2026, 9, 19))
    ws._fg("rel", 2026, 60, Date(2026, 9, 19))
    assert "month=0" in calls[0][0] and "startdate" not in calls[0][0] and calls[0][1] == f"rel_{today.year}_w0_t0"
    assert "startdate=2026-03-01&enddate=2026-09-19" in calls[1][0] and calls[1][1] == "rel_2026_w0_t0_2026-09-19"
    assert "startdate=2026-07-21&enddate=2026-09-19" in calls[2][0] and calls[2][1] == "rel_2026_w60_t0_2026-09-19"
