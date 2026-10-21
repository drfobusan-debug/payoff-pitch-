"""Schedule bands: rest counted from archived finals + slates, one main-line row per game, graded only."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from nba_engine.audit import ledger, schedule_bands
from nba_engine.audit.settle import pnl
from nba_engine.data.boxes import write_results
from nba_engine.schemas import GameResult


def _row(
    day: str, matchup: str, market: str, side: str, line: float, fair: float, outcome: str
) -> ledger.LedgerRow:
    r = ledger.LedgerRow(
        day,
        f"{day}T23:00:00Z",
        f"ev-{matchup}-{day}",
        "",
        matchup,
        market,
        "incl_ot",
        side,
        "",
        line,
        "draftkings",
        -110.0,
        -110.0,
        2,
        "",
        5,
        5,
        fair,
        None,
        None,
        None,
        "Pass",
        "no_model",
        f"{day}T20:00:00Z",
        "t",
        "",
        outcome=outcome,
    )
    r.pnl = pnl(outcome, -110.0)
    return r


def _final(day: Date, away: str, home: str) -> GameResult:
    return GameResult(
        "x", day, away, home, "STATUS_FINAL", (25, 25, 25, 25), (26, 26, 26, 26), season_type=2
    )


def test_rest_and_home_windows_from_archived_finals(tmp_path: Path) -> None:
    d = Date(2026, 11, 10)
    write_results(
        tmp_path,
        d - 2 * schedule_bands.timedelta(days=1),
        [_final(d - schedule_bands.timedelta(days=2), "BOS", "LAL")],
    )
    write_results(
        tmp_path,
        d - schedule_bands.timedelta(days=5),
        [_final(d - schedule_bands.timedelta(days=5), "MIA", "LAL")],
    )
    rows = [_row("2026-11-10", "BOS @ LAL", "game_ats", "LAL", -3.5, 0.5, "win")]
    rows.append(_row("2026-11-10", "NYK @ CHI", "game_ats", "CHI", -2.5, 0.5, "void"))
    sched = schedule_bands.schedule_from(tmp_path, rows)
    assert sched.rest("BOS", d) == 1 and sched.rest("LAL", d) == 1
    assert sched.rest("CHI", d) is None and d not in sched.home_days.get("CHI", set())
    assert sched.nth_home_in("LAL", d, 4) == 2  # 11-08 and today
    assert sched.rest("MIA", d) == 4 and sched.rest("NYK", d) is None


def test_bands_take_the_main_line_once_per_game_and_only_when_the_spot_holds(
    tmp_path: Path,
) -> None:
    day = Date(2026, 11, 10)
    td = schedule_bands.timedelta
    write_results(tmp_path, day - td(2), [_final(day - td(2), "BOS", "MIA")])
    write_results(tmp_path, day - td(4), [_final(day - td(4), "LAL", "DEN")])
    # DEN hosts its 3rd home game in 4 days today; visitor PHX on 2 days' rest.
    write_results(
        tmp_path,
        day - td(3),
        [_final(day - td(3), "PHX", "UTA"), _final(day - td(3), "MIA", "DEN")],
    )
    write_results(tmp_path, day - td(1), [_final(day - td(1), "ORL", "DEN")])
    rows = [
        _row("2026-11-10", "BOS @ LAL", "game_ats", "LAL", -3.5, 0.50, "win"),
        _row(
            "2026-11-10", "BOS @ LAL", "game_ats", "LAL", -5.5, 0.42, "loss"
        ),  # alt line, not main
        _row("2026-11-10", "BOS @ LAL", "game_ats", "BOS", 3.5, 0.50, "loss"),
        _row("2026-11-10", "PHX @ DEN", "game_total", "under", 228.5, 0.49, "loss"),
        _row("2026-11-10", "PHX @ DEN", "game_total", "over", 228.5, 0.51, "win"),
        _row(
            "2026-11-10", "NYK @ CHI", "game_total", "under", 220.5, 0.5, "win"
        ),  # no rest context
    ]
    sched = schedule_bands.schedule_from(tmp_path, rows)
    ats, under = schedule_bands.read_bands(rows, sched)
    assert (ats.n, ats.record, round(ats.units, 2)) == (1, "1-0", round(pnl("win", -110.0) or 0, 2))
    assert (under.n, under.record) == (1, "0-1")
    assert ats.verdict.startswith("probation (1/100")
    text = "\n".join(schedule_bands.lines([ats, under]))
    assert "graded only" in text and "rest_host_ats" in text and "-> probation" in text
