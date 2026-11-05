"""Band context reads each game as of its own tip day, not a later slate that already holds its result."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import pytest

from cbb_engine import bands, cli
from cbb_engine.data import espn, torvik
from engine_runtime import ledger

NOV2, NOV3 = Date(2026, 11, 2), Date(2026, 11, 3)


def _row(slate: str, tip: str) -> ledger.Row:
    return ledger.Row(slate_date=slate, event_id="e", tip_utc=tip, market="game_total", side="under", line=140.5,
                      book="dk", pass_id="p", pass_kind="board", bet_of_record=True, status="research")


def test_tip_day_is_the_us_calendar_day() -> None:
    assert bands.tip_day("2026-11-03T00:30:00Z") == NOV2
    assert bands.tip_day("2026-11-03T18:00:00Z") == NOV3
    assert bands.tip_day("") is None


def test_a_game_on_two_slates_never_reads_its_own_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A Nov 2 evening tip is filed on the Nov 1 and Nov 3 slates; its own slow game is in the logs.
    logs = [bands.LogGame(NOV2, "Kentucky", 50, 50, 45), bands.LogGame(NOV2, "Toledo", 50, 45, 50),
            bands.LogGame(Date(2026, 11, 1), "Duke", 80, 90, 80)]
    board = [espn.Game("1", "2026-11-03T00:30Z", "Toledo Rockets", "Kentucky Wildcats", "STATUS_FINAL",
                       False, False, (20, 25), (20, 25))]
    monkeypatch.setattr(cli, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(cli.torvik, "as_of", lambda *a: torvik.AsOf(None, None, None))
    monkeypatch.setattr(cli.bands, "parse_logs", lambda _rows: logs)
    monkeypatch.setattr(cli.teamstats, "read_logs", lambda _path: [])
    monkeypatch.setattr(cli.quotes, "read_day", lambda *a: [])
    monkeypatch.setattr(cli, "_manifest_teams", lambda _root, _d: {"e": ("Kentucky Wildcats", "Toledo Rockets")})
    monkeypatch.setattr(cli.history, "latest_scoreboard", lambda _dir, d: board if d == NOV2.isoformat() else [])
    season = [_row("2026-11-01", "2026-11-03T00:30:00Z"), _row("2026-11-03", "2026-11-03T00:30:00Z")]
    ctx, _ = cli._season_ctx(tmp_path, season, NOV3)
    assert ctx["e"].home.slow is None and ctx["e"].away.slow is None
