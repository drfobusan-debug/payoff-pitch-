from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from test_stoppage import START, _iso, board, game

from live_edge import mail, stoppage
from live_edge.stoppage import Study, email_slate, slate

ROOT = Path(__file__).resolve().parents[2]


def _cbb_ticket(s: Study) -> None:
    t0 = START - timedelta(minutes=30)
    pre = {"draftkings": 140, "betmgm": 140, "fanduel": 140}
    s.tick([game("pre", 0, 0, 1, 1200, "STATUS_SCHEDULED")], board(pre, t0), t0)
    lines = {"draftkings": 151, "betmgm": 151, "fanduel": 151}
    for m in (60, 62):
        t = START + timedelta(minutes=m)
        s.tick([game("in", 40, 32, 1)], board(lines, t), t)
    s.tick([game("post", 75, 70, 2, 0, "STATUS_FINAL")], None, START + timedelta(hours=3))


def test_the_slate_email_lists_this_runs_tickets_then_the_running_report(tmp_path: Path):
    s = Study("cbb", tmp_path)
    _cbb_ticket(s)
    subject, body = slate(s, _iso(START - timedelta(hours=1)))
    assert subject == "CBB live-total paper test 2026-11-04: 1 tickets, 1 settled"
    assert "1 games at the stoppage, 1 tickets" in body
    assert "under 151 (draftkings -110), live 151 fair 146.944 gap +4.056; final 145 win" in body
    assert "lead-adjusted under win" in body
    assert body.index("Running totals") < body.index("CBB live-total test (paper only)")
    # A later run starts clean: the ticket is in the running totals, not in this run.
    _, later = slate(s, _iso(START + timedelta(days=1)))
    assert "0 games at the stoppage, 0 tickets" in later and "settled: 1" in later


def test_no_gmail_password_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    assert email_slate(Study("cbb", tmp_path), "2026-11-05") == 2
    assert "GMAIL_APP_PASSWORD is not set" in capsys.readouterr().err


def test_the_recipient_falls_back_through_the_engine_addresses(monkeypatch):
    for k in (*mail.RECIPIENT_KEYS, "EMAIL_ADDRESS", "GMAIL_USER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "x")
    # Only the password set: the audit default, sent from itself (as cbb_engine does).
    assert mail.addresses() == (mail.DEFAULT_TO, mail.DEFAULT_TO)
    monkeypatch.setenv("MLB_EMAIL_TO", "mlb@x.com")
    assert mail.addresses() == ("mlb@x.com", "mlb@x.com")
    monkeypatch.delenv("MLB_EMAIL_TO")
    monkeypatch.setenv("GMAIL_USER", "me@x.com")
    assert mail.addresses() == ("me@x.com", "me@x.com")
    monkeypatch.setenv("CBBE_EMAIL_TO", "cbb@x.com")
    assert mail.addresses() == ("me@x.com", "cbb@x.com")
    monkeypatch.setenv("LIVE_EDGE_EMAIL_TO", "live@x.com")
    assert mail.addresses() == ("me@x.com", "live@x.com")
    msg = mail.build("me@x.com", "live@x.com", "s", "a < b")
    assert "a &lt; b" in msg.as_string()


class _NoOdds:
    def fetch(self, sport, markets):  # pragma: no cover - no board is needed here
        raise AssertionError("no Odds API call expected")


def test_the_loop_waits_for_the_first_tip_then_exits_when_idle(tmp_path, monkeypatch):
    # Pre-game hours before tip (outside the 1h and 45m windows), then live, then final.
    far = START + timedelta(hours=5)
    pre = replace(game("pre", 0, 0, 1, 1200, "STATUS_SCHEDULED"), kickoff=_iso(far))
    boards = (
        [[pre]] * 30
        + [[game("in", 20, 18, 1, 600, "STATUS_IN_PROGRESS")]]
        + [[game("post", 75, 70, 2, 0, "STATUS_FINAL")]] * 40
    )
    calls = iter(boards)
    monkeypatch.setattr(stoppage, "fetch_scoreboard", lambda sport: next(calls))
    monkeypatch.setattr(stoppage.time, "sleep", lambda s: None)
    stoppage.run(
        Study("cbb", tmp_path), _NoOdds(), interval=1, hours=1, max_credits=0, idle_exit=10
    )
    left = list(calls)
    # 30 waiting polls did not count as idle; it stopped 10 polls after the final.
    assert len(left) == 40 - 10


def test_an_empty_or_stale_board_before_the_first_tip_is_not_idle(tmp_path, monkeypatch):
    # 11-07: launchd started at 10:00 ET, ESPN showed nothing for 13 min and the loop quit
    # before the noon tips. Empty boards, failed fetches and yesterday's finals all wait.
    stale = game("post", 75, 70, 2, 0, "STATUS_FINAL")
    boards = (
        [[]] * 15
        + [[stale]] * 15
        + [None]  # a raised fetch
        + [[game("in", 20, 18, 1, 600, "STATUS_IN_PROGRESS")]]
        + [[game("post", 75, 70, 2, 0, "STATUS_FINAL")]] * 40
    )
    calls = iter(boards)

    def board(sport):
        b = next(calls)
        if b is None:
            raise RuntimeError("espn down")
        return b

    monkeypatch.setattr(stoppage, "fetch_scoreboard", board)
    monkeypatch.setattr(stoppage.time, "sleep", lambda s: None)
    stoppage.run(
        Study("cbb", tmp_path), _NoOdds(), interval=1, hours=1, max_credits=0, idle_exit=10
    )
    assert len(list(calls)) == 40 - 10


def test_the_hours_deadline_bounds_a_day_with_no_games(tmp_path, monkeypatch):
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(stoppage.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(stoppage.time, "sleep", lambda s: None)
    n = 0

    def board(sport):
        nonlocal n
        n += 1
        return []

    monkeypatch.setattr(stoppage, "fetch_scoreboard", board)
    stoppage.run(
        Study("cbb", tmp_path), _NoOdds(), interval=1, hours=0.1, max_credits=0, idle_exit=10
    )
    assert 0 < n <= 5


def _install(repo: Path, home: Path) -> subprocess.CompletedProcess[str]:
    bin_dir = home / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "launchctl").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "launchctl").chmod(0o755)
    env = dict(os.environ, HOME=str(home), PATH=f"{bin_dir}:/usr/bin:/bin", LIVE_CBB_WEEKDAYS="5 6")
    script = repo / "scripts" / "live_edge" / "macos" / "install_schedule.command"
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env, timeout=60
    )


def _checkout(at: Path) -> Path:
    repo = at / "payoff-pitch-"
    (repo / "scripts").mkdir(parents=True)
    for d in ("live_edge", "macos"):
        shutil.copytree(ROOT / "scripts" / d, repo / "scripts" / d)
    return repo


@pytest.mark.parametrize("protected", [False, True])
def test_the_installer_writes_the_saturday_agent_or_refuses_a_protected_checkout(
    tmp_path: Path, protected: bool
):
    repo = _checkout(tmp_path / "Documents" if protected else tmp_path)
    proc = _install(repo, tmp_path)
    plist = tmp_path / "Library" / "LaunchAgents" / "com.payoffpitch.live.cbb.plist"
    if protected:
        assert proc.returncode == 1 and not plist.exists()
        return
    assert proc.returncode == 0, proc.stderr
    text = plist.read_text()
    assert f"{repo}/scripts/live_edge/macos/live_cbb.command" in text
    assert text.count("<key>Weekday</key>") == 2 and "<integer>10</integer>" in text


def test_the_saturday_command_runs_the_cbb_test_with_email():
    cmd = (ROOT / "scripts" / "live_edge" / "macos" / "live_cbb.command").read_text()
    assert "live_edge.stoppage run --sport cbb" in cmd and "--email" in cmd
    assert os.access(ROOT / "scripts" / "live_edge" / "macos" / "live_cbb.command", os.X_OK)
