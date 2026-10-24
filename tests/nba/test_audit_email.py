"""The weekly NBA audit email: last 7 slates, then the season, from graded rows only."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from nba_engine import cli
from nba_engine.audit import ledger
from nba_engine.output import email as email_mod
from tests.nba.test_output import DAY, priced

ROOT = Path(__file__).resolve().parents[2]
OLD = DAY - timedelta(days=10)


class FakeSMTP:
    sent: list = []

    def __init__(self, host, port, context=None, timeout=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        pass

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


def graded(day, tag):
    rows = [replace(r, slate_date=day.isoformat()) for r in priced(tag)]
    return [
        replace(r, outcome="win" if i % 2 else "loss", pnl=0.9 if i % 2 else -1.0, graded_at="x")
        for i, r in enumerate(rows)
    ]


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("NBAE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NBAE_STATE_SYNC", "0")
    monkeypatch.delenv("NBAE_PRESEASON", raising=False)
    for var in ("GMAIL_APP_PASSWORD", "GMAIL_USER", "EMAIL_ADDRESS", "NBAE_EMAIL_TO"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli, "_today", lambda: DAY + timedelta(days=1))
    FakeSMTP.sent = []
    monkeypatch.setattr(email_mod.smtplib, "SMTP_SSL", FakeSMTP)
    for day, tag in ((OLD, "old"), (DAY, "new")):
        ledger.write_once(graded(day, tag), ledger.graded_path(tmp_path, day))
    return tmp_path


def test_weekly_audit_mails_the_week_then_the_season(scratch, monkeypatch, capsys):
    monkeypatch.setenv("GMAIL_USER", "me@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    week, season = len(graded(DAY, "new")), 2 * len(graded(DAY, "new"))

    assert (
        cli.main(["audit", "--recent", "7", "--email", "--to", "you@example.com", "--no-sync"]) == 0
    )

    (msg,) = FakeSMTP.sent
    assert msg["To"] == "you@example.com"
    assert "weekly ledger audit" in msg["Subject"] and "[paper]" in msg["Subject"]
    body = msg.get_body(("plain",)).get_content()
    recent, _, rest = body.partition("== Season to date ==")
    assert recent.startswith(f"== Last 7 days ({DAY - timedelta(days=6)} .. {DAY}) ==")
    assert f"{week} graded row(s) over 1 slate(s), {DAY} .. {DAY}" in recent
    assert f"{season} graded row(s) over 2 slate(s), {OLD} .. {DAY}" in rest
    assert "<pre" in msg.get_body(("html",)).get_content()
    assert "emailed the audit to you@example.com" in capsys.readouterr().out


def test_a_week_without_grades_still_mails_the_season(scratch, monkeypatch):
    monkeypatch.setenv("GMAIL_USER", "me@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setattr(cli, "_today", lambda: DAY + timedelta(days=30))

    assert cli.main(["audit", "--recent", "7", "--email", "--no-sync"]) == 0

    body = FakeSMTP.sent[0].get_body(("plain",)).get_content()
    assert "no graded ledger rows in these days" in body
    assert f"over 2 slate(s), {OLD} .. {DAY}" in body


def test_audit_email_without_credentials_prints_and_returns_zero(scratch, capsys):
    assert cli.main(["audit", "--recent", "7", "--email", "--no-sync"]) == 0
    out = capsys.readouterr().out
    assert "== Season to date ==" in out and "audit email not sent" in out
    assert FakeSMTP.sent == []


def test_plain_audit_has_no_section_headers_and_sends_nothing(scratch, capsys):
    assert cli.main(["audit", "--no-sync"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{2 * len(graded(DAY, 'new'))} graded row(s)") and "==" not in out
    assert FakeSMTP.sent == []


def test_installer_schedules_the_weekly_audit(tmp_path):
    repo = tmp_path / "payoff-pitch-"
    (repo / "scripts").mkdir(parents=True)
    for sub in ("nba", "macos"):
        shutil.copytree(ROOT / "scripts" / sub, repo / "scripts" / sub)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "launchctl").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "launchctl").chmod(0o755)
    env = dict(os.environ, HOME=str(tmp_path), PATH=f"{bin_dir}:/usr/bin:/bin")
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "nba" / "macos" / "install_schedule.command")],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    plist = (tmp_path / "Library" / "LaunchAgents" / "com.payoffpitch.nba.audit.plist").read_text()
    assert f"{repo}/scripts/nba/macos/nba_audit.command" in plist
    assert "<key>Weekday</key><integer>1</integer>" in plist
    assert "<key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer>" in plist
    assert "Weekly audit email on weekday 1 (0=Sun) at 09:00" in proc.stdout
