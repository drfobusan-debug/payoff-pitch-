"""The weekly NFL ledger audit email: report table + season audit, ledger attached."""

from __future__ import annotations

from typing import cast

import pytest

from nfl_engine.audit.ledger import LedgerEntry, save_ledger
from nfl_engine.output import email as email_mod
from nfl_engine.output.email import EmailNotConfigured
from scripts.nfl import ledger_audit


def row(**overrides: object) -> LedgerEntry:
    entry = LedgerEntry(
        season=2026,
        week=1,
        date="2026-09-13",
        matchup="BUF @ KC",
        market="moneyline",
        side="KC",
        line=None,
        book="dk",
        odds=110.0,
        opposite_odds=-130.0,
        tier="Pass",
        model_prob=0.55,
        fair_prob=0.52,
        ev_model=0.155,
        ev_fair=0.092,
        paired_books=3,
        captured_at="2026-09-09T18:00:00Z",
        kickoff_utc="2026-09-13T17:00:00Z",
        result="win",
        pnl=1.1,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch, tmp_path):
    path = tmp_path / "nfl_ledger.csv"
    monkeypatch.setattr(ledger_audit, "ledger_path", lambda root=None: path)
    monkeypatch.setattr(ledger_audit, "_final_scores", lambda season: {})
    return path


def test_email_sends_the_report_and_the_audit_with_the_ledger(
    monkeypatch: pytest.MonkeyPatch, ledger
) -> None:
    save_ledger(ledger, [row(), row(side="BUF", odds=-130.0, opposite_odds=110.0, result="loss")])
    sent: dict[str, object] = {}

    def capture(cfg, **kwargs):
        sent.update(kwargs)
        return "me@example.com"

    monkeypatch.setattr(email_mod, "send_package", capture)
    ledger_audit.main(["--season", "2026", "--email"])
    body = str(sent["text_body"])
    assert "nfl-engine report --all" in body
    assert "1) Every buy the engine made" in body
    assert "2026" in str(sent["subject"])
    attachments = cast(list[tuple[str, bytes]], sent["attachments"])
    assert [name for name, _ in attachments] == ["nfl_audit_2026.txt", "nfl_ledger.csv"]
    assert dict(attachments)["nfl_ledger.csv"] == ledger.read_bytes()


def test_a_season_with_no_rows_emails_nothing(monkeypatch: pytest.MonkeyPatch, ledger) -> None:
    save_ledger(ledger, [row(season=2025)])

    def refuse(cfg, **kwargs):
        raise AssertionError("an empty season must not be emailed")

    monkeypatch.setattr(email_mod, "send_package", refuse)
    ledger_audit.main(["--season", "2026", "--email"])


def test_missing_credentials_do_not_fail_the_run(
    monkeypatch: pytest.MonkeyPatch, ledger, capsys
) -> None:
    save_ledger(ledger, [row()])

    def unconfigured(cfg, **kwargs):
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")

    monkeypatch.setattr(email_mod, "send_package", unconfigured)
    ledger_audit.main(["--season", "2026", "--email"])
    assert "audit not emailed" in capsys.readouterr().out


def test_without_email_the_report_table_is_not_added(ledger, capsys) -> None:
    save_ledger(ledger, [row()])
    ledger_audit.main(["--season", "2026"])
    out = capsys.readouterr().out
    assert "1) Every buy the engine made" in out
    assert "nfl-engine report --all" not in out


def test_the_audit_grades_the_tracked_rules_on_the_leans(ledger, capsys) -> None:
    dog = row(
        market="spread",
        side="BUF",
        line=3.5,
        odds=-105.0,
        opposite_odds=-115.0,
        model_prob=0.53,
        fair_prob=0.50,
        screens="no_execution_edge",
    )
    save_ledger(ledger, [dog])
    ledger_audit.main(["--season", "2026"])
    out = capsys.readouterr().out
    assert "7) Graded-only rules" in out
    line = next(x for x in out.splitlines() if "spread_dog_gap_2_4" in x)
    assert "WATCHING" in line and "1-0" in line
