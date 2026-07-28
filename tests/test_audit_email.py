"""Tests for the nightly audit email dispatcher."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlb_engine.audit.email import send_audit_summary
from mlb_engine.config import Config, Credentials

SMTP_ENV = ("SMTP_SERVER", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "AUDIT_EMAIL")


def _write_dummy_ledger(path: Path) -> None:
    path.write_bytes(b"PKdummy-xlsx")


def _bare_config() -> Config:
    """A config with no credentials of any kind, whatever the ambient env holds."""
    return Config(creds=Credentials(smtp_server=None, smtp_user=None, smtp_pass=None,
                                    gmail_user=None, gmail_app_password=None))


def _gmail_config() -> Config:
    """Only a Gmail App Password configured -- the shipped production setup."""
    return Config(creds=Credentials(smtp_server=None, smtp_user=None, smtp_pass=None,
                                    gmail_user=None, gmail_app_password="abcd efgh ijkl mnop"))


@pytest.fixture(autouse=True)
def _no_ambient_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SMTP_ENV:
        monkeypatch.delenv(name, raising=False)


def test_send_audit_summary_without_creds_returns_false(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)
    assert send_audit_summary(ledger, "2026-07-24", _bare_config()) is False


def test_send_audit_summary_sends_message(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)

    mock_smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_smtp):
        result = send_audit_summary(
            ledger,
            "2026-07-24",
            _bare_config(),
            to="test@example.com",
            smtp_server="smtp.example.com",
            smtp_port=587,
            smtp_user="user",
            smtp_pass="pass",
        )

    assert result is True
    mock_smtp.__enter__.return_value.starttls.assert_called_once()
    mock_smtp.__enter__.return_value.login.assert_called_once_with("user", "pass")
    mock_smtp.__enter__.return_value.send_message.assert_called_once()
    sent = mock_smtp.__enter__.return_value.send_message.call_args[0][0]
    assert sent["To"] == "test@example.com"
    assert "Payoff Pitch Audit 2026-07-24" in sent["Subject"]


def test_gmail_app_password_is_enough(tmp_path: Path) -> None:
    """The audit used to skip silently whenever only GMAIL_APP_PASSWORD was set."""
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)
    cfg = _gmail_config()

    mock_smtp = MagicMock()
    with patch("smtplib.SMTP_SSL", return_value=mock_smtp) as ssl_ctor:
        assert send_audit_summary(ledger, "2026-07-24", cfg) is True

    assert ssl_ctor.call_args[0][0] == cfg.smtp_host
    # Spaces in the grouped App Password must be stripped before login.
    mock_smtp.__enter__.return_value.login.assert_called_once_with(
        cfg.audit_email, "abcdefghijklmnop"
    )
    sent = mock_smtp.__enter__.return_value.send_message.call_args[0][0]
    assert sent["To"] == cfg.audit_email
    # Gmail rejects a From that isn't the authenticated mailbox.
    assert sent["From"] == cfg.audit_email


def test_gmail_user_overrides_the_sender(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)
    cfg = _gmail_config()
    cfg = dataclasses.replace(
        cfg, creds=dataclasses.replace(cfg.creds, gmail_user="engine@example.com")
    )

    mock_smtp = MagicMock()
    with patch("smtplib.SMTP_SSL", return_value=mock_smtp):
        assert send_audit_summary(ledger, "2026-07-24", cfg, to="who@example.com") is True

    sent = mock_smtp.__enter__.return_value.send_message.call_args[0][0]
    assert sent["From"] == "engine@example.com"
    assert sent["To"] == "who@example.com"


def test_explicit_smtp_beats_the_gmail_fallback(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)

    mock_smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_smtp):
        assert send_audit_summary(
            ledger,
            "2026-07-24",
            _gmail_config(),
            smtp_server="relay.example.com",
            smtp_port=587,
            smtp_user="relay-user",
            smtp_pass="relay-pass",
        ) is True

    mock_smtp.__enter__.return_value.login.assert_called_once_with("relay-user", "relay-pass")


def test_send_failure_is_reported_not_swallowed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)

    with patch("smtplib.SMTP_SSL", side_effect=OSError("connection refused")):
        assert send_audit_summary(ledger, "2026-07-24", _gmail_config()) is False
    assert any(r.levelname == "ERROR" for r in caplog.records)
