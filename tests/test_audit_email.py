"""Tests for the nightly audit email dispatcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mlb_engine.audit.email import send_audit_summary


def _write_dummy_ledger(path: Path) -> None:
    path.write_bytes(b"PKdummy-xlsx")


def test_send_audit_summary_without_creds_returns_false(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)
    assert (
        send_audit_summary(ledger, "2026-07-24", smtp_server=None, smtp_user=None, smtp_pass=None)
        is False
    )


def test_send_audit_summary_sends_message(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.xlsx"
    _write_dummy_ledger(ledger)

    mock_smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_smtp):
        result = send_audit_summary(
            ledger,
            "2026-07-24",
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
