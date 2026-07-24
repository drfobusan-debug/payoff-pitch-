"""Nightly audit email dispatch via SMTP."""

from __future__ import annotations

import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from mlb_engine.config import Config

log = logging.getLogger(__name__)


def send_audit_summary(
    ledger_xlsx: Path,
    audit_date: str,
    cfg: Config | None = None,
    *,
    to: str | None = None,
    smtp_server: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_pass: str | None = None,
) -> bool:
    """Email the nightly audit workbook to the configured address.

    Parameters can be provided explicitly or pulled from environment variables:
    ``SMTP_SERVER``, ``SMTP_PORT`` (default 587), ``SMTP_USER``, ``SMTP_PASS``.
    The recipient defaults to ``drfobusan@gmail.com`` unless overridden by
    ``AUDIT_EMAIL`` or ``cfg.audit_email``.
    """
    cfg = cfg or Config()
    recipient = to or cfg.audit_email or os.getenv("AUDIT_EMAIL", "drfobusan@gmail.com")
    server = smtp_server or cfg.creds.smtp_server or os.getenv("SMTP_SERVER")
    port = smtp_port or cfg.creds.smtp_port or int(os.getenv("SMTP_PORT", "587"))
    user = smtp_user or cfg.creds.smtp_user or os.getenv("SMTP_USER")
    password = smtp_pass or cfg.creds.smtp_pass or os.getenv("SMTP_PASS")

    if not recipient or not server or not user or not password:
        log.warning("SMTP credentials not configured; skipping audit email")
        return False

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = recipient
    msg["Subject"] = f"Payoff Pitch Audit {audit_date}"

    body = f"Attached: Payoff Pitch nightly audit ledger for {audit_date}."
    msg.attach(MIMEText(body, "plain"))

    if ledger_xlsx.exists():
        with ledger_xlsx.open("rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={ledger_xlsx.name}",
        )
        msg.attach(part)
    else:
        log.warning("ledger workbook not found at %s; email will have no attachment", ledger_xlsx)

    try:
        with smtplib.SMTP(server, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        log.info("audit email sent to %s", to)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to send audit email: %s", exc)
        return False
