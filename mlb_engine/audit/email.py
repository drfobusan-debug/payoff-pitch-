"""Nightly audit email dispatch via SMTP."""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
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

    Credentials come from explicit arguments, then a generic SMTP relay
    (``SMTP_SERVER``/``SMTP_PORT``/``SMTP_USER``/``SMTP_PASS``), and finally the
    same Gmail App Password the daily card uses (``GMAIL_APP_PASSWORD``, sender
    ``GMAIL_USER`` falling back to the recipient). The recipient defaults to
    ``cfg.audit_email``/``AUDIT_EMAIL``.

    A missing credential is an error, not a silent skip: an audit that cannot
    reach an inbox looks identical to an audit that never ran.
    """
    cfg = cfg or Config()
    recipient = to or cfg.audit_email or os.getenv("AUDIT_EMAIL")
    server = smtp_server or cfg.creds.smtp_server or os.getenv("SMTP_SERVER")
    port = smtp_port or cfg.creds.smtp_port or int(os.getenv("SMTP_PORT", "587"))
    user = smtp_user or cfg.creds.smtp_user or os.getenv("SMTP_USER")
    password = smtp_pass or cfg.creds.smtp_pass or os.getenv("SMTP_PASS")

    # Gmail App Password path: one credential drives both the card and the audit.
    gmail = False
    if not (server and user and password) and cfg.creds.gmail_app_password:
        server, port = cfg.smtp_host, cfg.smtp_port
        user = cfg.creds.gmail_user or recipient
        password = cfg.creds.gmail_app_password.replace(" ", "")
        gmail = True

    if not recipient or not server or not user or not password:
        missing = ", ".join(
            name
            for name, value in (
                ("recipient", recipient),
                ("server", server),
                ("user", user),
                ("password", password),
            )
            if not value
        )
        log.error(
            "audit email not sent: missing %s. Set GMAIL_APP_PASSWORD (+ GMAIL_USER) or "
            "SMTP_SERVER/SMTP_USER/SMTP_PASS, and MLBE_AUDIT_EMAIL for the recipient.",
            missing,
        )
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
        if gmail or port == 465:
            with smtplib.SMTP_SSL(
                server, port, context=ssl.create_default_context(), timeout=30
            ) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(server, port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(user, password)
                smtp.send_message(msg)
        log.info("audit email sent to %s", recipient)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("failed to send audit email to %s: %s", recipient, exc)
        return False
