"""Email delivery for the daily card via SMTP (Gmail App Password by default)."""

from __future__ import annotations

import mimetypes
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

from mlb_engine.config import Config

try:
    import certifi
except Exception:  # pragma: no cover - certifi is bundled with requests
    certifi = None  # type: ignore[assignment]


def _ssl_context() -> ssl.SSLContext:
    """Create a TLS context that uses certifi's CA bundle when the system lacks one."""
    if certifi is not None and Path(certifi.where()).exists():
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()

mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
)
mimetypes.add_type("text/markdown", ".md")


class EmailNotConfigured(RuntimeError):
    """Raised when an email send is requested without the required credentials."""


def send_card_email(
    cfg: Config,
    *,
    subject: str,
    html_body: str,
    text_body: str,
    to: str | None = None,
    attachments: list[tuple[str, bytes]] | None = None,
) -> str:
    """Send the card as a multipart HTML email. Returns the recipient address.

    ``attachments`` is a list of ``(filename, data)``; the MIME type is inferred
    from the filename extension (e.g. ``.pdf`` -> application/pdf, ``.xlsx`` ->
    a spreadsheet), falling back to ``application/octet-stream``. Raises
    :class:`EmailNotConfigured` when the SMTP password or recipient is missing.
    """
    creds = cfg.creds
    if not creds.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    sender = creds.gmail_user or to or cfg.email_to
    recipient = to or cfg.email_to or sender
    if not recipient:
        raise EmailNotConfigured("no recipient set (pass --to or MLBE_EMAIL_TO)")
    if not sender:
        raise EmailNotConfigured("no sender set (GMAIL_USER/EMAIL_ADDRESS)")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    for filename, data in attachments or []:
        ctype, _ = mimetypes.guess_type(filename)
        maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

    # Gmail App Passwords are shown grouped in 4s; strip any spaces the user kept.
    password = creds.gmail_app_password.replace(" ", "")
    context = _ssl_context()
    with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    return recipient
