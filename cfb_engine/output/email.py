"""Email delivery via SMTP (Gmail App Password by default)."""

from __future__ import annotations

import mimetypes
import smtplib
import ssl
from email.message import EmailMessage

import certifi

from cfb_engine.config import Config

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
    """Send a multipart HTML email with optional attachments. Returns recipient.

    The MIME type of each ``(filename, data)`` attachment is inferred from the
    extension (``.pdf`` -> application/pdf, ``.xlsx`` -> spreadsheet, ``.mp3`` ->
    audio/mpeg), defaulting to ``application/octet-stream``. Raises
    :class:`EmailNotConfigured` when the SMTP password or recipient is missing.
    """
    creds = cfg.creds
    if not creds.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    recipient = to or cfg.email_to or creds.gmail_user or cfg.audit_email
    sender = creds.gmail_user or recipient
    if not recipient:
        raise EmailNotConfigured("no recipient set (pass --to or CFBE_EMAIL_TO)")
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

    # Gmail App Passwords are displayed in groups of four; strip any spaces.
    password = creds.gmail_app_password.replace(" ", "")
    context = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    return recipient
