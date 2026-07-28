"""Email delivery for the daily card via SMTP (Gmail App Password by default)."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

from mlb_engine.config import Config


class EmailNotConfigured(RuntimeError):
    """Raised when an email send is requested without the required credentials."""


def send_card_email(
    cfg: Config,
    *,
    subject: str,
    html_body: str,
    text_body: str,
    to: str | None = None,
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> str:
    """Send the card as a multipart HTML email. Returns the recipient address.

    ``attachments`` is a list of ``(filename, data, subtype)`` (e.g.
    ``("card.md", b"...", "markdown")``). Raises :class:`EmailNotConfigured`
    when the SMTP password or recipient is missing.
    """
    creds = cfg.creds
    if not creds.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    recipient = to or cfg.email_to or creds.gmail_user or cfg.audit_email
    # The App Password belongs to one mailbox, so the recipient is the safest
    # sender fallback when GMAIL_USER is unset -- Gmail rejects a mismatched From.
    sender = creds.gmail_user or recipient
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
    for filename, data, subtype in attachments or []:
        msg.add_attachment(data, maintype="text", subtype=subtype, filename=filename)

    # Gmail App Passwords are shown grouped in 4s; strip any spaces the user kept.
    password = creds.gmail_app_password.replace(" ", "")
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    return recipient
