"""SMTP delivery of the weekly package (Gmail App Password by default).

Same shape as the MLB engine's sender, and deliberately the same credentials: one
``engine.env`` on the machine serves both engines, so a rotated app password does
not have to be found twice.
"""

from __future__ import annotations

import mimetypes
import smtplib
import ssl
from email.message import EmailMessage

import certifi

from nfl_engine.config import Config

mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
mimetypes.add_type("text/markdown", ".md")


class EmailNotConfigured(RuntimeError):
    """Raised when a send is requested without the credentials to make it."""


def send_package(
    cfg: Config,
    *,
    subject: str,
    html_body: str,
    text_body: str,
    to: str | None = None,
    attachments: list[tuple[str, bytes]] | None = None,
) -> str:
    """Send the card as a multipart HTML email. Returns the recipient address.

    ``attachments`` is ``(filename, data)``; the MIME type is inferred from the
    extension. Raises :class:`EmailNotConfigured` rather than exiting, so the
    caller can keep the artifacts it already wrote to disk.
    """
    creds, delivery = cfg.creds, cfg.delivery
    if not creds.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    recipient = to or delivery.email_to or creds.gmail_user
    # The App Password belongs to one mailbox, so the recipient is the safest
    # sender fallback when GMAIL_USER is unset -- Gmail rejects a mismatched From.
    sender = creds.gmail_user or recipient
    if not recipient:
        raise EmailNotConfigured("no recipient set (pass --to or NFLE_EMAIL_TO)")
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

    # Gmail shows App Passwords grouped in fours; strip any spaces kept on paste.
    password = creds.gmail_app_password.replace(" ", "")
    context = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(delivery.smtp_host, delivery.smtp_port, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    return recipient
