"""SMTP delivery of the daily package (Gmail App Password by default).

The MLB and NFL senders' shape and credentials (``GMAIL_USER``,
``GMAIL_APP_PASSWORD``), so one ``engine.env`` serves every engine. Raises
:class:`EmailNotConfigured` rather than exiting, so the caller keeps the files
it already wrote. Gmail refuses a message over 25 MB *after* base64, which
inflates an attachment ~1.37x: attachments that would push the message past
``MAX_EMAIL_BYTES`` are left out (last first) and named in the body instead.
"""

from __future__ import annotations

import mimetypes
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage

import certifi

from nba_engine.config import Config

mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")

GMAIL_LIMIT_BYTES = 25_000_000
# Headroom under Gmail's limit for headers, the HTML body and MIME boundaries.
MAX_EMAIL_BYTES = 20_000_000
SMTP_TIMEOUT_S = 120.0


class EmailNotConfigured(RuntimeError):
    """Raised when a send is requested without the credentials to make it."""


@dataclass(frozen=True)
class Sent:
    recipient: str
    attached: tuple[str, ...]
    left_out: tuple[str, ...]  # too large to mail; still on disk
    size: int  # bytes of the message as sent


def attachment_budget(message_bytes: int = MAX_EMAIL_BYTES) -> int:
    """Largest raw attachment that fits a ``message_bytes`` message once encoded.

    base64 writes 4 characters per 3 bytes and breaks lines at 76 characters
    with a CRLF, so a file on disk takes ~1.37x its size in the message.
    """
    return message_bytes * 3 * 76 // (4 * 78)


def _message(
    *,
    subject: str,
    sender: str,
    recipient: str,
    html_body: str,
    text_body: str,
    attachments: list[tuple[str, bytes]],
    left_out: list[str],
) -> EmailMessage:
    note = (
        "\n\nLeft out (too large to email, saved on disk): " + ", ".join(left_out)
        if left_out
        else ""
    )
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(text_body + note)
    html_note = f"<p><i>{note.strip()}</i></p>" if left_out else ""
    msg.add_alternative(html_body.replace("</body>", f"{html_note}</body>", 1), subtype="html")
    for filename, data in attachments:
        ctype, _ = mimetypes.guess_type(filename)
        maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return msg


def fit(
    *,
    subject: str,
    sender: str,
    recipient: str,
    html_body: str,
    text_body: str,
    attachments: list[tuple[str, bytes]],
    max_bytes: int = MAX_EMAIL_BYTES,
) -> tuple[EmailMessage, list[str]]:
    """The message with as many attachments as fit ``max_bytes``, measured as encoded.

    ``attachments`` are in priority order; the last is dropped first.
    """
    keep = list(attachments)
    dropped: list[str] = []
    while True:
        msg = _message(
            subject=subject,
            sender=sender,
            recipient=recipient,
            html_body=html_body,
            text_body=text_body,
            attachments=keep,
            left_out=dropped,
        )
        if len(msg.as_bytes()) <= max_bytes or not keep:
            return msg, dropped
        dropped.insert(0, keep.pop()[0])


def send_package(
    cfg: Config,
    *,
    subject: str,
    html_body: str,
    text_body: str,
    to: str | None = None,
    attachments: list[tuple[str, bytes]] | None = None,
    max_bytes: int = MAX_EMAIL_BYTES,
) -> Sent:
    """Send the card as a multipart HTML email with its files attached."""
    creds, delivery = cfg.creds, cfg.delivery
    if not creds.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    recipient = to or delivery.email_to or creds.gmail_user
    # The App Password belongs to one mailbox, so the recipient is the safest
    # sender fallback when GMAIL_USER is unset -- Gmail rejects a mismatched From.
    sender = creds.gmail_user or recipient
    if not recipient:
        raise EmailNotConfigured("no recipient set (pass --to or NBAE_EMAIL_TO)")
    if not sender:
        raise EmailNotConfigured("no sender set (GMAIL_USER/EMAIL_ADDRESS)")
    msg, left_out = fit(
        subject=subject,
        sender=sender,
        recipient=recipient,
        html_body=html_body,
        text_body=text_body,
        attachments=list(attachments or []),
        max_bytes=max_bytes,
    )
    size = len(msg.as_bytes())
    # Gmail shows App Passwords grouped in fours; strip any spaces kept on paste.
    password = creds.gmail_app_password.replace(" ", "")
    context = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(
        delivery.smtp_host, delivery.smtp_port, context=context, timeout=SMTP_TIMEOUT_S
    ) as server:
        server.login(sender, password)
        server.send_message(msg)
    attached = tuple(name for name, _ in (attachments or []) if name not in left_out)
    return Sent(recipient, attached, tuple(left_out), size)


__all__ = [
    "GMAIL_LIMIT_BYTES",
    "MAX_EMAIL_BYTES",
    "EmailNotConfigured",
    "Sent",
    "attachment_budget",
    "fit",
    "send_package",
]
