"""SMTP delivery of the priced card (Gmail App Password by default).

Same shape and credentials as the MLB/NFL/CFB senders: one ``engine.env`` on the
machine serves every engine. The text card is the body; the txt/md/xlsx files
the card run already wrote are attached as-is.
"""

from __future__ import annotations

import html
import mimetypes
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

import certifi

from nhl_engine.config import Config
from nhl_engine.outputs import render_card
from nhl_engine.pipeline import SlateCard

mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
mimetypes.add_type("text/markdown", ".md")


class EmailNotConfigured(RuntimeError):
    """Raised when a send is requested without the credentials to make it."""


def subject_for(card: SlateCard) -> str:
    buys = sum(1 for r in card.rows if r.is_buy)
    return f"NHL card {card.slate_date} [{card.tag}] -- {buys} buy{'s' if buys != 1 else ''}"


def send_card(
    cfg: Config,
    card: SlateCard,
    paths: dict[str, Path],
    *,
    to: str | None = None,
) -> str:
    """Email the card with its written outputs attached. Returns the recipient.

    Raises :class:`EmailNotConfigured` rather than exiting, so the caller keeps
    the artifacts already on disk.
    """
    creds, delivery = cfg.creds, cfg.delivery
    if not creds.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    recipient = to or delivery.email_to or creds.gmail_user
    sender = creds.gmail_user or recipient
    if not recipient:
        raise EmailNotConfigured("no recipient set (pass --to or NHLE_EMAIL_TO)")
    if not sender:
        raise EmailNotConfigured("no sender set (GMAIL_USER/EMAIL_ADDRESS)")

    text = render_card(card)
    msg = EmailMessage()
    msg["Subject"] = subject_for(card)
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(text)
    msg.add_alternative(
        f"<pre style='font-family:monospace;font-size:12px'>{html.escape(text)}</pre>",
        subtype="html",
    )
    for path in paths.values():
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
        msg.add_attachment(
            path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name
        )

    password = creds.gmail_app_password.replace(" ", "")
    context = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(delivery.smtp_host, delivery.smtp_port, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
    return recipient
