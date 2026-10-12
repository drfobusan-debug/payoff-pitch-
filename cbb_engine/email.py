"""Plain-text email for the card and the 03:00 audit (same Gmail/SMTP env as the other engines)."""

from __future__ import annotations

import html
import mimetypes
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

import certifi

from cbb_engine.config import Config


class EmailNotConfigured(RuntimeError):
    pass


def addresses(cfg: Config, to: str | None = None) -> tuple[str, str]:
    d = cfg.delivery
    if not d.gmail_app_password:
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    recipient = to or d.email_to or d.gmail_user
    sender = d.gmail_user or recipient
    if not recipient or not sender:
        raise EmailNotConfigured("no recipient/sender (CBBE_EMAIL_TO, GMAIL_USER)")
    return sender, recipient


def build(sender: str, recipient: str, subject: str, text: str, paths: dict[str, Path]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(text)
    msg.add_alternative(f"<pre style='font-family:monospace;font-size:12px'>{html.escape(text)}</pre>", subtype="html")
    for path in paths.values():
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
        msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)
    return msg


def send(cfg: Config, *, subject: str, text: str, paths: dict[str, Path] | None = None, to: str | None = None) -> str:
    sender, recipient = addresses(cfg, to)
    msg = build(sender, recipient, subject, text, paths or {})
    context = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(cfg.delivery.smtp_host, cfg.delivery.smtp_port, context=context) as server:
        server.login(sender, (cfg.delivery.gmail_app_password or "").replace(" ", ""))
        server.send_message(msg)
    return recipient


__all__ = ["EmailNotConfigured", "addresses", "build", "send"]
