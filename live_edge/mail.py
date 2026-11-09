"""Email a live-edge report as plain text (same Gmail/SMTP env as the engine emails).

GMAIL_USER (or EMAIL_ADDRESS)   sending address; falls back to the recipient, as cbb_engine does
GMAIL_APP_PASSWORD              Gmail app password
LIVE_EDGE_EMAIL_TO              recipient; falls back through RECIPIENT_KEYS (the engines'
                                addresses), then GMAIL_USER, then the MLB/CFB audit default
SMTP_HOST / SMTP_PORT           default smtp.gmail.com:465 (implicit TLS)
"""

from __future__ import annotations

import html
import os
import smtplib
import ssl
from email.message import EmailMessage

import certifi

RECIPIENT_KEYS = (
    "LIVE_EDGE_EMAIL_TO",
    "CBBE_EMAIL_TO",
    "NBAE_EMAIL_TO",
    "NHLE_EMAIL_TO",
    "NFLE_EMAIL_TO",
    "CFBE_EMAIL_TO",
    "MLBE_EMAIL_TO",
    "MLB_EMAIL_TO",
    "AUDIT_EMAIL",
    "MLBE_AUDIT_EMAIL",
)
DEFAULT_TO = "drfobusan@gmail.com"  # the MLBE_AUDIT_EMAIL / CFBE_AUDIT_EMAIL default


class EmailNotConfigured(RuntimeError):
    pass


def addresses() -> tuple[str, str]:
    if not os.getenv("GMAIL_APP_PASSWORD"):
        raise EmailNotConfigured("GMAIL_APP_PASSWORD is not set")
    user = os.getenv("GMAIL_USER") or os.getenv("EMAIL_ADDRESS")
    to = next((v for k in RECIPIENT_KEYS if (v := os.getenv(k))), user or DEFAULT_TO)
    return user or to, to


def build(sender: str, to: str, subject: str, text: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.set_content(text)
    msg.add_alternative(
        f"<pre style='font-family:monospace;font-size:12px'>{html.escape(text)}</pre>",
        subtype="html",
    )
    return msg


def send(subject: str, text: str) -> str:
    sender, to = addresses()
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    ctx = ssl.create_default_context(cafile=certifi.where())
    with smtplib.SMTP_SSL(host, port, context=ctx) as smtp:
        smtp.login(sender, os.environ["GMAIL_APP_PASSWORD"].replace(" ", ""))
        smtp.send_message(build(sender, to, subject, text))
    return to
