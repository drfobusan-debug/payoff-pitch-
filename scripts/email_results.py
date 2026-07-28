"""Email the latest MLB results workbook as an attachment.

Configuration is read from environment variables (set them in your shell
profile, e.g. ~/.zprofile, so non-interactive launcher runs can see them):

    MLB_EMAIL_TO         recipient address (defaults to GMAIL_USER)
    GMAIL_USER           sending Gmail address
    GMAIL_APP_PASSWORD   16-char Gmail app password (NOT your login password)

Optional overrides for non-Gmail SMTP servers:

    SMTP_HOST  (default smtp.gmail.com)
    SMTP_PORT  (default 465, implicit TLS/SSL)

Usage:
    python scripts/email_results.py                # newest workbook in output dir
    python scripts/email_results.py /path/file.xlsx
"""

from __future__ import annotations

import glob
import os
import smtplib
import ssl
import sys
from datetime import date
from email.message import EmailMessage
from pathlib import Path

from mlb_engine.config import load_config

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _output_dir() -> Path:
    return load_config().output_dir


def _latest_workbook() -> Path | None:
    pattern = str(_output_dir() / "mlb_recommendations_*.xlsx")
    files = [f for f in glob.glob(pattern) if "/~$" not in f and not Path(f).name.startswith("~$")]
    if not files:
        return None
    return Path(max(files, key=lambda f: os.path.getmtime(f)))


def send(path: Path) -> int:
    user = os.environ.get("GMAIL_USER", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    to_addr = os.environ.get("MLB_EMAIL_TO", user).strip()
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.environ.get("SMTP_PORT", "465"))

    if not user or not password:
        print(
            "ERROR: set GMAIL_USER and GMAIL_APP_PASSWORD (Gmail app password) "
            "in your shell profile to enable email.",
            file=sys.stderr,
        )
        return 2
    if not to_addr:
        print("ERROR: no recipient (set MLB_EMAIL_TO or GMAIL_USER).", file=sys.stderr)
        return 2
    if not path.exists():
        print(f"ERROR: workbook not found: {path}", file=sys.stderr)
        return 2

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = f"MLB predictions {date.today().isoformat()}"
    msg.set_content(
        "Attached is today's MLB predictions workbook "
        "(green Strong / yellow Moderate buy tabs).\n\n"
        f"File: {path.name}\n\nSent automatically by the MLB prediction engine."
    )
    msg.add_attachment(
        path.read_bytes(),
        maintype="application",
        subtype=_XLSX_MIME.split("/", 1)[1],
        filename=path.name,
    )

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=ctx) as smtp:
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"Emailed {path.name} -> {to_addr}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        target: Path | None = Path(argv[1])
    else:
        target = _latest_workbook()
    if target is None:
        print("ERROR: no workbook found to email (run predictions first).", file=sys.stderr)
        return 2
    return send(target)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
