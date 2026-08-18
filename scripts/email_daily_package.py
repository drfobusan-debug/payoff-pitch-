"""Email the full daily package as a single message.

Gathers the day's already-generated artifacts from the engine output dir and
sends them in one email (same Gmail App Password credentials the engine's
``run --email`` path uses):

    * mlb_recommendations_<day>.xlsx   (bet card)
    * PayoffPitch_Slate_<day>.pdf/.mp3 (slate preview article + audio)
    * PayoffPitch_Regression_<day>.pdf (combined regression article, arms + bats)
    * PayoffPitch_Mound_<day>.pdf      (pitcher regression stat cards)
    * PayoffPitch_Batter_<day>.pdf     (batter regression stat cards)
    * PayoffPitch_Regression_<day>.mp3 (combined regression narration)
    * regression_radar_<day>.pdf       (regression radar, if present)
    * power_screen_<day>.pdf           (morning power screen, if present)

Usage:
    python -m scripts.email_daily_package 2026-08-02   # explicit slate date
    python -m scripts.email_daily_package              # newest bet card on disk
    python -m scripts.email_daily_package --dry-run    # list attachments, don't send
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import date as Date
from pathlib import Path

from mlb_engine.config import load_config
from mlb_engine.output.email import EmailNotConfigured, send_card_email


def _resolve_day(out_dir: Path, argv: list[str]) -> Date | None:
    for arg in argv[1:]:
        if not arg.startswith("-"):
            return Date.fromisoformat(arg)
    cards = [
        f
        for f in glob.glob(str(out_dir / "mlb_recommendations_*.xlsx"))
        if not Path(f).name.startswith("~$")
    ]
    if not cards:
        return None
    newest = Path(max(cards, key=os.path.getmtime)).stem
    return Date.fromisoformat(newest.replace("mlb_recommendations_", ""))


def collect_attachments(out_dir: Path, day: Date) -> list[tuple[str, bytes]]:
    """Return (filename, bytes) for every artifact that exists for ``day``."""
    iso = day.isoformat()
    candidates = [
        f"mlb_recommendations_{iso}.xlsx",
        f"PayoffPitch_Slate_{iso}.pdf",
        f"PayoffPitch_Slate_{iso}.mp3",
        f"PayoffPitch_Regression_{iso}.pdf",
        f"PayoffPitch_Mound_{iso}.pdf",
        f"PayoffPitch_Batter_{iso}.pdf",
        f"PayoffPitch_Regression_{iso}.mp3",
        f"regression_radar_{iso}.pdf",
        f"power_screen_{iso}.pdf",
    ]
    attachments: list[tuple[str, bytes]] = []
    for name in candidates:
        path = out_dir / name
        if path.exists():
            attachments.append((name, path.read_bytes()))
    return attachments


def main(argv: list[str]) -> int:
    cfg = load_config()
    out_dir = cfg.output_dir
    day = _resolve_day(out_dir, argv)
    if day is None:
        print("ERROR: no slate date given and no bet card found.", file=sys.stderr)
        return 2

    attachments = collect_attachments(out_dir, day)
    if not attachments:
        print(f"ERROR: no artifacts found for {day} in {out_dir}.", file=sys.stderr)
        return 2

    names = [name for name, _ in attachments]
    if "--dry-run" in argv:
        print(f"Would email {len(names)} attachment(s) for {day}:")
        for name in names:
            print(f"  {name}")
        return 0

    nice = day.strftime("%A, %B %-d, %Y")
    html_body = (
        f"<p>Good morning — here's the full Payoff Pitch package for <b>{nice}</b>.</p>"
        "<p>Attached: the bet card (Excel), the slate preview article + audio, the "
        "combined regression article, and the Mound/Batter stat cards + narration.</p>"
        "<ul>" + "".join(f"<li>{name}</li>" for name in names) + "</ul>"
    )
    text_body = (
        f"Payoff Pitch daily package for {nice}.\n\n"
        "Attached:\n" + "\n".join(f"  - {name}" for name in names) + "\n"
    )
    try:
        recipient = send_card_email(
            cfg,
            subject=f"Payoff Pitch — daily package {day.isoformat()}",
            html_body=html_body,
            text_body=text_body,
            attachments=attachments,
        )
    except EmailNotConfigured as exc:
        print(f"ERROR: email not configured: {exc}", file=sys.stderr)
        return 2
    print(f"Emailed {len(names)} attachment(s) for {day} -> {recipient}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
