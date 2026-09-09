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
    * totals_sheet_<day>.xlsx          (hand-method totals sheet, if present)

Usage:
    python -m scripts.email_daily_package 2026-08-02   # explicit slate date
    python -m scripts.email_daily_package              # newest bet card on disk
    python -m scripts.email_daily_package --dry-run    # list attachments, don't send
    python -m scripts.email_daily_package 2026-08-02 --block evening
    python -m scripts.email_daily_package 2026-08-02 --block matinee --with-daily

``--block <name>`` (a ``mlb_engine.slate_blocks`` name) is a slate pass's
delivery: the bet card plus that block's slate article
(``PayoffPitch_Slate_<day>_<block>.pdf/.mp3``) and nothing else. It sends
nothing when the block's article is not on disk -- the pass priced no games.
``--with-daily`` adds the once-a-day pieces (regression articles, radar, power
screen); the runner passes it on the first pass of the day that has games.
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import date as Date
from pathlib import Path

from mlb_engine.config import load_config
from mlb_engine.output.email import EmailNotConfigured, send_card_email
from mlb_engine.slate_blocks import BLOCK_NAMES
from mlb_engine.slate_blocks import block as slate_block


def _resolve_day(out_dir: Path, argv: list[str]) -> Date | None:
    for arg in argv[1:]:
        if not arg.startswith("-") and arg not in BLOCK_NAMES:
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


def _block(argv: list[str]) -> str | None:
    if "--block" not in argv:
        return None
    i = argv.index("--block")
    if i + 1 >= len(argv) or argv[i + 1] not in BLOCK_NAMES:
        raise SystemExit(f"--block takes one of {', '.join(BLOCK_NAMES)}")
    return argv[i + 1]


def collect_attachments(
    out_dir: Path, day: Date, block: str | None = None, with_daily: bool = True
) -> list[tuple[str, bytes]]:
    """Return (filename, bytes) for every artifact that exists for ``day``.

    A block package is the card and that block's slate article; the once-a-day
    pieces ride only when ``with_daily`` is set.
    """
    iso = day.isoformat()
    slate = f"PayoffPitch_Slate_{iso}_{block}" if block else f"PayoffPitch_Slate_{iso}"
    candidates = [
        f"mlb_recommendations_{iso}.xlsx",
        f"{slate}.pdf",
        f"{slate}.mp3",
    ]
    if with_daily:
        candidates += [
            f"PayoffPitch_Regression_{iso}.pdf",
            f"PayoffPitch_Mound_{iso}.pdf",
            f"PayoffPitch_Batter_{iso}.pdf",
            f"PayoffPitch_Regression_{iso}.mp3",
            f"regression_radar_{iso}.pdf",
            f"power_screen_{iso}.pdf",
            f"totals_sheet_{iso}.xlsx",
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

    block = _block(argv)
    with_daily = block is None or "--with-daily" in argv
    attachments = collect_attachments(out_dir, day, block, with_daily)
    if block and not any(n.startswith("PayoffPitch_Slate_") for n, _ in attachments):
        print(f"no {block} slate article for {day}: the pass priced no games; nothing sent")
        return 0
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
    if block:
        b = slate_block(block)
        what = f"the {block} games"
        subject = f"Payoff Pitch — {block} games {day.isoformat()}"
        intro = (
            f"<p>{b.greeting} — here are the Payoff Pitch bets for <b>{what}</b> ({b.starts}) "
            f"on <b>{nice}</b>, priced inside three hours of first pitch off the posted "
            "lineups.</p>"
        )
        text_intro = (
            f"Payoff Pitch {block} games ({b.starts}) for {nice}, priced inside three hours "
            "of first pitch."
        )
    else:
        subject = f"Payoff Pitch — daily package {day.isoformat()}"
        intro = f"<p>Good morning — here's the full Payoff Pitch package for <b>{nice}</b>.</p>"
        text_intro = f"Payoff Pitch daily package for {nice}."
    html_body = (
        intro + "<p>Attached: the bet card (Excel), the slate preview article + audio, and "
        "whichever regression articles and stat cards were written today.</p>"
        "<ul>" + "".join(f"<li>{name}</li>" for name in names) + "</ul>"
    )
    text_body = text_intro + "\n\nAttached:\n" + "\n".join(f"  - {name}" for name in names) + "\n"
    try:
        recipient = send_card_email(
            cfg,
            subject=subject,
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
