"""Render the full daily slate preview article (per-game arms-vs-bats, regression,
weather/park, game shape, top HR prop, bold best bets) to a PDF + MP3.

    python -m scripts.regen_slate 2026-08-24                  # the whole slate
    python -m scripts.regen_slate 2026-08-24 --block evening  # the games a slate
                                                              #   pass just priced

With ``--block`` the article reads the slate pass's own previews (the games it
priced, inside the clock window) and the bets on those games from the card, and
writes ``PayoffPitch_Slate_<day>_<block>.pdf``; the whole-slate file is left as
it is. A pass that priced no games writes nothing, and removes the block's files
from an earlier run of the same day, so nothing stale is emailed in their place.
"""

from __future__ import annotations

import argparse
from datetime import date as Date

from mlb_engine.cli import late_previews_path
from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import to_mp3
from mlb_engine.output.daily_preview import build_preview_report
from mlb_engine.preview import load_previews
from mlb_engine.recommendations import load_json
from mlb_engine.slate_blocks import BLOCK_NAMES


def merge_pdf(htmls: list[str]):
    from weasyprint import HTML

    docs = [HTML(string=h).render() for h in htmls]
    pages = [pg for d in docs for pg in d.pages]
    return docs[0].copy(pages).write_pdf()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("day", type=Date.fromisoformat)
    p.add_argument(
        "--block", choices=BLOCK_NAMES, help="preview only the games the slate pass priced"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    day: Date = args.day
    cfg = load_config()
    iso = day.isoformat()

    if args.block:
        previews = load_previews(late_previews_path(cfg, day))
        games = {p.game_pk for p in previews}
        pred = cfg.audit_dir / f"predictions_{iso}.json"
        recs = [r for r in load_json(pred) if r.game_pk in games] if pred.exists() else []
        stem = f"PayoffPitch_Slate_{iso}_{args.block}"
    else:
        previews = load_previews(cfg.audit_dir / f"previews_{iso}.json")
        recs = []
        stem = f"PayoffPitch_Slate_{iso}"

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{stem}.pdf"
    mp3 = out / f"{stem}.mp3"
    if args.block and not previews:
        for stale in (pdf, mp3):
            stale.unlink(missing_ok=True)
        print(f"no {args.block} games priced for {iso}; no slate article written")
        return

    html, narr = build_preview_report(day, previews, recs or None, block=args.block)
    pdf.write_bytes(merge_pdf([html]))
    to_mp3(narr, mp3)
    print("PDF:", pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
