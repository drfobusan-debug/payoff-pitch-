"""Render the full daily slate preview article (per-game arms-vs-bats, regression,
weather/park, game shape, top HR prop, bold best bets) to a PDF + MP3."""

from __future__ import annotations

import sys
from datetime import date as Date

from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import to_mp3
from mlb_engine.output.daily_preview import build_preview_report
from mlb_engine.preview import load_previews


def merge_pdf(htmls: list[str]):
    from weasyprint import HTML

    docs = [HTML(string=h).render() for h in htmls]
    pages = [pg for d in docs for pg in d.pages]
    return docs[0].copy(pages).write_pdf()


def main() -> None:
    day = Date.fromisoformat(sys.argv[1])
    cfg = load_config()
    previews = load_previews(cfg.audit_dir / f"previews_{day.isoformat()}.json")

    html, narr = build_preview_report(day, previews)

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"PayoffPitch_Slate_{day.isoformat()}.pdf"
    pdf.write_bytes(merge_pdf([html]))
    mp3 = out / f"PayoffPitch_Slate_{day.isoformat()}.mp3"
    to_mp3(narr, mp3)
    print("PDF:", pdf)
    print("MP3:", mp3)


if __name__ == "__main__":
    main()
