"""Draw a ledger-book icon and emit PNG / .ico / .icns (no external assets).

The icon is a classic bound accounting ledger: a dark-green cover with a
darker spine, a white page ruled with horizontal rows and debit/credit column
rules, and a green/red money band — matching the workbook's green spectrum.

Usage:
    python scripts/make_ledger_icon.py [out_dir]

Writes ``ledger.png`` (1024px), ``ledger.ico`` (multi-size) and, where the
platform supports it, ``ledger.icns`` into ``out_dir`` (default: repo
``assets/``). Everything is drawn with Pillow so no binary art is committed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Palette (matches the Strong-buy green spectrum used in the workbook).
COVER = (27, 94, 32)  # #1B5E20 dark green
SPINE = (13, 51, 17)  # #0D3311 darker spine
PAGE = (250, 250, 245)
RULE = (176, 190, 197)  # light blue-grey ruled lines
COLUMN = (198, 40, 40)  # #C62828 red column rule (classic ledger)
MONEY = (102, 187, 106)  # #66BB6A green money band
SHADOW = (0, 0, 0, 60)


def _draw(size: int) -> Image.Image:
    """Render the ledger icon at ``size`` x ``size`` px (RGBA)."""
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def px(frac: float) -> int:
        return int(round(frac * s))

    radius = px(0.09)

    # Soft drop shadow.
    d.rounded_rectangle(
        [px(0.16), px(0.15), px(0.90), px(0.89)], radius=radius, fill=SHADOW
    )

    # Book cover.
    cover = [px(0.12), px(0.10), px(0.88), px(0.90)]
    d.rounded_rectangle(cover, radius=radius, fill=COVER)

    # Spine down the left edge.
    d.rounded_rectangle(
        [px(0.12), px(0.10), px(0.26), px(0.90)], radius=radius, fill=SPINE
    )
    d.rectangle([px(0.22), px(0.10), px(0.26), px(0.90)], fill=SPINE)

    # White page inset to the right of the spine.
    page = [px(0.30), px(0.16), px(0.82), px(0.84)]
    d.rounded_rectangle(page, radius=px(0.02), fill=PAGE)

    # Green money band across the top of the page (the "$$$" header row).
    d.rectangle([page[0], page[1], page[2], px(0.24)], fill=MONEY)

    # Horizontal ruled rows.
    top, bottom = px(0.28), page[3]
    rows = 6
    step = (bottom - top) / rows
    line_w = max(1, px(0.006))
    for i in range(rows + 1):
        y = int(top + i * step)
        d.line([page[0] + px(0.02), y, page[2] - px(0.02), y], fill=RULE, width=line_w)

    # Two vertical column rules (debit / credit).
    for frac in (0.62, 0.72):
        x = px(frac)
        d.line([x, top, x, bottom], fill=COLUMN, width=line_w)

    return img


def _save_icns(base: Image.Image, out: Path) -> bool:
    """Best-effort .icns write; returns True on success."""
    try:
        base.save(out, format="ICNS")
        return True
    except Exception:
        # macOS fallback: build an iconset and run iconutil.
        import shutil
        import subprocess
        import tempfile

        if shutil.which("iconutil") is None:
            return False
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "ledger.iconset"
            iconset.mkdir()
            for sz in (16, 32, 64, 128, 256, 512):
                base.resize((sz, sz), Image.LANCZOS).save(iconset / f"icon_{sz}x{sz}.png")
                base.resize((sz * 2, sz * 2), Image.LANCZOS).save(
                    iconset / f"icon_{sz}x{sz}@2x.png"
                )
            subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True
            )
        return True


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _draw(1024)
    base.save(out_dir / "ledger.png")

    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base.save(out_dir / "ledger.ico", sizes=sizes)

    icns = out_dir / "ledger.icns"
    if _save_icns(base, icns):
        print(f"icons -> {out_dir}/ledger.png, ledger.ico, {icns.name}")
    else:
        print(f"icons -> {out_dir}/ledger.png, ledger.ico (.icns skipped: unsupported here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
