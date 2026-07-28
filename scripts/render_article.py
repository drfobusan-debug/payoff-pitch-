"""Render a newsletter markdown file to styled HTML + PDF (same look as the
daily regression newsletters).

Usage:
    python scripts/render_article.py <path-to-article.md>

Writes <article>.html and <article>.pdf next to the source file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1.3rem;color:#1a1a1a;line-height:1.6;font-size:1rem}
h1{font-size:1.9rem;margin-bottom:.1rem;border-bottom:3px solid #0b5;padding-bottom:.3rem}
h3{color:#666;font-weight:600;margin-top:.2rem}
h2{margin-top:1.9rem;border-bottom:1px solid #e2e2e2;padding-bottom:.25rem}
table{border-collapse:collapse;width:100%;font-size:.8rem;margin:.6rem 0}
th,td{border:1px solid #ccc;padding:3px 6px;text-align:center} th{background:#f0f4f8}
td:nth-child(2),th:nth-child(2){text-align:left}
strong{color:#111} hr{border:none;border-top:1px solid #e2e2e2;margin:1.5rem 0}
ul{margin:.4rem 0} li{margin:.35rem 0} em{color:#555}"""


def render(md_path: Path) -> tuple[Path, Path]:
    body = markdown.markdown(md_path.read_text(), extensions=["tables", "smarty"])
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>\n'
        f"{CSS}\n</style></head><body>{body}</body></html>"
    )
    html_path = md_path.with_suffix(".html")
    pdf_path = md_path.with_suffix(".pdf")
    html_path.write_text(html)
    HTML(string=html).write_pdf(str(pdf_path))
    return html_path, pdf_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python scripts/render_article.py <article.md>")
    h, p = render(Path(sys.argv[1]))
    print(f"wrote {h}\nwrote {p}")
