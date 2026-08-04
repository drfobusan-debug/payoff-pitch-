"""Reader-facing audit article (HTML/PDF) + MP3 narration for a graded slate.

Mirrors the daily card: the nightly audit emails the Excel ledger workbook
alongside a short written recap and an audio read of how the model's picks did,
so the audit package matches the prediction package.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from pathlib import Path

from cfb_engine.audit.clv import ClvSummary
from cfb_engine.audit.ledger import OverallMetrics
from cfb_engine.config import Config
from cfb_engine.output.render import to_mp3, to_pdf

logger = logging.getLogger(__name__)

_CSS = """
@page { size: A4; margin: 1.4cm 1.5cm; }
body{font-family:Georgia,serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:8px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
h1{font-size:20pt;color:#16324f;margin:6px 0;}
h2{font-size:14pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:18px 0 6px;}
table{border-collapse:collapse;width:100%;font-size:9.4pt;margin:6px 0;}
th,td{border:1px solid #d7dbe0;padding:3px 6px;text-align:right;}
th:first-child,td:first-child{text-align:left;}
th{background:#eef2f6;color:#16324f;}
.pos{color:#2e7d32;font-weight:bold;}.neg{color:#b23b3b;font-weight:bold;}
.fine{font-size:7.8pt;color:#9aa0a8;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;}
"""


def _cls(v: float) -> str:
    return "pos" if v >= 0 else "neg"


def _metric_table(rows: list[OverallMetrics]) -> str:
    head = (
        "<tr><th>Segment</th><th>N</th><th>Win%</th><th>ROI</th><th>Units</th>"
        "<th>PPV</th></tr>"
    )
    body = "".join(
        f"<tr><td>{m.tier}</td><td>{m.n}</td><td>{m.win_pct * 100:.1f}%</td>"
        f"<td class='{_cls(m.roi)}'>{m.roi * 100:+.1f}%</td>"
        f"<td class='{_cls(m.units)}'>{m.units:+.1f}</td><td>{m.ppv:.3f}</td></tr>"
        for m in rows
    )
    return f"<table>{head}{body}</table>"


def _clv_table(rows: list[ClvSummary]) -> str:
    if not rows:
        return "<p>No closing snapshot captured for this slate.</p>"
    head = "<tr><th>Market</th><th>N</th><th>Mean CLV</th><th>Beat close</th></tr>"
    body = "".join(
        f"<tr><td>{c.label}</td><td>{c.n}</td>"
        f"<td class='{_cls(c.mean_clv)}'>{c.mean_clv * 100:+.2f}</td>"
        f"<td>{c.beat_close_pct * 100:.0f}%</td></tr>"
        for c in rows
    )
    return f"<table>{head}{body}</table>"


def build_audit_article(
    audit_date: Date,
    overall: list[OverallMetrics],
    clv_rows: list[ClvSummary],
    n_graded: int,
) -> tuple[str, str]:
    """Return ``(html, narration_text)`` for the graded slate."""
    nice = audit_date.strftime("%A, %B %-d, %Y")
    buy = next((m for m in overall if m.tier == "Buy (S+M)"), None)
    masthead = (
        "<div class='masthead'><div class='brand'>Payoff Pitch · Gridiron Audit</div>"
        f"<h1>Slate Report — {nice}</h1></div>"
    )
    if buy and buy.n:
        lead = (
            f"Graded <b>{n_graded}</b> markets. The engine's buys went "
            f"<b>{buy.wins}-{buy.losses}</b> "
            f"(<span class='{_cls(buy.roi)}'>{buy.roi * 100:+.1f}% ROI</span>, "
            f"{buy.units:+.1f} units)."
        )
    else:
        lead = f"Graded <b>{n_graded}</b> markets; no buys cleared the threshold on this slate."
    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>"
        f"{masthead}<p>{lead}</p>"
        f"<h2>By segment</h2>{_metric_table(overall)}"
        f"<h2>Closing line value</h2>{_clv_table(clv_rows)}"
        "<p class='fine'>Cumulative through this slate. Model audit, not investment advice.</p>"
        "</body></html>"
    )
    return html, _narration(nice, overall, n_graded)


def _narration(nice: str, overall: list[OverallMetrics], n_graded: int) -> str:
    buy = next((m for m in overall if m.tier == "Buy (S+M)"), None)
    parts = [f"Payoff Pitch Gridiron audit for {nice}. We graded {n_graded} markets. "]
    if buy and buy.n:
        verb = "up" if buy.units >= 0 else "down"
        parts.append(
            f"The model's buys went {buy.wins} and {buy.losses}, "
            f"{verb} {abs(buy.units):.1f} units, an ROI of {buy.roi * 100:.0f} percent. "
        )
    else:
        parts.append("No plays cleared the buy threshold on this slate. ")
    parts.append("Full breakdown by market and closing line value is in the attached ledger. ")
    parts.append("That's the audit. Payoff Pitch, out.")
    return "".join(parts)


def generate_audit_report(
    audit_date: Date,
    overall: list[OverallMetrics],
    clv_rows: list[ClvSummary],
    n_graded: int,
    cfg: Config,
    *,
    email: bool,
    to: str | None,
    extra_attachments: list[tuple[str, bytes]] | None = None,
) -> dict[str, Path | None]:
    """Write the audit article PDF + MP3 and optionally email with the ledger."""
    out: dict[str, Path | None] = {"pdf": None, "mp3": None, "html": None}
    html, narr = build_audit_article(audit_date, overall, clv_rows, n_graded)
    iso = audit_date.isoformat()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.output_dir / f"cfb_audit_{iso}.html"
    html_path.write_text(html)
    out["html"] = html_path

    attachments: list[tuple[str, bytes]] = list(extra_attachments or [])
    try:
        pdf_bytes = to_pdf(html)
        pdf_path = cfg.output_dir / f"PayoffPitch_CFB_Audit_{iso}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        out["pdf"] = pdf_path
        attachments.insert(0, (pdf_path.name, pdf_bytes))
    except Exception as exc:  # noqa: BLE001 - PDF is best-effort
        logger.warning("audit article PDF not written: %s", exc)

    try:
        mp3_path = cfg.output_dir / f"PayoffPitch_CFB_Audit_{iso}.mp3"
        mp3_bytes = to_mp3(narr, mp3_path)
        out["mp3"] = mp3_path
        attachments.append((mp3_path.name, mp3_bytes))
    except Exception as exc:  # noqa: BLE001 - audio is best-effort
        logger.warning("audit article MP3 not written: %s", exc)

    if email and attachments:
        from cfb_engine.output.email import EmailNotConfigured, send_card_email

        body_html = (
            "<div style='font-family:Georgia,serif;color:#1a1a1a;max-width:640px'>"
            "<h2 style='color:#16324f'>Payoff Pitch — CFB Audit</h2>"
            f"<p>Your college football audit for <b>{audit_date.strftime('%A, %B %-d, %Y')}</b> "
            "is attached: the ledger workbook, a written recap (PDF), and an audio read.</p></div>"
        )
        try:
            recipient = send_card_email(
                cfg,
                subject=f"Payoff Pitch — CFB Audit ({iso})",
                html_body=body_html,
                text_body="Your Payoff Pitch college-football audit (ledger + PDF + audio) is attached.",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed CFB audit ({len(attachments)} attachments) to {recipient}")
        except EmailNotConfigured as exc:
            print(f"CFB audit email not sent: {exc}")
    return out
