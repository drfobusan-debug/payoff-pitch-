"""Weekly audit article (HTML/PDF): the past week, then the ledger to date.

Both halves carry the same tables and the same plain-language summary, so a
week can always be read against the whole history it belongs to.
"""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path

from cfb_engine.audit.weekly import Accuracy, ScopeAudit, Segment, WeeklyAudit
from cfb_engine.config import Config
from cfb_engine.output.audit_report import _CSS
from cfb_engine.output.render import to_pdf

logger = logging.getLogger(__name__)

_EXTRA_CSS = """
ul.find li{margin:2px 0;}
h3{font-size:11.5pt;color:#16324f;margin:12px 0 4px;}
.lead{font-size:11pt;}
"""


def _cls(v: float) -> str:
    return "pos" if v >= 0 else "neg"


def _seg_table(rows: list[Segment], *, clv: bool = True) -> str:
    rows = [s for s in rows if s.n or s.pushes]
    if not rows:
        return "<p>No graded bets.</p>"
    head = (
        "<tr><th>Segment</th><th>Bets</th><th>W-L-P</th><th>Win%</th><th>Break-even</th>"
        "<th>Units</th><th>ROI</th><th>95% range</th>"
        + ("<th>Beat close</th>" if clv else "")
        + "<th>Verdict</th></tr>"
    )
    body = ""
    for s in rows:
        beat = f"<td>{s.beat_close:.0%}</td>" if s.clv_n else "<td>-</td>"
        body += (
            f"<tr><td>{escape(s.label)}</td><td>{s.n + s.pushes}</td><td>{s.record}</td>"
            f"<td>{s.win_pct * 100:.1f}%</td><td>{s.breakeven * 100:.1f}%</td>"
            f"<td class='{_cls(s.units)}'>{s.units:+.1f}</td>"
            f"<td class='{_cls(s.roi)}'>{s.roi * 100:+.1f}%</td>"
            f"<td>{s.roi_lo * 100:+.0f}% to {s.roi_hi * 100:+.0f}%</td>"
            + (beat if clv else "")
            + f"<td>{escape(s.verdict)}</td></tr>"
        )
    return f"<table>{head}{body}</table>"


def _clv_table(b: Segment) -> str:
    if not b.clv_n:
        return "<p>No closing prices captured for these buys.</p>"
    return (
        "<table><tr><th>Buys with a close</th><th>Beat close</th><th>Mean CLV (prob pts)</th>"
        "<th>EV at the closing price</th></tr>"
        f"<tr><td>{b.clv_n}</td><td>{b.beat_close:.0%}</td>"
        f"<td class='{_cls(b.clv_mean)}'>{b.clv_mean * 100:+.2f}</td>"
        f"<td class='{_cls(b.close_ev)}'>{b.close_ev * 100:+.1f}%</td></tr></table>"
    )


def _accuracy_table(a: Accuracy) -> str:
    if not a.n and not a.n_close:
        return "<p>No graded rows with market probabilities.</p>"
    rows = ""
    if a.n:
        rows += (
            f"<tr><td>Model</td><td>{a.n}</td><td>{a.model:.4f}</td></tr>"
            f"<tr><td>Market at bet time (no-vig)</td><td>{a.n}</td><td>{a.market:.4f}</td></tr>"
        )
    if a.n_close:
        rows += (
            f"<tr><td>Model, rows with a close</td><td>{a.n_close}</td>"
            f"<td>{a.model_at_close:.4f}</td></tr>"
            f"<tr><td>Closing line (no-vig)</td><td>{a.n_close}</td><td>{a.close:.4f}</td></tr>"
        )
    return (
        "<table><tr><th>Probability</th><th>Graded rows</th><th>Brier (lower is better)</th></tr>"
        f"{rows}</table>"
    )


def _scope_html(scope: ScopeAudit, period_title: str) -> str:
    finds = "".join(f"<li>{escape(f)}</li>" for f in scope.findings)
    slates = f"{len(scope.slates)} slate{'s' if len(scope.slates) != 1 else ''}"
    return (
        f"<h2>{escape(scope.title)}</h2>"
        f"<p class='lead'>{slates} graded.</p>"
        f"<h3>Summary</h3><ul class='find'>{finds}</ul>"
        f"<h3>Buys</h3>{_seg_table([scope.buys, *scope.by_tier])}"
        f"<h3>By market</h3>{_seg_table(scope.by_market)}"
        f"<h3>Sides, spreads and prices</h3>{_seg_table(scope.splits)}"
        f"<h3>By model edge (EV)</h3>{_seg_table(scope.ev_bands)}"
        f"<h3>Closing line value</h3>{_clv_table(scope.buys)}"
        f"{_seg_table(scope.clv_split, clv=False)}"
        f"<h3>Accuracy: model vs market</h3>{_accuracy_table(scope.accuracy)}"
        f"<h3>{period_title}</h3>{_seg_table(scope.periods)}"
        f"<h3>Pass rows (graded, not bet)</h3>{_seg_table([scope.passes], clv=False)}"
    )


def _lead(report: WeeklyAudit) -> str:
    w, t = report.week.buys, report.ledger.buys
    week = (
        f"This week's buys went <b>{w.record}</b> "
        f"(<span class='{_cls(w.units)}'>{w.units:+.1f}u</span>)"
        if w.n or w.pushes
        else "No buys were graded this week"
    )
    return (
        f"<p class='lead'>{week}. Ledger to date: <b>{t.record}</b>, "
        f"<span class='{_cls(t.units)}'>{t.units:+.1f}u</span>, "
        f"ROI <span class='{_cls(t.roi)}'>{t.roi * 100:+.1f}%</span> on {t.n + t.pushes} buys.</p>"
    )


def build_weekly_article(report: WeeklyAudit) -> str:
    title = f"Weekly Audit: {report.start:%b} {report.start.day} - {report.end:%b} {report.end.day}, {report.end.year}"
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}{_EXTRA_CSS}</style>"
        "</head><body>"
        "<div class='masthead'><div class='brand'>Payoff Pitch · Gridiron Audit</div>"
        f"<h1>{title}</h1></div>"
        f"{_lead(report)}"
        f"{_scope_html(report.week, 'By slate')}"
        f"{_scope_html(report.ledger, 'By week')}"
        "<p class='fine'>Break-even = the win rate the prices taken demanded. 95% range = "
        "ROI plus or minus 1.96 standard errors (never narrower than a coin flip at those "
        "prices), and nothing under 30 bets is called beyond noise; 'needs ~N' is how many bets this ROI would "
        "need for that range to exclude zero. Brier scores use every graded side, bet or "
        "not. EV at the closing price judges the price taken against the closing no-vig "
        "probability. Model audit, not investment advice.</p>"
        "</body></html>"
    )


def generate_weekly_report(
    report: WeeklyAudit, cfg: Config, *, email: bool, to: str | None
) -> dict[str, Path | None]:
    """Write the weekly article HTML + PDF and optionally email the PDF."""
    out: dict[str, Path | None] = {"html": None, "pdf": None}
    html = build_weekly_article(report)
    iso = report.end.isoformat()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.output_dir / f"cfb_weekly_audit_{iso}.html"
    html_path.write_text(html)
    out["html"] = html_path
    attachments: list[tuple[str, bytes]] = []
    try:
        pdf_bytes = to_pdf(html)
        pdf_path = cfg.output_dir / f"PayoffPitch_CFB_Weekly_Audit_{iso}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        out["pdf"] = pdf_path
        attachments.append((pdf_path.name, pdf_bytes))
    except Exception as exc:  # noqa: BLE001 - PDF is best-effort
        logger.warning("weekly audit PDF not written: %s", exc)

    if email and attachments:
        from cfb_engine.output.email import EmailNotConfigured, send_card_email

        finds = "".join(f"<li>{escape(f)}</li>" for f in report.ledger.findings)
        body = (
            "<div style='font-family:Georgia,serif;color:#1a1a1a;max-width:640px'>"
            "<h2 style='color:#16324f'>Payoff Pitch — CFB Weekly Audit</h2>"
            f"{_lead(report)}<p><b>Ledger to date</b></p><ul>{finds}</ul>"
            "<p>The full article (this week and the ledger to date) is attached.</p></div>"
        )
        try:
            recipient = send_card_email(
                cfg,
                subject=f"Payoff Pitch — CFB Weekly Audit ({report.start.isoformat()} to {iso})",
                html_body=body,
                text_body="Your Payoff Pitch college-football weekly audit is attached.",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed CFB weekly audit to {recipient}")
        except EmailNotConfigured as exc:
            print(f"CFB weekly audit email not sent: {exc}")
    return out
