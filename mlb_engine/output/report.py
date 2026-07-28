"""Audit report generator — renders the graded ledger into a formatted article.

The report mirrors the layout the engine's owner settled on:

1. Executive summary (data-driven prose)
2. Core metrics (whole-engine + tier rows)
3. Market scorecard (per-market PPV / NPV / ROI / min-p-to-play / verdict)
4. Most common errors (detected from the ledger)
5. Recommendations (each mapped to a goal: PPV / NPV / FP / FN)
6. What to play and fade right now

The same renderer produces the **daily** report (one graded slate) and the
**weekly** report (trailing seven days), differing only in the period label and
the set of ledger rows fed in. Output is Markdown + HTML, with an optional PDF
rendered from the HTML via WeasyPrint.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

from mlb_engine.audit.analysis import BREAKEVEN, false_negative_insights
from mlb_engine.audit.ledger import (
    LedgerEntry,
    OverallMetrics,
    engine_metrics,
    market_metrics,
    overall_metrics,
)

# --- verdict thresholds (documented in the report itself) ------------------
PLAY = "Play"
NEUTRAL = "Neutral"
FADE = "Fade"

PLAY_MIN_PPV = 0.55  # backed-side win rate to call a market playable
FADE_MAX_ROI = -0.15  # ROI at or below this is bleeding -> fade
FADE_MAX_PPV = 0.45  # backed-side win rate this low -> fade
MIN_SAMPLE = 5  # fewer favored picks than this -> not enough to judge

PLAY_FLOOR = "0.58"  # conviction floor for a green market
NEUTRAL_FLOOR = "0.62"  # higher bar before betting a yellow market

_DOT = {PLAY: "🟢", NEUTRAL: "🟡", FADE: "🔴"}

MARKET_LABELS: dict[str, str] = {
    "game_ml": "Game moneyline",
    "game_rl": "Game run line",
    "game_total": "Game total",
    "f5_ml": "First-5 moneyline",
    "f5_rl": "First-5 run line",
    "f5_total": "First-5 total",
    "batter_h": "Batter hits",
    "batter_1b": "Batter singles",
    "batter_2b": "Batter doubles",
    "batter_hr": "Batter home runs",
    "batter_hrr": "Batter H+R+RBI combo",
    "batter_r": "Batter runs",
    "batter_rbi": "Batter RBI",
    "batter_tb": "Batter total bases",
    "pitcher_k": "Pitcher strikeouts",
    "pitcher_outs": "Pitcher outs",
    "pitcher_h": "Pitcher hits allowed",
    "pitcher_bb": "Pitcher walks",
    "pitcher_er": "Pitcher earned runs",
}


def _label(market: str) -> str:
    return MARKET_LABELS.get(market, market)


@dataclass
class MarketRow:
    market: str
    label: str
    n: int
    ppv: float
    npv: float
    roi: float
    verdict: str
    min_p: str
    abstained: bool
    reason: str


def _classify(m: OverallMetrics) -> MarketRow:
    label = _label(m.tier)  # OverallMetrics.tier holds the market name here
    if m.n == 0:
        return MarketRow(
            m.tier, label, 0, m.ppv, m.npv, m.roi, NEUTRAL, "—", True,
            "model correctly abstains — no favored picks",
        )
    if m.n < MIN_SAMPLE:
        return MarketRow(
            m.tier, label, m.n, m.ppv, m.npv, m.roi, NEUTRAL, NEUTRAL_FLOOR, False,
            "thin sample — wait for more data",
        )
    if m.ppv >= PLAY_MIN_PPV and m.roi > 0:
        return MarketRow(
            m.tier, label, m.n, m.ppv, m.npv, m.roi, PLAY, PLAY_FLOOR, False,
            "backed side wins above breakeven and turns a profit",
        )
    if m.roi <= FADE_MAX_ROI or m.ppv < FADE_MAX_PPV:
        return MarketRow(
            m.tier, label, m.n, m.ppv, m.npv, m.roi, FADE, "avoid", False,
            "losing money at these edges",
        )
    return MarketRow(
        m.tier, label, m.n, m.ppv, m.npv, m.roi, NEUTRAL, NEUTRAL_FLOOR, False,
        "no proven edge — coin-flip",
    )


@dataclass
class ReportData:
    period_label: str
    subtitle: str
    n_dates: int
    engine: OverallMetrics
    tiers: list[OverallMetrics]
    rows: list[MarketRow]
    errors: list[str]
    recommendations: list[str]  # each: "text || goal-tag"
    play: list[str]
    neutral: list[str]
    fade: list[str]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _tier_row(tiers: list[OverallMetrics], name: str) -> OverallMetrics | None:
    for t in tiers:
        if t.tier == name:
            return t
    return None


def _under_bias(entries: list[LedgerEntry]) -> tuple[bool, float, int]:
    """Detect a systematic Under lean: favored total picks that are Unders + lost."""
    unders = [
        e for e in entries
        if e.market in ("game_total", "f5_total")
        and e.model_prob >= 0.5
        and "under" in e.selection.lower()
        and e.result != "push"
    ]
    if not unders:
        return (False, 0.0, 0)
    wins = sum(1 for e in unders if e.result == "win")
    wr = wins / len(unders)
    return (wr < 0.5 and len(unders) >= 3, wr, len(unders))


def _outs_faded_deep(entries: list[LedgerEntry]) -> tuple[bool, float, int]:
    faded = [
        e for e in entries
        if e.market == "pitcher_outs" and e.model_prob < 0.5 and e.result != "push"
    ]
    if len(faded) < 3:
        return (False, 0.0, 0)
    wr = sum(1 for e in faded if e.result == "win") / len(faded)
    return (wr > 0.5, wr, len(faded))


def build_report_data(
    entries: list[LedgerEntry], *, period_label: str, subtitle: str
) -> ReportData:
    n_dates = len({e.date for e in entries})
    engine = engine_metrics(entries)
    tiers = overall_metrics(entries)
    rows = [_classify(m) for m in market_metrics(entries)]

    play = [r.label for r in rows if r.verdict == PLAY]
    neutral = [r.label for r in rows if r.verdict == NEUTRAL]
    fade = [r.label for r in rows if r.verdict == FADE]

    errors: list[str] = []
    recs: list[str] = []

    # 1 — thin-edge / EV-chasing buys
    buy = _tier_row(tiers, "Buy (S+M)")
    if buy is not None and buy.n > 0 and buy.win_pct < BREAKEVEN:
        errors.append(
            f"**Coin-flips dressed as strong bets.** Strong/Moderate buys won only "
            f"{_pct(buy.win_pct)} (n={buy.n}) — below the {_pct(BREAKEVEN)} breakeven. "
            f"Plus-money prices are promoting near-toss-ups to 'buys' on EV alone."
        )
        recs.append(
            "Add a conviction floor to bet selection — require model probability "
            f"≥ ~{PLAY_FLOOR} *in addition to* positive EV before tagging any play a "
            "Moderate/Strong buy.||✕ eliminate false positives · ↑ PPV"
        )

    # 2 — systematic Under bias
    ub, wr, n = _under_bias(entries)
    if ub:
        errors.append(
            f"**Slate-wide Under bias.** Favored Under totals won just {_pct(wr)} "
            f"(n={n}) — the run/offense model is under-projecting scoring and getting "
            f"burned when games go over."
        )
        recs.append(
            "Recalibrate the offense/run model upward — re-check the run-total mean "
            "and the park + weather scoring multipliers so projected totals aren't low."
            "||✕ eliminate false positives · ↓ reduce false negatives · ↑ PPV & NPV"
        )

    # 3 — under-rated offensive upside (false negatives on props)
    fns = false_negative_insights(entries)
    fn_markets = sorted({_label(i.market) for i in fns})
    if fn_markets:
        shown = ", ".join(fn_markets[:4])
        errors.append(
            f"**Underrating offensive upside.** Faded picks kept winning in {shown} "
            "— the model treats the ceiling of good hitters as lower than reality."
        )
        recs.append(
            "Fatten the offensive upside tail — give quality hitters more weight on "
            "the multi-hit (o1.5) and combo (o2.5) lines where faded picks keep hitting."
            "||↓ reduce false negatives · ↑ NPV & PPV"
        )

    # 4 — starter length sold short
    od, owr, on = _outs_faded_deep(entries)
    if od:
        errors.append(
            f"**Selling starters short.** Faded 'outs' overs still won {_pct(owr)} "
            f"(n={on}) — efficient starters are pitching deeper than the model expects."
        )
        recs.append(
            "Loosen starter-length limits for efficient arms — raise the outs "
            "projection for low-pitch-per-batter starters."
            "||↓ reduce false negatives · ↑ NPV"
        )

    # 5 — fade pockets
    fade_labels = ", ".join(fade)
    if fade:
        errors.append(
            f"**Money-losing pockets.** {fade_labels} are currently below breakeven "
            "and should be sat out until the fixes above ship."
        )
        recs.append(
            f"Gate or drop the red markets ({fade_labels}) — do not bet them until "
            "their PPV recovers."
            "||✕ eliminate false positives · ↑ PPV"
        )

    # 6 — fading moderate market favorites (NPV leak on moneylines)
    ml_faded_won = [
        e for e in entries
        if e.market == "game_ml" and e.model_prob < 0.5 and e.result == "win"
    ]
    if ml_faded_won:
        recs.append(
            "Shrink thin edges toward the market — when the market prices a side "
            "≥ ~.57 but the model is under .50, blend toward the market instead of "
            "fully fading it.||↓ reduce false negatives · ↑ NPV"
        )

    # always: protect what works
    if play:
        recs.append(
            "Leave the green markets alone — " + ", ".join(play) + " are the model's "
            "strengths; keep them and size up where conviction is real."
            "||protects existing PPV"
        )

    return ReportData(
        period_label=period_label,
        subtitle=subtitle,
        n_dates=n_dates,
        engine=engine,
        tiers=tiers,
        rows=rows,
        errors=errors,
        recommendations=recs,
        play=play,
        neutral=neutral,
        fade=fade,
    )


def _summary_paragraph(d: ReportData) -> str:
    eng = d.engine
    strengths = ", ".join(d.play[:3]) if d.play else "its highest-probability picks"
    leaks = ", ".join(d.fade[:3]) if d.fade else "a few thin-edge markets"
    verdict = "profitable" if eng.roi > 0 else "roughly break-even" if eng.roi > -0.02 else "in the red"
    return (
        f"Across every graded market, the side the model favored won "
        f"**{_pct(eng.ppv)}** of the time, for a **{eng.roi * 100:+.1f}% ROI** on the "
        f"whole book — {verdict} overall. Its sharpest work is in {strengths}. "
        f"The losses concentrate in {leaks}, and the pattern below is consistent: "
        f"the engine is a solid handicapper whose leaks are a too-loose bet-selection "
        f"filter and a slightly cold offensive model — both fixable without touching "
        f"what already works."
    )


# --- Markdown ---------------------------------------------------------------
def render_markdown_report(d: ReportData) -> str:
    L: list[str] = []
    L.append("# PayoffPitch Engine — Audit Report")
    L.append(f"\n### {d.period_label} · {d.subtitle}\n")
    L.append("---\n")

    L.append("## Executive summary\n")
    L.append(_summary_paragraph(d) + "\n")

    L.append("---\n")
    L.append("## Core metrics\n")
    L.append("| Scope | n | PPV (pick win%) | NPV | ROI |")
    L.append("|---|---|---|---|---|")
    eng = d.engine
    L.append(
        f"| **Whole engine** (favored side) | {eng.n} | **{_pct(eng.ppv)}** | "
        f"{eng.npv:.2f} | **{eng.roi * 100:+.1f}%** |"
    )
    for name in ("Strong buy", "Moderate buy"):
        t = _tier_row(d.tiers, name)
        if t is not None:
            L.append(
                f"| {name}s | {t.n} | {_pct(t.win_pct)} | — | {t.roi * 100:+.1f}% |"
            )
    L.append("")

    L.append("---\n")
    L.append("## Market scorecard\n")
    L.append(
        "Every graded market rated on PPV / NPV / ROI, sorted highest-to-lowest "
        "return. **🟢 Play** = profitable and above breakeven; **🟡 Neutral** = no "
        "usable edge yet (or the model correctly abstains) — wait for more data; "
        "**🔴 Fade** = losing money, do not bet until fixed. *Min p to Play* is the "
        "minimum model probability a selection must clear before the engine fires a "
        "bet in that market.\n"
    )
    L.append("| Market | PPV | NPV | ROI | Min p to Play | Verdict |")
    L.append("|---|---|---|---|---|---|")
    for r in d.rows:
        ppv = "—" if r.abstained else f"{r.ppv:.2f}"
        roi = "~0.0%" if r.abstained else f"{r.roi * 100:+.1f}%"
        L.append(
            f"| {r.label} | {ppv} | {r.npv:.2f} | {roi} | {r.min_p} | "
            f"{_DOT[r.verdict]} {r.verdict} |"
        )
    L.append("")
    L.append(
        "**Definitions —** **PPV (Positive Predictive Value):** of the sides the "
        "model *backs*, the share that actually win. **NPV (Negative Predictive "
        "Value):** of the sides the model *fades*, the share that actually lose. "
        "*(Diagnostic accuracy terms — NPV here is **not** the financial 'Net "
        "Present Value'.)*\n"
    )

    L.append("---\n")
    L.append("## Most common errors\n")
    if d.errors:
        for e in d.errors:
            L.append(f"- {e}")
    else:
        L.append("- No systematic error pattern cleared the detection thresholds "
                 "this period.")
    L.append("")

    L.append("---\n")
    L.append("## Recommendations\n")
    L.append(
        "*Each action is mapped to the goal it serves:* **↑ PPV · ↑ NPV · "
        "✕ eliminate false positives · ↓ reduce false negatives.**\n"
    )
    for i, rec in enumerate(d.recommendations, 1):
        text, _, goal = rec.partition("||")
        L.append(f"> **{i} — {text.strip()}**")
        L.append(f"> → **{goal.strip()}**\n")

    L.append("### What to play and fade right now\n")
    L.append("Based on this audit, until the fixes above ship *(verdicts match the "
             "scorecard)*:\n")
    L.append(
        f"- **🟢 Play:** {', '.join(d.play) if d.play else '—'} — each only when the "
        f"selection clears the **{PLAY_FLOOR}** conviction floor."
    )
    L.append(
        f"- **🔴 Fade / avoid:** {', '.join(d.fade) if d.fade else '—'} — bleeding "
        "money right now."
    )
    L.append(
        f"- **🟡 Neutral (wait for data):** {', '.join(d.neutral) if d.neutral else '—'} "
        f"— coin-flips or markets the model abstains from; require a higher "
        f"**{NEUTRAL_FLOOR}** bar before betting."
    )
    L.append("")
    L.append(
        f"*Sample note: this report covers {d.n_dates} graded "
        f"{'day' if d.n_dates == 1 else 'days'} — treat per-market figures as "
        "directional; the accumulating ledger firms them up over time.*"
    )
    return "\n".join(L)


# --- HTML -------------------------------------------------------------------
_HTML_STYLE = (
    "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
    "max-width:820px;margin:40px auto;padding:0 24px;color:#1a1f2b;line-height:1.6}"
    "h1{font-size:28px;border-bottom:3px solid #0b5;padding-bottom:8px;margin-bottom:4px}"
    "h3{color:#556;font-weight:500;margin-top:0}"
    "h2{margin-top:36px;color:#0a3;border-bottom:1px solid #e3e6ea;padding-bottom:6px}"
    "table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}"
    "th,td{border:1px solid #e3e6ea;padding:8px 12px;text-align:left}"
    "th{background:#f4f7f9}tr:nth-child(even){background:#fafbfc}"
    "blockquote{background:#f4faf6;border-left:4px solid #0b5;margin:14px 0;"
    "padding:12px 18px;border-radius:0 6px 6px 0}blockquote strong{color:#083}"
    "hr{border:0;border-top:1px solid #e3e6ea;margin:28px 0}li{margin:6px 0}em{color:#667}"
)


def _md_inline_to_html(text: str) -> str:
    """Escape then apply the tiny inline markdown the report uses (**bold**, *em*)."""
    out = html.escape(text)
    while "**" in out:
        out = out.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    while "*" in out:
        out = out.replace("*", "<em>", 1).replace("*", "</em>", 1)
    return out


def render_html_report(d: ReportData) -> str:
    b: list[str] = []
    b.append(f"<h1>PayoffPitch Engine — Audit Report</h1><h3>{html.escape(d.period_label)} · "
             f"{html.escape(d.subtitle)}</h3>")

    b.append("<h2>Executive summary</h2>")
    b.append(f"<p>{_md_inline_to_html(_summary_paragraph(d))}</p>")

    b.append("<h2>Core metrics</h2>")
    eng = d.engine
    rows_html = [
        "<tr><th>Scope</th><th>n</th><th>PPV (pick win%)</th><th>NPV</th><th>ROI</th></tr>",
        f"<tr><td><strong>Whole engine</strong> (favored side)</td><td>{eng.n}</td>"
        f"<td><strong>{_pct(eng.ppv)}</strong></td><td>{eng.npv:.2f}</td>"
        f"<td><strong>{eng.roi * 100:+.1f}%</strong></td></tr>",
    ]
    for name in ("Strong buy", "Moderate buy"):
        t = _tier_row(d.tiers, name)
        if t is not None:
            rows_html.append(
                f"<tr><td>{name}s</td><td>{t.n}</td><td>{_pct(t.win_pct)}</td>"
                f"<td>—</td><td>{t.roi * 100:+.1f}%</td></tr>"
            )
    b.append("<table>" + "".join(rows_html) + "</table>")

    b.append("<h2>Market scorecard</h2>")
    b.append(
        "<p>Every graded market rated on PPV / NPV / ROI, sorted highest-to-lowest "
        "return. 🟢 Play = profitable and above breakeven; 🟡 Neutral = no usable "
        "edge yet (or the model correctly abstains); 🔴 Fade = losing money, do not "
        "bet until fixed. <em>Min p to Play</em> is the minimum model probability a "
        "selection must clear before the engine fires a bet.</p>"
    )
    sc = ["<tr><th>Market</th><th>PPV</th><th>NPV</th><th>ROI</th>"
          "<th>Min p to Play</th><th>Verdict</th></tr>"]
    for r in d.rows:
        ppv = "—" if r.abstained else f"{r.ppv:.2f}"
        roi = "~0.0%" if r.abstained else f"{r.roi * 100:+.1f}%"
        sc.append(
            f"<tr><td>{html.escape(r.label)}</td><td>{ppv}</td><td>{r.npv:.2f}</td>"
            f"<td>{roi}</td><td>{html.escape(r.min_p)}</td>"
            f"<td>{_DOT[r.verdict]} {r.verdict}</td></tr>"
        )
    b.append("<table>" + "".join(sc) + "</table>")
    b.append(
        "<p><strong>Definitions —</strong> <strong>PPV (Positive Predictive Value):"
        "</strong> of the sides the model backs, the share that win. <strong>NPV "
        "(Negative Predictive Value):</strong> of the sides the model fades, the "
        "share that lose. <em>(NPV here is not the financial 'Net Present Value'.)</em></p>"
    )

    b.append("<h2>Most common errors</h2><ul>")
    if d.errors:
        for e in d.errors:
            b.append(f"<li>{_md_inline_to_html(e)}</li>")
    else:
        b.append("<li>No systematic error pattern cleared the detection thresholds "
                 "this period.</li>")
    b.append("</ul>")

    b.append("<h2>Recommendations</h2>")
    b.append("<p><em>Each action is mapped to the goal it serves: ↑ PPV · ↑ NPV · "
             "✕ eliminate false positives · ↓ reduce false negatives.</em></p>")
    for i, rec in enumerate(d.recommendations, 1):
        text, _, goal = rec.partition("||")
        b.append(
            f"<blockquote><strong>{i} — {_md_inline_to_html(text.strip())}</strong>"
            f"<br>→ <strong>{html.escape(goal.strip())}</strong></blockquote>"
        )

    b.append("<h3>What to play and fade right now</h3><ul>")
    b.append(f"<li><strong>🟢 Play:</strong> {html.escape(', '.join(d.play) or '—')} — "
             f"each only above the {PLAY_FLOOR} conviction floor.</li>")
    b.append(f"<li><strong>🔴 Fade / avoid:</strong> {html.escape(', '.join(d.fade) or '—')} "
             "— bleeding money right now.</li>")
    b.append(f"<li><strong>🟡 Neutral (wait for data):</strong> "
             f"{html.escape(', '.join(d.neutral) or '—')} — require a higher "
             f"{NEUTRAL_FLOOR} bar before betting.</li>")
    b.append("</ul>")
    b.append(
        f"<p><em>Sample note: this report covers {d.n_dates} graded "
        f"{'day' if d.n_dates == 1 else 'days'} — treat per-market figures as "
        "directional; the accumulating ledger firms them up over time.</em></p>"
    )

    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_HTML_STYLE}</style>"
        f"</head><body>{''.join(b)}</body></html>"
    )


class PdfNotAvailable(RuntimeError):
    """Raised when a PDF render is requested but WeasyPrint is not installed."""


def render_pdf(html_body: str) -> bytes:
    """Render the report HTML to PDF bytes via WeasyPrint (imported lazily)."""
    try:
        from weasyprint import HTML  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise PdfNotAvailable(
            "PDF export needs WeasyPrint (`pip install weasyprint`)"
        ) from exc
    return bytes(HTML(string=html_body).write_pdf())


# --- period helpers ---------------------------------------------------------
def daily_entries(entries: list[LedgerEntry], day: Date) -> list[LedgerEntry]:
    iso = day.isoformat()
    return [e for e in entries if e.date == iso]


def weekly_entries(entries: list[LedgerEntry], end: Date) -> list[LedgerEntry]:
    start = end - timedelta(days=6)
    return [e for e in entries if start.isoformat() <= e.date <= end.isoformat()]
