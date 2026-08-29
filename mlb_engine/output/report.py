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

from mlb_engine.audit.analysis import (
    BREAKEVEN,
    PriceBucket,
    RunLineMissMatrix,
    dog_vs_favorite,
    false_negative_insights,
    lineup_findings,
    price_bucket_findings,
    price_buckets,
    run_line_miss_findings,
    run_line_miss_matrix,
)
from mlb_engine.audit.ledger import (
    LedgerEntry,
    OverallMetrics,
    engine_metrics,
    market_metrics,
    one_side_per_prop,
    overall_metrics,
)
from mlb_engine.audit.probation import (
    Probation,
    candidate_probation,
    market_probation,
    screen_probation,
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
    rl_matrix: RunLineMissMatrix
    rl_findings: list[str]
    price_rows: list[PriceBucket]
    price_sides: list[PriceBucket]
    price_findings: list[str]
    price_n_dates: int
    lineup_findings: list[str]
    probation: list[Probation]


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
    entries: list[LedgerEntry],
    *,
    period_label: str,
    subtitle: str,
    history: list[LedgerEntry] | None = None,
) -> ReportData:
    """Build one period's report, pricing the buckets off ``history`` when given.

    A single slate carries a few dozen buys spread over six price bands, far too
    few to read a band off. The price section therefore measures the whole
    ledger even inside the daily report, and labels the sample it used.
    """
    n_dates = len({e.date for e in entries})
    # Both sides of a prop are in the ledger; a measurement wants one row per
    # wager, or the near-certain complement grades itself correct for free.
    entries = one_side_per_prop(entries)
    priced = one_side_per_prop(history) if history is not None else entries
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

    # run-line miss matrix: where backed run-line picks lose (one-run-win vs
    # blowout). Rendered in its own report section.
    rl_matrix = run_line_miss_matrix(entries)
    rl_findings = run_line_miss_findings(entries)

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
        rl_matrix=rl_matrix,
        rl_findings=rl_findings,
        price_rows=price_buckets(priced),
        price_sides=dog_vs_favorite(priced),
        price_findings=price_bucket_findings(priced),
        price_n_dates=len({e.date for e in priced if e.odds is not None}),
        # Whether the card saw the lineup that batted. Reads the history for the
        # same reason the price bands do: one slate carries nowhere near the rows
        # the comparison needs.
        lineup_findings=lineup_findings(priced),
        # Probation is a standing judgement on the whole book, so it reads the
        # history rather than the day: a market cannot be condemned or cleared
        # by one slate, which is the entire point of it.
        probation=[
            *market_probation(priced),
            *screen_probation(priced),
            *candidate_probation(priced),
        ],
    )


def _priced_roi(m: OverallMetrics) -> str:
    """The priced return, or the count of rows it would have to be read off."""
    if not m.priced_n:
        return "— (no priced rows)"
    return f"{m.priced_roi * 100:+.1f}% (n={m.priced_n})"


def _verdict(roi: float) -> str:
    return "profitable" if roi > 0 else "roughly break-even" if roi > -0.02 else "in the red"


def _summary_paragraph(d: ReportData) -> str:
    """The headline, quoted on the money that was actually available.

    The ROI over every favored row pays anything the board never priced at an
    assumed -110, and those rows both outnumber and out-win the priced ones, so
    the blended figure has read as profitable through stretches in which every
    real price lost. The verdict is taken from the priced rows and the blended
    one is named as the assumption it is.
    """
    eng = d.engine
    strengths = ", ".join(d.play[:3]) if d.play else "its highest-probability picks"
    leaks = ", ".join(d.fade[:3]) if d.fade else "a few thin-edge markets"
    assumed = eng.n - eng.priced_n
    priced = (
        f"On the {eng.priced_n} of those that carried a real book price the return is "
        f"**{eng.priced_roi * 100:+.1f}%** — {_verdict(eng.priced_roi)}, and the only "
        f"figure that describes money. The other {assumed} were graded at an assumed "
        f"-110 nobody offered"
        if eng.priced_n
        else "None of those rows carried a real book price, so there is no return to report"
    )
    return (
        f"Across every graded market, the side the model favored won "
        f"**{_pct(eng.ppv)}** of the time. {priced}. Its sharpest work is in "
        f"{strengths}. The losses concentrate in {leaks}, and the pattern below is "
        f"consistent: the engine handicaps direction better than it prices it, and "
        f"the leaks are a too-loose bet-selection filter and a slightly cold "
        f"offensive model."
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
    L.append(
        "**ROI (priced)** counts only the rows that carried a real book price. "
        "**ROI (blended)** also pays the rows the board never priced, at an assumed "
        "-110, and is reported for continuity rather than as a return.\n"
    )
    L.append("| Scope | n | PPV (pick win%) | NPV | ROI (priced) | ROI (blended) |")
    L.append("|---|---|---|---|---|---|")
    eng = d.engine
    L.append(
        f"| **Whole engine** (favored side) | {eng.n} | **{_pct(eng.ppv)}** | "
        f"{eng.npv:.2f} | **{_priced_roi(eng)}** | {eng.roi * 100:+.1f}% |"
    )
    for name in ("Strong buy", "Moderate buy"):
        t = _tier_row(d.tiers, name)
        if t is not None:
            L.append(
                f"| {name}s | {t.n} | {_pct(t.win_pct)} | — | {_priced_roi(t)} | "
                f"{t.roi * 100:+.1f}% |"
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

    if d.price_rows:
        L.append("---\n")
        L.append("## Price buckets — is the payout covering the miss?\n")
        L.append(
            f"Every buy that carried a **real book price**, over the "
            f"{d.price_n_dates} slate(s) that have one, grouped by how long the "
            "price was. A dog is *meant* to win under half its bets, so the column "
            "that matters is **Need** — the win rate the price demands — and the "
            "gap to it. A positive gap is a profitable band whatever the raw win "
            "rate says; a negative one is a leak the payout is not covering. Rows "
            "graded at an assumed -110 are excluded, so this table is smaller than "
            "the scorecard above and is the only one whose ROI is real.\n"
        )
        L.append("| Price | n | Win% | Need | Gap | ROI | Units |")
        L.append("|---|---|---|---|---|---|---|")
        for b in [*d.price_sides, *d.price_rows]:
            L.append(
                f"| {b.label} | {b.n} | {_pct(b.win_rate)} | {_pct(b.breakeven)} | "
                f"**{b.shortfall * 100:+.1f} pts** | {b.roi * 100:+.1f}% | "
                f"{b.units:+.2f} |"
            )
        L.append("")
        for f in d.price_findings:
            L.append(f"- {f}")
        L.append("")

    if d.lineup_findings:
        L.append("---\n")
        L.append("## Did the card see the lineup?\n")
        for f in d.lineup_findings:
            L.append(f"- {f}")
        L.append("")

    if d.probation:
        L.append("---\n")
        L.append("## Probation\n")
        L.append(
            "Each market graded on its own buys, each screen on the picks it "
            "refused, and each proposed screen on the buys it *would* refuse. A "
            "verdict changes nothing on its own: acting needs volume, a margin "
            "bigger than one standard error, **and** both halves of the window "
            "agreeing — the last condition being the one that caught every false "
            "finding this engine has shipped.\n"
        )
        L.append("| Subject | Kind | n | ROI | se | Older half | Newer half | Verdict |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in d.probation:
            verdict = f"**{p.status}**" if p.actionable else p.status
            L.append(
                f"| {p.name} | {p.kind} | {p.n} | {p.roi * 100:+.1f}% | "
                f"{p.se * 100:.1f} | {p.first_half * 100:+.1f}% | "
                f"{p.second_half * 100:+.1f}% | {verdict} |"
            )
        L.append("")
        for p in d.probation:
            if p.actionable:
                L.append(f"- {p.finding}")
        L.append("")

    if d.rl_matrix.has_data:
        m = d.rl_matrix
        L.append("---\n")
        L.append("## Run-line miss matrix\n")
        L.append(
            "Where the backed run line actually lands. A **-1.5 favorite** miss is "
            "either a *one-run win* (right team, not enough margin) or an *outright "
            "loss* (wrong team). A **+1.5 dog** miss is either a *2-4 run loss* "
            "(close) or a *5+ blowout* (variance blew it open).\n"
        )
        L.append("| Backed side | n | Covered | One-run win | Outright loss |")
        L.append("|---|---|---|---|---|")
        L.append(
            f"| Favorite -1.5 | {m.fav_n} | {m.fav_cover} | {m.fav_one_run} | "
            f"{m.fav_outright} |"
        )
        L.append("")
        L.append("| Backed side | n | Covered | 2-4 run loss | 5+ blowout |")
        L.append("|---|---|---|---|---|")
        L.append(
            f"| Underdog +1.5 | {m.dog_n} | {m.dog_cover} | {m.dog_moderate} | "
            f"{m.dog_blowout} |"
        )
        L.append("")
        for f in d.rl_findings:
            L.append(f"- {f}")
        L.append("")

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
        "<tr><th>Scope</th><th>n</th><th>PPV (pick win%)</th><th>NPV</th>"
        "<th>ROI (priced)</th><th>ROI (blended)</th></tr>",
        f"<tr><td><strong>Whole engine</strong> (favored side)</td><td>{eng.n}</td>"
        f"<td><strong>{_pct(eng.ppv)}</strong></td><td>{eng.npv:.2f}</td>"
        f"<td><strong>{html.escape(_priced_roi(eng))}</strong></td>"
        f"<td>{eng.roi * 100:+.1f}%</td></tr>",
    ]
    for name in ("Strong buy", "Moderate buy"):
        t = _tier_row(d.tiers, name)
        if t is not None:
            rows_html.append(
                f"<tr><td>{name}s</td><td>{t.n}</td><td>{_pct(t.win_pct)}</td>"
                f"<td>—</td><td>{html.escape(_priced_roi(t))}</td>"
                f"<td>{t.roi * 100:+.1f}%</td></tr>"
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

    if d.price_rows:
        b.append("<h2>Price buckets — is the payout covering the miss?</h2>")
        b.append(
            "<p>Every buy that carried a <strong>real book price</strong>, over the "
            f"{d.price_n_dates} slate(s) that have one, grouped by how long the price "
            "was. A dog is <em>meant</em> to win under half its bets, so the column "
            "that matters is <strong>Need</strong> — the win rate the price demands — "
            "and the gap to it. A positive gap is a profitable band whatever the raw "
            "win rate says; a negative one is a leak the payout is not covering. Rows "
            "graded at an assumed -110 are excluded, so this is the only table here "
            "whose ROI is real.</p>"
        )
        pb = ["<tr><th>Price</th><th>n</th><th>Win%</th><th>Need</th><th>Gap</th>"
              "<th>ROI</th><th>Units</th></tr>"]
        for pr in [*d.price_sides, *d.price_rows]:
            pb.append(
                f"<tr><td>{html.escape(pr.label)}</td><td>{pr.n}</td>"
                f"<td>{_pct(pr.win_rate)}</td><td>{_pct(pr.breakeven)}</td>"
                f"<td><strong>{pr.shortfall * 100:+.1f} pts</strong></td>"
                f"<td>{pr.roi * 100:+.1f}%</td><td>{pr.units:+.2f}</td></tr>"
            )
        b.append("<table>" + "".join(pb) + "</table>")
        if d.price_findings:
            b.append("<ul>")
            for f in d.price_findings:
                b.append(f"<li>{_md_inline_to_html(f)}</li>")
            b.append("</ul>")

    if d.lineup_findings:
        b.append("<h2>Did the card see the lineup?</h2>")
        b.append("<ul>")
        for f in d.lineup_findings:
            b.append(f"<li>{_md_inline_to_html(f)}</li>")
        b.append("</ul>")

    if d.probation:
        b.append("<h2>Probation</h2>")
        b.append(
            "<p>Each market graded on its own buys, each screen on the picks it "
            "refused, and each proposed screen on the buys it <em>would</em> refuse. "
            "A verdict changes nothing on its own: acting needs volume, a margin "
            "bigger than one standard error, <strong>and</strong> both halves of the "
            "window agreeing — the last condition being the one that caught every "
            "false finding this engine has shipped.</p>"
        )
        prb = [
            "<tr><th>Subject</th><th>Kind</th><th>n</th><th>ROI</th><th>se</th>"
            "<th>Older half</th><th>Newer half</th><th>Verdict</th></tr>"
        ]
        for p in d.probation:
            verdict = f"<strong>{p.status}</strong>" if p.actionable else p.status
            prb.append(
                f"<tr><td>{html.escape(p.name)}</td><td>{p.kind}</td><td>{p.n}</td>"
                f"<td>{p.roi * 100:+.1f}%</td><td>{p.se * 100:.1f}</td>"
                f"<td>{p.first_half * 100:+.1f}%</td>"
                f"<td>{p.second_half * 100:+.1f}%</td><td>{verdict}</td></tr>"
            )
        b.append("<table>" + "".join(prb) + "</table>")
        actionable = [p for p in d.probation if p.actionable]
        if actionable:
            b.append("<ul>")
            for p in actionable:
                b.append(f"<li>{_md_inline_to_html(p.finding)}</li>")
            b.append("</ul>")

    if d.rl_matrix.has_data:
        m = d.rl_matrix
        b.append("<h2>Run-line miss matrix</h2>")
        b.append(
            "<p>Where the backed run line actually lands. A <strong>-1.5 favorite</strong> "
            "miss is either a <em>one-run win</em> (right team, not enough margin) or an "
            "<em>outright loss</em> (wrong team). A <strong>+1.5 dog</strong> miss is "
            "either a <em>2-4 run loss</em> (close) or a <em>5+ blowout</em> "
            "(variance blew it open).</p>"
        )
        b.append(
            "<table>"
            "<tr><th>Backed side</th><th>n</th><th>Covered</th>"
            "<th>One-run win</th><th>Outright loss</th></tr>"
            f"<tr><td>Favorite -1.5</td><td>{m.fav_n}</td><td>{m.fav_cover}</td>"
            f"<td>{m.fav_one_run}</td><td>{m.fav_outright}</td></tr>"
            "</table>"
        )
        b.append(
            "<table>"
            "<tr><th>Backed side</th><th>n</th><th>Covered</th>"
            "<th>2-4 run loss</th><th>5+ blowout</th></tr>"
            f"<tr><td>Underdog +1.5</td><td>{m.dog_n}</td><td>{m.dog_cover}</td>"
            f"<td>{m.dog_moderate}</td><td>{m.dog_blowout}</td></tr>"
            "</table>"
        )
        if d.rl_findings:
            b.append("<ul>")
            for f in d.rl_findings:
                b.append(f"<li>{_md_inline_to_html(f)}</li>")
            b.append("</ul>")

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
