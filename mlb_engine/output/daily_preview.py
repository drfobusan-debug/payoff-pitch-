"""Daily slate *preview* report — the reader-facing article that ships with a run.

Where the nightly audit grades the past, this report previews the day ahead. For
every game on the slate it tells the matchup story the pipeline already computed
and persisted as a :class:`~mlb_engine.preview.GamePreview`:

* each starting pitcher's stuff/command line vs. the lineup he faces,
* each bullpen's contact/command line vs. that same lineup,
* who is regressing positively (buy-low) or negatively (due to cool off),
* the shape of game the simulator expects — blowout vs. coin-flip, low- vs.
  high-run,
* park and weather context,
* the moneyline's market-implied probability, the model's probability, and the
  edge between them,
* and the engine's best bets for that game, in bold.

It renders a Morningstar-style HTML/PDF (same house style as the Audit Desk) plus
an energetic, sportscaster-cadence MP3 narration. It never re-runs the simulation
— everything comes from the persisted previews.

This is a model preview, not betting advice.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from pathlib import Path

import numpy as np

from mlb_engine.market.tiers import Tier
from mlb_engine.output.audit_insight import (
    GOLD,
    INK,
    MUTE,
    NAVY,
    RED,
    _fig_b64,
    market_label,
    to_mp3,
    to_pdf,
)
from mlb_engine.preview import GamePreview
from mlb_engine.recommendations import Recommendation

logger = logging.getLogger(__name__)

_TIER_RANK = {Tier.STRONG.value: 0, Tier.MODERATE.value: 1}


# --- interpretation helpers ------------------------------------------------
def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"


def game_shape(gp: GamePreview) -> tuple[str, str]:
    """Classify the expected game into (headline, one-line description)."""
    if gp.total_mean >= 9.5:
        run_env = "high-scoring"
    elif gp.total_mean <= 7.5:
        run_env = "low-scoring"
    else:
        run_env = "average-run"

    if gp.p_blowout >= 0.34 and gp.p_blowout >= gp.p_close:
        margin = "blowout-leaning"
    elif gp.p_close >= 0.30:
        margin = "coin-flip"
    else:
        margin = "modest-margin"

    fav = gp.fav_team or (gp.home if gp.p_home_win >= 0.5 else gp.away)
    label = f"{run_env.capitalize()}, {margin}"
    desc = (
        f"Sim projects ~{gp.total_mean:.1f} total runs and a {abs(gp.xrd):.1f}-run "
        f"lean toward {fav}. Blowout odds {_pct(gp.p_blowout)}, one-run-or-tie "
        f"{_pct(gp.p_close)}."
    )
    return label, desc


def _edge_cls(edge: float | None) -> str:
    if edge is None:
        return ""
    return "pos" if edge >= 0 else "neg"


# --- charts ----------------------------------------------------------------
def _matchup_chart(gp: GamePreview) -> str:
    """Grouped bars: each offense's lineup xwOBA vs the pitching it faces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        (f"{gp.home} bats — vs {gp.away} SP", gp.home_lineup.xwoba, gp.away_starter.xwoba_allowed),
        (f"{gp.home} bats — vs {gp.away} pen", gp.home_lineup.xwoba, gp.away_pen.xwoba_allowed or 0.0),
        (f"{gp.away} bats — vs {gp.home} SP", gp.away_lineup.xwoba, gp.home_starter.xwoba_allowed),
        (f"{gp.away} bats — vs {gp.home} pen", gp.away_lineup.xwoba, gp.home_pen.xwoba_allowed or 0.0),
    ]
    labels = [r[0] for r in rows]
    bats = [r[1] for r in rows]
    arms = [r[2] for r in rows]
    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(labels) + 1.0))
    ax.barh(y - h / 2, bats, height=h, color=NAVY, zorder=3, label="Lineup xwOBA")
    ax.barh(y + h / 2, arms, height=h, color=GOLD, zorder=3, label="xwOBA allowed")
    for yi, (b, a) in enumerate(zip(bats, arms, strict=False)):
        ax.text(b + 0.004, yi - h / 2, f"{b:.3f}", va="center", fontsize=8, color=INK)
        ax.text(a + 0.004, yi + h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max([*bats, *arms, 0.35]) * 1.18)
    ax.set_xlabel("xwOBA (higher bar = the edge)", fontsize=9, color=MUTE)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Bats vs. the arms they face", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


def _shape_chart(gp: GamePreview) -> str:
    """Horizontal bars for the projected game shape."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{gp.home} win", f"{gp.away} win", "One run or tie", "Blowout (4+)"]
    vals = [gp.p_home_win * 100, (1 - gp.p_home_win) * 100, gp.p_close * 100, gp.p_blowout * 100]
    colors = [NAVY, RED, GOLD, MUTE]
    fig, ax = plt.subplots(figsize=(7.6, 2.2))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    for yi, v in enumerate(vals):
        ax.text(v + 1.0, yi, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Simulated probability (%)", fontsize=9, color=MUTE)
    ax.grid(axis="x", color="#e6e8ec", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Projected game shape", fontsize=11, color=NAVY, loc="left")
    return _fig_b64(fig)


# --- HTML pieces -----------------------------------------------------------
def _starter_row(tag: str, sl) -> str:
    spin = "" if sl.spin is None else f", {sl.spin:.0f} rpm"
    return (
        f"<tr><td class='l'><b>{tag}</b> {sl.name}</td>"
        f"<td>{sl.k_pct * 100:.0f}% (x{sl.xk_pct * 100:.0f})</td>"
        f"<td>{sl.bb_pct * 100:.0f}% (x{sl.xbb_pct * 100:.0f})</td>"
        f"<td>{sl.csw * 100:.0f}%</td><td>{sl.zone_pct * 100:.0f}%</td>"
        f"<td>{sl.xwoba_allowed:.3f}</td><td>{sl.barrel_allowed * 100:.0f}%{spin}</td></tr>"
    )


def _reg_bits(gp: GamePreview) -> str:
    def side(team: str, lu) -> str:
        hot = ", ".join(f"{f.name} (+{f.points:.0f})" for f in lu.hot) or "—"
        cold = ", ".join(f"{f.name} (−{f.points:.0f})" for f in lu.cold) or "—"
        return (
            f"<p><b>{team}.</b> <span class='neg'>Due to cool off:</span> {hot}. "
            f"<span class='pos'>Buy-low / due to heat up:</span> {cold}.</p>"
        )

    return side(gp.home, gp.home_lineup) + side(gp.away, gp.away_lineup)


def top_hr_prop(hr_recs: list[Recommendation]) -> Recommendation | None:
    """The single most likely home-run prop in a game (highest model prob)."""
    priced = [r for r in hr_recs if r.model_prob is not None]
    return max(priced, key=lambda r: r.model_prob) if priced else None


def _hr_line(hr_recs: list[Recommendation]) -> str:
    best = top_hr_prop(hr_recs)
    if best is None:
        return "<p class='hr'><b>Top HR prop:</b> no home-run market priced for this game.</p>"
    odds = "" if best.market_american is None else f" ({best.market_american:+.0f})"
    name = best.selection.replace(" HR o0.5", "").replace(" o0.5", "")
    return (
        f"<p class='hr'><b>Top HR prop:</b> {name}{odds} — model gives him "
        f"<b>{best.model_prob * 100:.1f}%</b> to go yard, the best shot in this game.</p>"
    )


def _best_bets_block(gp: GamePreview) -> str:
    if not gp.best_bets:
        return "<p><b>Best bets:</b> none clear the buy threshold — the model passes this game.</p>"
    items = ""
    for b in gp.best_bets:
        odds = "" if b.odds is None else f" ({b.odds:+.0f})"
        edge = "" if b.edge is None else f", edge {b.edge * 100:+.1f}%"
        items += (
            f"<li><b>{b.selection}{odds}</b> — {market_label(b.market)}, "
            f"model {b.model_prob * 100:.0f}%{edge} · <i>{b.tier}</i></li>"
        )
    return f"<p class='bets'><b>Best bets</b></p><ul class='bets'>{items}</ul>"


def _slate_best_bets_block(
    previews: list[GamePreview], recs: list[Recommendation]
) -> str:
    """Every buy across the slate, strongest first, bold, at the bottom.

    Built from the full ``recs`` (not ``GamePreview.best_bets``, which the
    pipeline truncates to the top four per game) so the count is the true number
    of Strong/Moderate plays and no qualifying bet is silently dropped.
    """
    labels = {gp.game_pk: f"{gp.away}@{gp.home}" for gp in previews}
    rows = [r for r in recs if r.tier in (Tier.STRONG, Tier.MODERATE)]
    rows.sort(key=lambda r: (_TIER_RANK.get(r.tier.value, 9), -(r.edge or 0.0)))
    if not rows:
        return (
            "<div class='slatebets'><h2>Slate best bets</h2>"
            "<p>The model passes the entire board today — no selection clears the buy threshold.</p></div>"
        )
    items = ""
    for r in rows:
        odds = "" if r.market_american is None else f" ({r.market_american:+.0f})"
        edge = "" if r.edge is None else f", edge {r.edge * 100:+.1f}%"
        where = f" ({labels[r.game_pk]})" if r.game_pk in labels else ""
        items += (
            f"<li><b>{r.selection}{odds}</b> — {market_label(r.market)}"
            f"{where}, model {r.model_prob * 100:.0f}%{edge} · <i>{r.tier.value}</i></li>"
        )
    return (
        "<div class='slatebets'><h2>Slate best bets</h2>"
        f"<p class='sbnote'>{len(rows)} plays clear the buy threshold, strongest first:</p>"
        f"<ul class='bets big'>{items}</ul></div>"
    )


def _game_section(gp: GamePreview, hr_recs: list[Recommendation]) -> str:
    shape_label, shape_desc = game_shape(gp)
    env_bits = []
    if gp.park_name:
        pf = "" if gp.park_factor is None else f" (park factor {gp.park_factor:.2f})"
        env_bits.append(f"{gp.park_name}{pf}")
    if gp.roof:
        env_bits.append(f"roof {gp.roof}")
    if gp.wx_summary:
        hr = "" if gp.wx_hr_mult is None else f", HR carry ×{gp.wx_hr_mult:.2f}"
        env_bits.append(f"{gp.wx_summary}{hr}")
    env = " · ".join(env_bits) if env_bits else "no park/weather data"

    ml = (
        f"<div class='ml'>Moneyline: <b>{gp.fav_team}</b> favored — market implies "
        f"<b>{_pct(gp.fav_implied)}</b>, model says "
        f"<b>{_pct(gp.home_ml_prob if gp.fav_side == 'home' else gp.away_ml_prob)}</b>, "
        f"edge <span class='{_edge_cls(gp.fav_edge)}'>{_pct(gp.fav_edge)}</span>.</div>"
    )

    starter_tbl = (
        "<table><tr><th class='l'>Starter</th><th>K% (x)</th><th>BB% (x)</th>"
        "<th>CSW%</th><th>Zone%</th><th>xwOBA</th><th>Barrel% / spin</th></tr>"
        + _starter_row(gp.home, gp.home_starter)
        + _starter_row(gp.away, gp.away_starter)
        + "</table>"
    )

    return (
        f"<div class='game'><h2>{gp.away} @ {gp.home}</h2>"
        f"<p class='env'>{env}</p>"
        f"<div class='shape'><span class='tag'>{shape_label}</span> {shape_desc}</div>"
        f"{ml}"
        f"<h3>Starters vs. the lineups</h3>{starter_tbl}"
        f"<img class='chart' src='data:image/png;base64,{_matchup_chart(gp)}'/>"
        f"<h3>Regression watch</h3>{_reg_bits(gp)}"
        f"<img class='chart' src='data:image/png;base64,{_shape_chart(gp)}'/>"
        f"{_hr_line(hr_recs)}"
        f"{_best_bets_block(gp)}"
        "</div>"
    )


CSS = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.5;font-size:10.5pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:4px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#16324f;}
h1{font-size:22pt;color:#16324f;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-size:15pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:20px 0 6px;}
h3{font-size:11pt;color:#16324f;margin:14px 0 4px;}
p{margin:6px 0;}
.lead{font-size:11pt;}
.game{page-break-inside:avoid;border-bottom:2px solid #eceef1;padding-bottom:10px;margin-bottom:6px;}
.env{color:#6b7280;font-style:italic;font-size:9.4pt;margin:2px 0;}
.shape{background:#eef2f6;border-left:4px solid #16324f;padding:6px 10px;margin:8px 0;font-size:9.8pt;}
.shape .tag{display:inline-block;background:#16324f;color:#fff;padding:1px 8px;border-radius:10px;font-size:8.4pt;font-family:'DejaVu Sans',sans-serif;margin-right:6px;}
.ml{font-size:10pt;margin:6px 0;}
table{border-collapse:collapse;width:100%;font-family:'DejaVu Sans',Helvetica,sans-serif;font-size:8.6pt;margin:8px 0;}
th{background:#16324f;color:#fff;padding:5px 6px;text-align:center;font-weight:600;}
td{border-bottom:1px solid #e6e8ec;padding:5px 6px;text-align:center;}
tr:nth-child(even) td{background:#f5f6f8;}
td.l,th.l{text-align:left;}
.pos{color:#2e7d32;font-weight:bold;}.neg{color:#b23b3b;font-weight:bold;}
img.chart{width:100%;margin:6px 0 2px;}
p.bets{margin:10px 0 2px;font-size:11pt;color:#16324f;}
ul.bets{margin:2px 0 4px 0;font-size:10pt;}
ul.bets b{color:#111;}
.hr{background:#fff8e6;border-left:4px solid #c8a02e;padding:5px 10px;margin:8px 0;font-size:9.8pt;}
.slatebets{page-break-inside:avoid;background:#0f2438;color:#f4f6f8;border-radius:6px;padding:12px 16px;margin:22px 0 8px;}
.slatebets h2{color:#ffd76a;border:none;margin:0 0 4px;}
.slatebets .sbnote{color:#c6ccd4;font-style:italic;font-size:9.4pt;margin:0 0 6px;}
ul.bets.big{font-size:10.5pt;}
ul.bets.big b{color:#fff;}
.slatebets i{color:#ffd76a;}
.callout{background:#eef2f6;border-left:4px solid #16324f;padding:8px 12px;margin:10px 0;font-size:9.6pt;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""


def _hr_by_game(recs: list[Recommendation]) -> dict[int, list[Recommendation]]:
    out: dict[int, list[Recommendation]] = {}
    for r in recs:
        if r.market == "batter_hr":
            out.setdefault(r.game_pk, []).append(r)
    return out


def build_preview_report(
    day: Date, previews: list[GamePreview], recs: list[Recommendation] | None = None
) -> tuple[str, str]:
    hr_map = _hr_by_game(recs or [])
    nice = day.strftime("%A, %B %-d, %Y")
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Slate Preview</div>"
        "<h1>Today's Slate</h1>"
        "<p class='sub'>Arms vs. bats, who's regressing, the shape of the game, and where the edge is.</p>"
        f"<div class='dateline'>Slate previewed · {nice}</div></div>"
    )
    n_bets = sum(len(p.best_bets) for p in previews)
    lead = (
        f"Good morning — here's the {len(previews)}-game board for {nice.split(',')[0]}. For every matchup "
        "we stack each lineup's expected offense against the arms it draws (starter first, then the bullpen it "
        "meets late), flag the hitters the Statcast model says are running hot or cold, and let the simulator "
        "call the shape of the game. Then we put the moneyline's market-implied number next to the model's and "
        f"read the edge. The engine's flagged <b>{n_bets}</b> best bets across the slate — they're in bold under "
        "each game and gathered at the very bottom. Each game also gets its single most likely home-run prop. "
        "This is a model preview, not betting advice."
    )
    body = "".join(_game_section(gp, hr_map.get(gp.game_pk, [])) for gp in previews)
    body += _slate_best_bets_block(previews, recs or [])
    fine = (
        "<p class='fine'>Methodology: probabilities and run distribution come from the engine's Monte Carlo game "
        "simulation and F5 Markov model; xwOBA lines are trailing-window Statcast. Regression flags are the gap "
        "between a hitter's actual and expected wOBA (points). Implied probability is the devig-free American-odds "
        "conversion of the best posted price; edge is model minus implied. Model preview, not investment advice.</p>"
    )
    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{body}{fine}</body></html>"
    )
    narr = _narration(day, previews, hr_map)
    return html, narr


def _narration(
    day: Date, previews: list[GamePreview], hr_map: dict[int, list[Recommendation]] | None = None
) -> str:
    hr_map = hr_map or {}
    nice = day.strftime("%A, %B %-d")
    parts = [
        f"What's up everybody, welcome into the Payoff Pitch Slate Preview for {nice}. "
        f"We got {len(previews)} games on the board, so let's run the card. ",
    ]
    for gp in previews:
        shape_label, _ = game_shape(gp)
        edge = gp.fav_edge
        edge_txt = ""
        if edge is not None and gp.fav_implied is not None:
            side = "value" if edge >= 0 else "no value"
            edge_txt = (
                f" The market implies {gp.fav_implied * 100:.0f} percent, the model says "
                f"{(gp.home_ml_prob if gp.fav_side == 'home' else gp.away_ml_prob) * 100:.0f}, "
                f"so there's {side} on {gp.fav_team}."
            )
        parts.append(
            f"{gp.away} at {gp.home}. The sim likes a {shape_label.lower()} game, "
            f"about {gp.total_mean:.1f} runs, leaning {gp.fav_team}.{edge_txt} "
        )
        if gp.best_bets:
            b = gp.best_bets[0]
            odds = "" if b.odds is None else f" at {b.odds:+.0f}"
            parts.append(
                f"Best bet here: {b.selection}{odds}, {market_label(b.market)}, "
                f"model's got it at {b.model_prob * 100:.0f} percent. "
            )
        else:
            parts.append("No bet here, the model passes. ")
        hr_best = top_hr_prop(hr_map.get(gp.game_pk, []))
        if hr_best is not None:
            name = hr_best.selection.replace(" HR o0.5", "").replace(" o0.5", "")
            parts.append(
                f"If you want a longball, {name} is the top home-run shot here at "
                f"{hr_best.model_prob * 100:.0f} percent. "
            )
    strong = [
        (gp, b)
        for gp in previews
        for b in gp.best_bets
        if _TIER_RANK.get(b.tier, 9) == 0
    ]
    strong.sort(key=lambda t: -(t[1].edge or 0.0))
    if strong:
        parts.append("Alright, the headline plays of the day. ")
        for _gp, b in strong[:5]:
            odds = "" if b.odds is None else f" at {b.odds:+.0f}"
            parts.append(f"{b.selection}{odds}, {market_label(b.market)}. ")
    parts.append(
        "That's the slate. Bet the edges, skip the coin-flips, and we'll grade it all tomorrow. "
        "Payoff Pitch, out."
    )
    return "".join(parts)


# --- top-level entry point -------------------------------------------------
def generate_daily_preview(
    previews: list[GamePreview],
    slate_date: Date,
    cfg,
    *,
    email: bool,
    to: str | None,
    recs: list[Recommendation] | None = None,
    extra_attachments: list[tuple[str, bytes]] | None = None,
) -> dict[str, Path | None]:
    """Build the preview PDF + MP3 and optionally email them with the ledger."""
    out: dict[str, Path | None] = {"pdf": None, "mp3": None, "html": None}
    if not previews:
        logger.warning("no previews for %s; skipping slate preview report", slate_date)
        return out

    html, narr = build_preview_report(slate_date, previews, recs)
    iso = slate_date.isoformat()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.output_dir / f"slate_preview_{iso}.html"
    html_path.write_text(html)
    out["html"] = html_path

    attachments: list[tuple[str, bytes]] = list(extra_attachments or [])
    try:
        pdf_bytes = to_pdf(html)
        pdf_path = cfg.output_dir / f"PayoffPitch_Slate_{iso}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        out["pdf"] = pdf_path
        attachments.insert(0, (pdf_path.name, pdf_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slate preview PDF not written: %s", exc)

    try:
        mp3_path = cfg.output_dir / f"PayoffPitch_Slate_{iso}.mp3"
        mp3_bytes = to_mp3(narr, mp3_path)
        out["mp3"] = mp3_path
        attachments.append((mp3_path.name, mp3_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slate preview MP3 not written: %s", exc)

    print("Slate preview -> " + ", ".join(str(p) for p in (out["pdf"], out["mp3"]) if p))

    if email and attachments:
        from mlb_engine.output.email import EmailNotConfigured, send_card_email

        body_html = (
            "<div style='font-family:Georgia,serif;color:#1a1a1a;max-width:640px'>"
            "<h2 style='color:#16324f'>Payoff Pitch — Slate Preview</h2>"
            f"<p>Your preview for <b>{slate_date.strftime('%A, %B %-d, %Y')}</b> is attached:</p>"
            "<ul><li><b>Slate article (PDF)</b> — per-game arms-vs-bats, regression watch, game shape, "
            "weather, moneyline edge, and the best bets in bold.</li>"
            "<li><b>Audio narration (MP3)</b> — the same read, sportscaster style.</li>"
            "<li><b>Excel bet sheet</b> — every priced market (when attached).</li></ul>"
            "<p style='color:#6b7280;font-size:13px'>Model preview, not investment advice.</p></div>"
        )
        try:
            recipient = send_card_email(
                cfg,
                subject=f"Payoff Pitch — Slate Preview ({iso})",
                html_body=body_html,
                text_body="Your Payoff Pitch slate preview (PDF + audio) is attached.",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed slate preview ({len(attachments)} attachments) to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Slate preview email not sent: {exc}")

    return out
