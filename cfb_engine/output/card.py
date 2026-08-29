"""The reader-facing slate article (HTML/PDF) and its audio narration (MP3).

Built entirely from the persisted :class:`~cfb_engine.recommendations.Recommendation`
list -- it never re-runs the simulation. For each game it tells the projection
story (expected margin and total, the favorite, the market's number vs the
model's) and lists the buys in bold; a slate-wide "best bets" block gathers
every Strong/Moderate play strongest-first.

This is a model preview, not betting advice.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from pathlib import Path

from cfb_engine.market.ordering import order_buys, order_recs
from cfb_engine.market.tiers import Tier
from cfb_engine.output.render import to_mp3, to_pdf
from cfb_engine.recommendations import Recommendation

logger = logging.getLogger(__name__)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _odds(x: float | None) -> str:
    if x is None:
        return ""
    o = round(x)
    return f"+{o}" if o > 0 else str(o)


def _by_game(recs: list[Recommendation]) -> dict[str, list[Recommendation]]:
    out: dict[str, list[Recommendation]] = {}
    for r in recs:
        out.setdefault(r.game_id, []).append(r)
    return out


def _game_shape(recs: list[Recommendation]) -> tuple[str, str, str]:
    """(matchup, headline, description) for a game's recommendation group."""
    r = recs[0]
    matchup = r.matchup
    margin = r.exp_margin or 0.0
    total = r.exp_total or 0.0
    fav = r.home_abbrev if margin >= 0 else r.away_abbrev
    if total >= 58:
        env = "Shootout"
    elif total <= 45:
        env = "Rock fight"
    else:
        env = "Average-scoring"
    if abs(margin) >= 17:
        shape = "blowout-leaning"
    elif abs(margin) <= 4:
        shape = "coin-flip"
    else:
        shape = "one-score"
    headline = f"{env}, {shape}"
    desc = (
        f"Model projects ~{total:.0f} total points and a {abs(margin):.1f}-point "
        f"lean to {fav}."
    )
    return matchup, headline, desc


def _ml_line(recs: list[Recommendation]) -> str:
    mls = [r for r in recs if r.market == "game_ml"]
    if not mls:
        return ""
    fav = min(mls, key=lambda r: r.market_american if r.market_american is not None else 1e9)
    return (
        f"<div class='ml'>Moneyline: <b>{fav.selection}</b> — market implies "
        f"<b>{_pct(fav.fair_prob)}</b>, model says <b>{_pct(fav.model_prob)}</b>, "
        f"edge <span class='{'pos' if (fav.edge or 0) >= 0 else 'neg'}'>{_pct(fav.edge)}</span>.</div>"
    )


def _best_bets(recs: list[Recommendation]) -> list[Recommendation]:
    return order_buys(recs)


def _game_best_block(recs: list[Recommendation]) -> str:
    buys = _best_bets(recs)
    if not buys:
        return "<p class='bets'><b>Best bets:</b> none clear the buy threshold — model passes.</p>"
    items = "".join(
        f"<li><b>{b.selection} ({_odds(b.market_american)})</b> — {b.display_category}, "
        f"model {b.model_prob * 100:.0f}%, edge {(b.edge or 0.0) * 100:+.1f}% · <i>{b.tier.value}</i></li>"
        for b in buys
    )
    return f"<p class='bets'><b>Best bets</b></p><ul class='bets'>{items}</ul>"


def _game_section(recs: list[Recommendation]) -> str:
    matchup, headline, desc = _game_shape(recs)
    return (
        f"<div class='game'><h2>{matchup}</h2>"
        f"<div class='shape'><span class='tag'>{headline}</span> {desc}</div>"
        f"{_ml_line(recs)}"
        f"{_game_best_block(recs)}</div>"
    )


def _slate_best_block(recs: list[Recommendation]) -> str:
    buys = _best_bets(recs)
    if not buys:
        return (
            "<div class='slatebets'><h2>Slate best bets</h2>"
            "<p>The model passes the entire board today.</p></div>"
        )
    items = "".join(
        f"<li><b>{b.selection} ({_odds(b.market_american)})</b> — {b.display_category} "
        f"({b.matchup}), model {b.model_prob * 100:.0f}%, edge {(b.edge or 0.0) * 100:+.1f}% "
        f"· <i>{b.tier.value}</i></li>"
        for b in buys
    )
    return (
        "<div class='slatebets'><h2>Slate best bets</h2>"
        f"<p class='sbnote'>{len(buys)} plays clear the buy threshold, strongest first:</p>"
        f"<ul class='bets big'>{items}</ul></div>"
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
p{margin:6px 0;}
.lead{font-size:11pt;}
.game{page-break-inside:avoid;border-bottom:2px solid #eceef1;padding-bottom:10px;margin-bottom:6px;}
.shape{background:#eef2f6;border-left:4px solid #16324f;padding:6px 10px;margin:8px 0;font-size:9.8pt;}
.shape .tag{display:inline-block;background:#16324f;color:#fff;padding:1px 8px;border-radius:10px;font-size:8.4pt;font-family:'DejaVu Sans',sans-serif;margin-right:6px;}
.ml{font-size:10pt;margin:6px 0;}
.pos{color:#2e7d32;font-weight:bold;}.neg{color:#b23b3b;font-weight:bold;}
p.bets{margin:10px 0 2px;font-size:11pt;color:#16324f;}
ul.bets{margin:2px 0 4px 0;font-size:10pt;}
ul.bets b{color:#111;}
.slatebets{page-break-inside:avoid;background:#0f2438;color:#f4f6f8;border-radius:6px;padding:12px 16px;margin:22px 0 8px;}
.slatebets h2{color:#ffd76a;border:none;margin:0 0 4px;}
.slatebets .sbnote{color:#c6ccd4;font-style:italic;font-size:9.4pt;margin:0 0 6px;}
ul.bets.big{font-size:10.5pt;}ul.bets.big b{color:#fff;}.slatebets i{color:#ffd76a;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""


def build_article(day: Date, recs: list[Recommendation]) -> tuple[str, str]:
    """Return ``(html, narration_text)`` for the slate."""
    groups = _by_game(recs)
    ordered_games = sorted(groups.values(), key=lambda g: g[0].matchup)
    nice = day.strftime("%A, %B %-d, %Y")
    n_bets = len(_best_bets(recs))
    masthead = (
        "<div class='masthead'>"
        "<div class='brand'><span class='pp'>Payoff</span> Pitch · Gridiron Slate</div>"
        "<h1>Today's Board</h1>"
        "<p class='sub'>Power ratings vs. the market — moneyline, spread, and totals.</p>"
        f"<div class='dateline'>Slate previewed · {nice}</div></div>"
    )
    lead = (
        f"Good morning — here's the {len(ordered_games)}-game college football board for "
        f"{nice.split(',')[0]}. For every matchup we project the expected margin and total "
        "from team power ratings, set the model's number next to the market's, and read the "
        f"edge across moneyline, spread, and total. The engine flagged <b>{n_bets}</b> best "
        "bets — in bold under each game and gathered at the bottom. Model preview, not betting advice."
    )
    body = "".join(_game_section(g) for g in ordered_games)
    body += _slate_best_block(recs)
    fine = (
        "<p class='fine'>Methodology: expected margin and total come from CFBD SP+ (and PFF, "
        "when supplied) adjusted offense/defense, blended toward the market and run through a "
        "Monte Carlo score simulation. Implied probability is the devig-free conversion of the "
        "best posted price; edge is model minus market. Model preview, not investment advice.</p>"
    )
    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{masthead}<p class='lead'>{lead}</p>{body}{fine}</body></html>"
    )
    return html, _narration(day, ordered_games, recs)


def _narration(day: Date, games: list[list[Recommendation]], recs: list[Recommendation]) -> str:
    nice = day.strftime("%A, %B %-d")
    parts = [
        f"What's up everybody, welcome into the Payoff Pitch Gridiron Slate for {nice}. "
        f"We've got {len(games)} games on the board, so let's run the card. "
    ]
    for group in games:
        matchup, headline, _ = _game_shape(group)
        r = group[0]
        fav = r.home_abbrev if (r.exp_margin or 0) >= 0 else r.away_abbrev
        parts.append(
            f"{matchup}. The model likes a {headline.lower()} game, about "
            f"{r.exp_total or 0:.0f} points, leaning {fav}. "
        )
        buys = _best_bets(group)
        if buys:
            b = buys[0]
            parts.append(
                f"Best bet here: {b.selection} at {_odds(b.market_american)}, "
                f"{b.display_category}, model's got it at {b.model_prob * 100:.0f} percent. "
            )
        else:
            parts.append("No bet here, the model passes. ")
    strong = order_recs([r for r in recs if r.tier == Tier.STRONG])
    if strong:
        parts.append("Alright, the headline plays of the day. ")
        for b in strong[:5]:
            parts.append(f"{b.selection} at {_odds(b.market_american)}, {b.display_category}. ")
    parts.append(
        "That's the board. Bet the edges, skip the coin-flips, and we'll grade it all next week. "
        "Payoff Pitch, out."
    )
    return "".join(parts)


def generate_daily_card(
    recs: list[Recommendation],
    slate_date: Date,
    cfg,
    *,
    email: bool,
    to: str | None,
    extra_attachments: list[tuple[str, bytes]] | None = None,
) -> dict[str, Path | None]:
    """Build the article PDF + MP3 and optionally email them with any extras."""
    out: dict[str, Path | None] = {"pdf": None, "mp3": None, "html": None}
    if not recs:
        logger.warning("no recommendations for %s; skipping card", slate_date)
        return out

    html, narr = build_article(slate_date, recs)
    iso = slate_date.isoformat()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = cfg.output_dir / f"cfb_slate_{iso}.html"
    html_path.write_text(html)
    out["html"] = html_path

    attachments: list[tuple[str, bytes]] = list(extra_attachments or [])
    try:
        pdf_bytes = to_pdf(html)
        pdf_path = cfg.output_dir / f"PayoffPitch_CFB_Slate_{iso}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        out["pdf"] = pdf_path
        attachments.insert(0, (pdf_path.name, pdf_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slate article PDF not written: %s", exc)

    try:
        mp3_path = cfg.output_dir / f"PayoffPitch_CFB_Slate_{iso}.mp3"
        mp3_bytes = to_mp3(narr, mp3_path)
        out["mp3"] = mp3_path
        attachments.append((mp3_path.name, mp3_bytes))
    except Exception as exc:  # noqa: BLE001
        logger.warning("slate article MP3 not written: %s", exc)

    print("CFB slate article -> " + ", ".join(str(p) for p in (out["pdf"], out["mp3"]) if p))

    if email and attachments:
        from cfb_engine.output.email import EmailNotConfigured, send_card_email

        body_html = (
            "<div style='font-family:Georgia,serif;color:#1a1a1a;max-width:640px'>"
            "<h2 style='color:#16324f'>Payoff Pitch — CFB Slate</h2>"
            f"<p>Your college football card for <b>{slate_date.strftime('%A, %B %-d, %Y')}</b> "
            "is attached:</p>"
            "<ul><li><b>Excel bet sheet</b> — every priced moneyline, spread, and total.</li>"
            "<li><b>Slate article (PDF)</b> — per-game projection, edge, and best bets.</li>"
            "<li><b>Audio narration (MP3)</b> — the same read, sportscaster style.</li></ul>"
            "<p style='color:#6b7280;font-size:13px'>Model preview, not investment advice.</p></div>"
        )
        try:
            recipient = send_card_email(
                cfg,
                subject=f"Payoff Pitch — CFB Slate ({iso})",
                html_body=body_html,
                text_body="Your Payoff Pitch college-football slate (Excel + PDF + audio) is attached.",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed CFB slate ({len(attachments)} attachments) to {recipient}")
        except EmailNotConfigured as exc:
            print(f"CFB slate email not sent: {exc}")

    return out
