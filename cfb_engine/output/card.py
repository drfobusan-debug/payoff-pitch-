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
from html import escape
from pathlib import Path

from cfb_engine.market.ordering import order_buys, order_recs
from cfb_engine.market.tiers import Tier
from cfb_engine.output.brief import GameBrief, TeamBrief
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


def _home_side(recs: list[Recommendation], market: str) -> Recommendation | None:
    rows = [r for r in recs if r.market == market]
    for r in rows:
        if r.team_side == "home" or r.side == "over":
            return r
    return rows[0] if rows else None


def _market_line(recs: list[Recommendation]) -> str:
    """The market's no-vig probabilities beside the model's, one italic line."""
    bits: list[str] = []
    ml = _home_side(recs, "game_ml")
    if ml is not None:
        bits.append(
            f"ML {escape(ml.selection)} market {_pct(ml.fair_prob)} · model {_pct(ml.model_prob)}"
        )
    ats = _home_side(recs, "game_ats")
    if ats is not None:
        bits.append(
            f"ATS {escape(ats.selection)} market {_pct(ats.fair_prob)} · model {_pct(ats.model_prob)}"
        )
    tot = _home_side(recs, "game_total")
    if tot is not None:
        bits.append(
            f"Total {escape(tot.selection)} market {_pct(tot.fair_prob)} · model {_pct(tot.model_prob)}"
        )
    if not bits:
        return ""
    return f"<p class='mkt'><i>{' &nbsp;|&nbsp; '.join(bits)}</i></p>"


def _rank(n: int | None) -> str:
    return "—" if n is None else f"#{n}"


def _rating(x: float | None, rank: int | None) -> str:
    if x is None:
        return "—"
    return f"{x:+.1f} <span class='rk'>{_rank(rank)}</span>"


def _team_row(t: TeamBrief, *, ats: bool) -> str:
    name = escape(t.name)
    if t.poll_rank is not None:
        name = f"<span class='ap'>No. {t.poll_rank}</span> {name}"
    form = t.record
    if t.streak:
        form += f" <span class='streak'>{t.streak}</span>"
    other = " / ".join(
        s for s in (
            f"TR {_rank(t.tr_rank)}" if t.tr_rank is not None else "",
            f"FPI {_rank(t.fpi_rank)}" if t.fpi_rank is not None else "",
        ) if s
    ) or "—"
    sp = _rank(t.sp_rank) if t.rated else "<span class='muted'>unrated</span>"
    if t.rated and t.sp_rating is not None:
        sp += f" <span class='rk'>({t.sp_rating:+.1f})</span>"
    ats_cell = f"<td>{escape(t.ats or '—')}</td>" if ats else ""
    return (
        f"<tr><td class='tn'>{name}</td><td>{form}</td><td>{sp}</td>"
        f"<td>{_rating(t.off_rating, t.off_rank)}</td><td>{_rating(t.def_rating, t.def_rank)}</td>"
        f"<td>{other}</td>{ats_cell}</tr>"
    )


def _team_table(b: GameBrief) -> str:
    ats = bool(b.away.ats or b.home.ats)
    return (
        "<table class='teams'><thead><tr><th></th><th>Record</th><th>SP+</th>"
        f"<th>Offense</th><th>Defense</th><th>Other ranks</th>{'<th>ATS</th>' if ats else ''}</tr></thead><tbody>"
        f"{_team_row(b.away, ats=ats)}{_team_row(b.home, ats=ats)}</tbody></table>"
    )


def _players_line(b: GameBrief) -> str:
    parts = []
    for t in (b.away, b.home):
        names = t.leaders or t.key_players
        if names:
            parts.append(f"<b>{escape(t.name)}</b>: {escape('; '.join(names[:3]))}")
    if not parts:
        return ""
    src = "ESPN leaders" if (b.away.leaders or b.home.leaders) else "season PPA, garbage time excluded"
    return f"<p class='ctx'><b>Who matters</b> <span class='muted'>({src})</span> — {' · '.join(parts)}</p>"


def _out_line(b: GameBrief) -> str:
    parts = [f"<b>{escape(t.name)}</b>: {escape(', '.join(t.out))}" for t in (b.away, b.home) if t.out]
    if not parts:
        return ""
    return f"<p class='ctx'><b>Out</b> <span class='muted'>(reported, not scored)</span> — {' · '.join(parts)}</p>"


def _venue_line(b: GameBrief) -> str:
    bits: list[str] = []
    where = escape(b.venue) if b.venue else None
    if where and b.city:
        where += f", {escape(b.city)}"
    if where:
        if b.grass is True:
            where += " (grass)"
        elif b.grass is False:
            where += " (turf)"
        bits.append(where)
    if b.neutral_site:
        bits.append("neutral site — no home-field charge")
    elif b.hfa_pts is not None:
        src = "VSiN venue table" if b.hfa_listed else "flat league value"
        bits.append(f"home edge priced at <b>{b.hfa_pts:.1f} pts</b> ({src})")
    if b.dome:
        bits.append("indoors")
    else:
        wx = []
        if b.temperature_f is not None:
            wx.append(f"{b.temperature_f:.0f}°F")
        if b.wind_mph is not None:
            wx.append(f"wind {b.wind_mph:.0f} mph")
        elif b.gust_mph is not None:
            wx.append(f"gusts to {b.gust_mph:.0f} mph")
        if b.precipitation is not None and b.precipitation > 0:
            wx.append(f"{b.precipitation:.2f}\" rain expected")
        elif b.precip_pct is not None:
            wx.append(f"{b.precip_pct:.0f}% rain chance")
        if wx:
            bits.append("kickoff forecast " + ", ".join(wx))
    if b.rest_home is not None and b.rest_away is not None and b.rest_home != b.rest_away:
        bits.append(f"rest {b.rest_home}d home vs {b.rest_away}d away")
    if b.conference_game:
        bits.append("conference game")
    if not bits:
        return ""
    return f"<p class='ctx'><b>Venue &amp; weather</b> — {'; '.join(bits)}.</p>"


def _story_line(b: GameBrief) -> str:
    if not b.headline and not b.story and not b.tags:
        return ""
    chips = "".join(f"<span class='chip'>{escape(t)}</span>" for t in b.tags)
    text = escape(b.headline or "")
    if b.story:
        text = f"<b>{text}</b> {escape(b.story)}" if text else escape(b.story)
    return f"<p class='story'>{chips}{text} <span class='muted'>— ESPN/AP preview</span></p>"


def _sharp_line(recs: list[Recommendation]) -> str:
    rows = [r for r in recs if r.sharp_div is not None and r.market in ("game_ats", "game_ml")]
    if not rows:
        return ""
    r = max(rows, key=lambda x: abs(x.sharp_div or 0.0))
    div = r.sharp_div or 0.0
    if abs(div) < 5:
        return ""
    lean = "money is heavier than tickets" if div > 0 else "tickets are heavier than money"
    return (
        f"<p class='ctx'><b>VSiN splits</b> — on {escape(r.selection)} the {lean} "
        f"({div:+.0f} pts handle minus tickets); <span class='muted'>recorded, priced only on the moneyline</span>.</p>"
    )


def _take(b: GameBrief, recs: list[Recommendation]) -> str:
    """The casual read, sentence by sentence, each one only if the data is there."""
    r = recs[0]
    margin = r.exp_margin or 0.0
    fav, dog = (b.home, b.away) if margin >= 0 else (b.away, b.home)
    parts: list[str] = []

    def tag(t: TeamBrief) -> str:
        s = f"{t.name} ({t.record}"
        if t.streak and t.wins + t.losses > 0:
            s += f", {t.streak}"
        if t.rated:
            s += f", SP+ {_rank(t.sp_rank)}"
        return s + ")"

    where = f"in {b.city.split(',')[0]}" if b.city else "on the road"
    if b.neutral_site:
        where = "at a neutral site"
    parts.append(f"{tag(b.away)} visits {tag(b.home)} {where}.")
    if b.away.last or b.home.last:
        lasts = [f"{t.name} {t.last}" for t in (b.away, b.home) if t.last]
        parts.append("Last time out: " + "; ".join(lasts) + ".")
    if fav.off_rank is not None and dog.def_rank is not None:
        parts.append(
            f"The matchup to watch is {fav.name}'s {_ordinal(fav.off_rank)}-ranked offense "
            f"against a {dog.name} defense SP+ has {_ordinal(dog.def_rank)}"
            + (" — a mismatch on paper." if fav.off_rank + 30 < dog.def_rank else ".")
        )
    elif not fav.rated or not dog.rated:
        unr = [t.name for t in (fav, dog) if not t.rated]
        parts.append(
            f"SP+ does not rate {' or '.join(unr)} (FCS), so the model leans on the market's number "
            "more than usual here."
        )
    parts.append(
        f"Our sim has {fav.name} by {abs(margin):.1f} with about {r.exp_total or 0:.0f} total points"
        + (f"; ESPN's FPI gives the home side {b.fpi_home:.0f}%." if b.fpi_home is not None else ".")
    )
    ml = _home_side(recs, "game_ml")
    if ml is not None and ml.fair_prob is not None:
        gap = (ml.model_prob - ml.fair_prob) * 100
        if gap >= 3:
            parts.append(f"We like {ml.selection} more than the market does ({gap:+.1f} pts).")
        elif gap <= -3:
            parts.append(f"The market is higher on {ml.selection} than we are ({gap:+.1f} pts), so no moneyline play.")
        else:
            parts.append("Model and market are within a few points on the moneyline.")
    return f"<p class='take'>{escape(' '.join(parts))}</p>"


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _game_section(recs: list[Recommendation]) -> str:
    matchup, headline, desc = _game_shape(recs)
    b = recs[0].brief
    context = ""
    if b is not None:
        context = (
            f"{_team_table(b)}{_take(b, recs)}{_story_line(b)}{_players_line(b)}"
            f"{_out_line(b)}{_venue_line(b)}{_sharp_line(recs)}"
        )
    return (
        f"<div class='game'><h2>{escape(matchup)}</h2>"
        f"{_market_line(recs)}"
        f"<div class='shape'><span class='tag'>{headline}</span> {desc}</div>"
        f"{context}"
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
.mkt{margin:-2px 0 4px;font-size:9.4pt;color:#4b5563;}
table.teams{width:100%;border-collapse:collapse;font-size:8.9pt;font-family:'DejaVu Sans',sans-serif;margin:6px 0 4px;}
table.teams th{text-align:left;font-weight:normal;color:#6b7280;border-bottom:1px solid #d7dbe0;padding:2px 6px;font-size:7.8pt;text-transform:uppercase;letter-spacing:.5px;}
table.teams td{padding:3px 6px;border-bottom:1px solid #eceef1;vertical-align:top;}
table.teams td.tn{font-weight:bold;color:#16324f;}
.rk{color:#6b7280;font-size:7.8pt;}.ap{color:#c8102e;font-weight:bold;}.streak{color:#16324f;font-weight:bold;}.muted{color:#6b7280;font-style:italic;font-size:8.6pt;}
.take{font-size:10.3pt;margin:6px 0;}
.ctx{font-size:9.2pt;margin:3px 0;color:#2b2f36;}
.story{font-size:9.4pt;margin:6px 0;background:#fbf7ec;border-left:3px solid #c8a951;padding:4px 8px;}
.chip{display:inline-block;background:#c8102e;color:#fff;padding:0 7px;border-radius:9px;font-size:7.6pt;font-family:'DejaVu Sans',sans-serif;margin-right:4px;text-transform:uppercase;letter-spacing:.4px;}
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


def _spoken_context(b: GameBrief) -> str:
    def say(t: TeamBrief) -> str:
        s = t.name
        if t.poll_rank is not None:
            s = f"number {t.poll_rank} {s}"
        if t.wins + t.losses > 0:
            s += f", {t.wins} and {t.losses}"
            if t.streak and int(t.streak[1:]) >= 2:
                s += f", {'won' if t.streak[0] == 'W' else 'lost'} {t.streak[1:]} straight"
        if t.rated:
            s += f", SP plus number {t.sp_rank}"
        return s

    out = f"{say(b.away)} at {say(b.home)}. "
    if b.tags:
        out += "Storyline: " + ", ".join(b.tags) + ". "
    outs = [f"{t.name} without {', '.join(o.split(' ', 1)[-1] for o in t.out[:2])}" for t in (b.away, b.home) if t.out]
    if outs:
        out += "Injuries: " + "; ".join(outs) + ". "
    if not b.dome and b.precipitation and b.precipitation > 0:
        out += "Rain in the forecast. "
    elif not b.dome and b.wind_mph is not None and b.wind_mph >= 15:
        out += f"Windy, about {b.wind_mph:.0f} miles an hour. "
    return out


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
        parts.append(f"{matchup}. ")
        brief = r.brief
        if brief is not None:
            parts.append(_spoken_context(brief))
        parts.append(
            f"The model likes a {headline.lower()} game, about "
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
