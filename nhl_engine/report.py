"""PDF card: the priced markets plus the context a reader wants next to them --
team form (streak, last 5, last 5 home/away), offensive/defensive ranks, who is
out, and the skaters/goalies who carry each side (outputs plan §2).

Form and ranks come from MoneyPuck game logs already cached for pricing; "out"
is the slate's availability log (RotoWire + roster diffs); key players are the
Phase 3 projections for the skaters the books quote. Nothing here changes a
price -- the PDF reads the same ``SlateCard`` the txt/md/xlsx do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as Date
from html import escape
from pathlib import Path

import pandas as pd

from engine_common.podcasts.extract import Pick as ArticlePick
from engine_common.podcasts.render import bet_line
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.audit.podcast_picks import HOSTS, UNATTRIBUTED, Pick, StatedRecord, Tally, tally
from nhl_engine.data.availability import current_out, read_log
from nhl_engine.data.moneypuck import MoneyPuckClient, as_of
from nhl_engine.data.teamnames import CODES
from nhl_engine.features.podcast_read import PodcastRead, teams_in
from nhl_engine.features.props import GoalieProjection, SkaterProjection
from nhl_engine.market.pricing import is_prop
from nhl_engine.pipeline import GameCard, PropContext, SlateCard

# Offense ranks high-to-low, defense low-to-high (rate against).
RANKED = {
    "xgf60_5v5": ("5v5 xGF/60", False),
    "xga60_5v5": ("5v5 xGA/60", True),
    "pp_xgf60": ("PP xGF/60", False),
    "pk_xga60": ("PK xGA/60", True),
}
FORM_GAMES = 5
KEY_SKATERS = 3


@dataclass(frozen=True)
class TeamForm:
    games: int
    wins: int
    losses: int
    streak: str  # e.g. W3 / L2 / SO1 / --
    last5: str  # W-L (shootout games counted by goal tie as "SO")
    last5_home: str
    last5_away: str
    gf5: float
    ga5: float

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}" if self.games else "0 GP"


def _results(mp: MoneyPuckClient, code: str, slate: Date, *, season: int) -> pd.DataFrame:
    """Current-season games before ``slate`` (all-situation rows), oldest first."""
    df = mp.team_games(code)
    if df.empty or "gameDate" not in df.columns:
        return pd.DataFrame(columns=["gameDate", "home_or_away", "goalsFor", "goalsAgainst"])
    if "situation" in df.columns:
        df = df[df["situation"] == "all"]
    df = as_of(df, slate, season=season)
    return df.sort_values("gameDate", kind="stable")


def _wl(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for gf, ga in zip(df["goalsFor"], df["goalsAgainst"], strict=True):
        out.append("W" if gf > ga else "L" if gf < ga else "SO")
    return out


def _tally(res: list[str]) -> str:
    if not res:
        return "--"
    w, lo, so = res.count("W"), res.count("L"), res.count("SO")
    return f"{w}-{lo}" + (f" ({so} SO)" if so else "")


def _streak(res: list[str]) -> str:
    if not res:
        return "--"
    last = res[-1]
    n = 0
    for r in reversed(res):
        if r != last:
            break
        n += 1
    return f"{last}{n}"


def team_form(mp: MoneyPuckClient, code: str, slate: Date, *, season: int) -> TeamForm:
    df = _results(mp, code, slate, season=season)
    res = _wl(df)
    home = df[df["home_or_away"] == "HOME"]
    away = df[df["home_or_away"] == "AWAY"]
    tail = df.tail(FORM_GAMES)
    return TeamForm(
        games=len(res),
        wins=res.count("W"),
        losses=res.count("L"),
        streak=_streak(res),
        last5=_tally(res[-FORM_GAMES:]),
        last5_home=_tally(_wl(home.tail(FORM_GAMES))),
        last5_away=_tally(_wl(away.tail(FORM_GAMES))),
        gf5=float(tail["goalsFor"].mean()) if len(tail) else 0.0,
        ga5=float(tail["goalsAgainst"].mean()) if len(tail) else 0.0,
    )


def league_ranks(rates: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    """``team -> metric -> rank`` (1 = best) over whatever teams ``rates`` holds."""
    out: dict[str, dict[str, int]] = {t: {} for t in rates}
    for key, (_, ascending) in RANKED.items():
        order = sorted(rates, key=lambda t: rates[t].get(key, 0.0), reverse=not ascending)
        for i, t in enumerate(order, 1):
            out[t][key] = i
    return out


@dataclass
class ReportContext:
    form: dict[str, TeamForm] = field(default_factory=dict)
    ranks: dict[str, dict[str, int]] = field(default_factory=dict)
    out: dict[str, list[str]] = field(default_factory=dict)  # team -> "Name (status, role)"
    skaters: dict[str, list[SkaterProjection]] = field(default_factory=dict)
    goalies: dict[str, GoalieProjection | None] = field(default_factory=dict)
    n_ranked: int = 0
    podcast: PodcastRead | None = None  # external read; never feeds a price or gate
    pod_picks: list[Pick] = field(default_factory=list)  # tonight's logged picks
    pod_history: list[Pick] = field(default_factory=list)  # every logged pick, graded or not
    pod_stated: list[StatedRecord] = field(default_factory=list)  # records read on air
    vsin_picks: list[Pick] = field(default_factory=list)  # tonight's VSiN game bets
    vsin_other: list[ArticlePick] = field(default_factory=list)  # tonight's VSiN props, as written
    vsin_history: list[Pick] = field(default_factory=list)  # every VSiN game bet, graded or not


def _key_skaters(props: PropContext | None, team: str) -> list[SkaterProjection]:
    if props is None:
        return []
    seen: set[int] = set()
    mine: list[SkaterProjection] = []
    for (t, _), p in props.skaters.items():
        if t == team and p.player_id not in seen:
            seen.add(p.player_id)
            mine.append(p)
    mine.sort(key=lambda p: p.mean("pts"), reverse=True)
    return mine[:KEY_SKATERS]


def _goalie(props: PropContext | None, team: str, name: str) -> GoalieProjection | None:
    if props is None:
        return None
    for (t, _), g in props.goalies.items():
        if t == team and g.name == name:
            return g
    return next((g for (t, _), g in props.goalies.items() if t == team), None)


def build_context(
    card: SlateCard,
    *,
    mp: MoneyPuckClient,
    data_dir: Path,
    all_rates: dict[str, dict[str, float]],
) -> ReportContext:
    """``all_rates`` is ``team -> rates`` for every team you want in the ranking
    (pass all 32 for league ranks; the slate's teams alone ranks within the slate)."""
    ctx = ReportContext(ranks=league_ranks(all_rates), n_ranked=len(all_rates))
    log = read_log(data_dir, card.slate_date)
    for g in card.games:
        for ti in (g.away_in, g.home_in):
            team = ti.code
            if team in ctx.form:
                continue
            ctx.form[team] = team_form(mp, team, card.slate_date, season=card.season)
            ctx.out[team] = [
                f"{a.name} ({a.status}{', ' + a.role if a.role else ''})"
                for a in sorted(current_out(log, team).values(), key=lambda a: a.name)
            ]
            ctx.skaters[team] = _key_skaters(card.props, team)
            ctx.goalies[team] = _goalie(card.props, team, ti.starter.name)
    return ctx


# ---------------------------------------------------------------- rendering

CSS = """
@page { size: A4; margin: 1.4cm 1.5cm 1.6cm; }
* { box-sizing: border-box; }
body{font-family:Georgia,'Times New Roman',serif;color:#1a1a1a;line-height:1.45;font-size:10pt;margin:0;}
.masthead{border-bottom:3px solid #16324f;padding-bottom:8px;margin-bottom:4px;}
.brand{font-size:12pt;letter-spacing:2px;color:#c8102e;font-weight:bold;text-transform:uppercase;}
.brand .pp{color:#16324f;}
h1{font-size:22pt;color:#16324f;margin:6px 0 2px;line-height:1.08;}
.sub{color:#6b7280;font-style:italic;font-size:10.5pt;margin:0 0 2px;}
.dateline{font-size:8.5pt;color:#6b7280;letter-spacing:1px;text-transform:uppercase;margin-top:4px;}
h2{font-size:15pt;color:#16324f;border-bottom:1px solid #d7dbe0;padding-bottom:3px;margin:18px 0 6px;}
h3{font-size:12.5pt;color:#16324f;margin:14px 0 2px;}
h3 .kick{float:right;font-family:'DejaVu Sans',sans-serif;font-size:9pt;color:#6b7280;font-weight:normal;padding-top:4px;}
p{margin:5px 0;}
.game{border-bottom:2px solid #eceef1;padding-bottom:10px;margin-bottom:6px;}
.shape{background:#eef2f6;border-left:4px solid #16324f;padding:6px 10px;margin:8px 0;font-size:9.6pt;}
.shape .tag{display:inline-block;background:#16324f;color:#fff;padding:1px 8px;border-radius:10px;font-size:8.4pt;font-family:'DejaVu Sans',sans-serif;margin-right:6px;}
table.teams,table.rows{width:100%;border-collapse:collapse;font-size:8.7pt;font-family:'DejaVu Sans',sans-serif;margin:6px 0 4px;}
table.teams th,table.rows th{text-align:left;font-weight:normal;color:#6b7280;border-bottom:1px solid #d7dbe0;padding:2px 6px;font-size:7.8pt;text-transform:uppercase;letter-spacing:.5px;}
table.teams,table.rows{page-break-inside:avoid;}
table.teams td,table.rows td{padding:3px 6px;border-bottom:1px solid #eceef1;vertical-align:top;}
table.rows td.sel{white-space:nowrap;}
table.rows td.st{font-size:7.6pt;color:#4b5563;}
table.teams td.tn{font-weight:bold;color:#16324f;}
table.rows td.num{text-align:right;font-variant-numeric:tabular-nums;}
.rk{color:#6b7280;font-size:7.8pt;}.streak{color:#16324f;font-weight:bold;}.muted{color:#6b7280;font-style:italic;font-size:8.6pt;}
.ctx{font-size:9.2pt;margin:3px 0;color:#2b2f36;}
.chip{display:inline-block;background:#c8102e;color:#fff;padding:0 7px;border-radius:9px;font-size:7.6pt;font-family:'DejaVu Sans',sans-serif;margin-right:4px;text-transform:uppercase;letter-spacing:.4px;}
.chip.gray{background:#6b7280;}
.pos{color:#2e7d32;font-weight:bold;}.neg{color:#b23b3b;font-weight:bold;}
.slatebets{page-break-inside:avoid;background:#0f2438;color:#f4f6f8;border-radius:6px;padding:12px 16px;margin:16px 0 8px;}
.slatebets h2{color:#ffd76a;border:none;margin:0 0 4px;}
.slatebets .sbnote{color:#c6ccd4;font-style:italic;font-size:9.4pt;margin:0 0 6px;}
.slatebets table{color:#f4f6f8;}
.slatebets td,.slatebets th{border-color:#2b4460 !important;}
.pod{border:1px solid #d7dbe0;border-left:4px solid #b8860b;background:#fbf8ef;padding:6px 10px;margin:8px 0;font-size:9pt;page-break-inside:avoid;}
.pod .ptag{display:inline-block;background:#b8860b;color:#fff;padding:1px 8px;border-radius:10px;font-size:8pt;font-family:'DejaVu Sans',sans-serif;margin-right:6px;}
.pod ul{margin:3px 0 3px 16px;padding:0;}.pod li{margin:1px 0;}
.pod .q{color:#4b5563;font-size:8.3pt;}.pod .q .ts{font-family:'DejaVu Sans',sans-serif;color:#9aa0a8;font-size:7.4pt;margin-right:4px;}
.fine{font-size:7.6pt;color:#9aa0a8;font-family:'DejaVu Sans',sans-serif;border-top:1px solid #e6e8ec;margin-top:16px;padding-top:6px;line-height:1.35;}
"""


def _american(x: float) -> str:
    return f"{x:+.0f}"


def _sel(r: LedgerRow) -> str:
    line = "" if r.line is None else f" {r.line:g}"
    return f"{r.side} {r.entity}{line}".strip()


def _signed(x: float) -> str:
    cls = "pos" if x > 0 else "neg" if x < 0 else ""
    return f"<span class='{cls}'>{x:+.3f}</span>"


def _rank(ctx: ReportContext, team: str, key: str, value: float, fmt: str) -> str:
    r = ctx.ranks.get(team, {}).get(key)
    rk = f" <span class='rk'>#{r}/{ctx.n_ranked}</span>" if r else ""
    return f"{value:{fmt}}{rk}"


def _team_row(g: GameCard, side: str, ctx: ReportContext) -> str:
    ti = g.away_in if side == "away" else g.home_in
    t = ti.code
    f = ctx.form.get(t)
    form = (
        f"{f.record} <span class='streak'>{escape(f.streak)}</span>"
        f"<br><span class='rk'>L5 {escape(f.last5)} · home {escape(f.last5_home)} · away {escape(f.last5_away)}"
        f" · GF/GA {f.gf5:.1f}/{f.ga5:.1f}</span>"
        if f
        else "<span class='muted'>no games</span>"
    )
    r = ti.rates
    return (
        f"<tr><td class='tn'>{escape(t)}</td><td>{form}</td>"
        f"<td>{_rank(ctx, t, 'xgf60_5v5', r.get('xgf60_5v5', 0.0), '.2f')}</td>"
        f"<td>{_rank(ctx, t, 'xga60_5v5', r.get('xga60_5v5', 0.0), '.2f')}</td>"
        f"<td>{_rank(ctx, t, 'pp_xgf60', r.get('pp_xgf60', 0.0), '.1f')}</td>"
        f"<td>{_rank(ctx, t, 'pk_xga60', r.get('pk_xga60', 0.0), '.1f')}</td>"
        f"<td>{escape(ti.starter.name)} <span class='rk'>{escape(ti.starter.status)}, {ti.starter.gsax60:+.2f} GSAx/60</span></td></tr>"
    )


def _team_table(g: GameCard, ctx: ReportContext) -> str:
    return (
        "<table class='teams'><thead><tr><th></th><th>Record · form</th><th>Offense (5v5 xGF/60)</th>"
        "<th>Defense (5v5 xGA/60)</th><th>PP</th><th>PK</th><th>Goalie</th></tr></thead><tbody>"
        f"{_team_row(g, 'away', ctx)}{_team_row(g, 'home', ctx)}</tbody></table>"
    )


def _players_line(g: GameCard, ctx: ReportContext) -> str:
    parts = []
    for t in (g.away_in.code, g.home_in.code):
        names = [
            f"{p.name} ({p.position}, {p.minutes:.0f} min, {p.mean('pts'):.2f} pts/g, {p.mean('sog'):.1f} SOG/g)"
            for p in ctx.skaters.get(t, [])
        ]
        gl = ctx.goalies.get(t)
        if gl is not None:
            shots = f"on {gl.shots} shots" if gl.shots else "prior, no shots yet"
            names.append(f"{gl.name} (G, .{gl.sv_pct * 1000:.0f} SV% {shots})")
        if names:
            parts.append(f"<b>{escape(t)}</b>: {escape('; '.join(names))}")
    if not parts:
        return ""
    return (
        "<p class='ctx'><b>Who matters</b> <span class='muted'>(projected from this and last season's"
        f" game logs; skaters the books quote)</span> — {' · '.join(parts)}</p>"
    )


def _out_line(g: GameCard, ctx: ReportContext) -> str:
    parts = [
        f"<b>{escape(t)}</b>: {escape(', '.join(ctx.out[t]))}"
        for t in (g.away_in.code, g.home_in.code)
        if ctx.out.get(t)
    ]
    if not parts:
        return "<p class='ctx'><b>Out</b> — <span class='muted'>nobody listed in today's availability log</span></p>"
    return (
        "<p class='ctx'><b>Out</b> <span class='muted'>(RotoWire + roster diffs)</span> — "
        f"{' · '.join(parts)}</p>"
    )


def _shape(g: GameCard) -> str:
    s = g.summary
    return (
        "<div class='shape'><span class='tag'>Model</span>"
        f"{escape(g.home_in.code)} wins {s['home_win']:.1%} (reg {s['home_reg_win']:.1%}, OT/SO {s['draw_reg']:.1%})"
        f" · expected {escape(g.away_in.code)} {s['exp_away']:.2f} – {escape(g.home_in.code)} {s['exp_home']:.2f}"
        f" (total {s['exp_total']:.2f}; P1/P2/P3 {s['p1_exp_total']:.2f}/{s['p2_exp_total']:.2f}/{s['p3_exp_total']:.2f})"
        "</div>"
    )


def _pod_block(g: GameCard, ctx: ReportContext) -> str:
    pod = ctx.podcast
    if pod is None or pod.status != "ok":
        return ""
    read = pod.games.get(g.matchup)
    if read is None:
        return (
            "<div class='pod'><span class='ptag'>Pod read</span>"
            "<span class='muted'>game not discussed on the episode</span></div>"
        )
    quoted = " · ".join(
        [f"{escape(t)} {p:+d}" for t, p in read.quoted_ml.items()]
        + ([f"total {read.quoted_total:g}"] if read.quoted_total else [])
    )
    parts = [
        "<div class='pod'><span class='ptag'>Pod read</span>",
        f"<span class='muted'>from {escape(read.anchor)}"
        + (f"; lines quoted on air: {quoted}" if quoted else "")
        + "</span>",
    ]
    picks = [p for p in ctx.pod_picks if p.matchup == g.matchup]
    if picks:
        parts.append(
            "<div class='q'><b>Picks logged</b> "
            "<span class='muted'>(graded tomorrow; name only when heard)</span></div>"
            "<ul>" + "".join(_pick_line(p, g, ctx) for p in picks) + "</ul>"
        )
    else:
        parts.append("<br><span class='muted'>no bet made on air</span>")
    parts.append("</div>")
    return "".join(parts)


def _pick_sel(p: Pick) -> str:
    if p.market == "game_total":
        return f"{p.side.capitalize()} {p.line:g}" if p.line else p.side.capitalize()
    if p.market == "game_pl":
        return f"{p.side} {p.line:+g}"
    return f"{p.side} ML"


def _pick_record(host: str, ctx: ReportContext) -> str:
    """Our audited record for the host; what they claim on air stays in the records block."""
    rec = tally(ctx.pod_history, "host").get(host)
    if rec is not None and rec.n:
        return f"{rec.record()} graded"
    return "no graded picks yet"


_WHO_OR_STAKE = re.compile(
    r"(?:(?:[\d.]+|puck bucks?|units?|pb|"
    + "|".join(sorted({a for names in HOSTS.values() for a in names} | {h.lower() for h in HOSTS}))
    + r")(?:[\s,/&]+|$))+",
    re.I,
)


def _pick_reason(p: Pick, g: GameCard, ctx: ReportContext) -> str:
    """The host's reason for this bet, off the episode summary's "BET — who — reason" lines."""
    read = ctx.podcast.games.get(g.matchup) if ctx.podcast else None
    for line in read.summary if read else []:
        parts = [x.strip() for x in re.split(r"\s+[—–·-]\s+", line) if x.strip()]
        if len(parts) < 2:
            continue
        bet = parts[0].lower()
        if p.market == "game_total":
            if p.side not in bet:
                continue
        elif not (p.side.lower() in bet.split() or teams_in(parts[0], {p.side})):
            continue
        why = parts[-1]
        if not _WHO_OR_STAKE.fullmatch(why):
            return why
    return "no reason given"


def _pick_line(p: Pick, g: GameCard, ctx: ReportContext) -> str:
    name = p.host if p.host != UNATTRIBUTED else "Unattributed"
    reason = _pick_reason(p, g, ctx)
    return bet_line(name, _pick_record(p.host, ctx), _pick_sel(p), p.american, reason)


def _vsin_block(g: GameCard, ctx: ReportContext) -> str:
    picks = [p for p in ctx.vsin_picks if p.matchup == g.matchup]
    if not picks:
        return ""
    ours = tally(ctx.vsin_history, "host")
    lines = "".join(
        bet_line(
            p.host,
            f"{ours[p.host].record()} graded" if p.host in ours else "no graded picks yet",
            _pick_sel(p),
            p.american,
            p.text,
        )
        for p in picks
    )
    return (
        "<div class='pod'><span class='ptag'>VSiN best bets</span>"
        "<span class='muted'>(as written in VSiN's articles; graded tomorrow, not a model input)</span>"
        f"<ul>{lines}</ul></div>"
    )


def _vsin_header(ctx: ReportContext) -> str:
    if not ctx.vsin_picks and not ctx.vsin_other and not ctx.vsin_history:
        return ""
    total = Tally()
    for p in ctx.vsin_history:
        total.add(p)
    cells = " · ".join(
        f"<b>{escape(host)}</b> {escape(t.line())}"
        for host, t in sorted(tally(ctx.vsin_history, "host").items(), key=lambda kv: -kv[1].n)
    )
    other = ""
    if ctx.vsin_other:
        other = (
            "<p class='muted'>VSiN props and parlays tonight (listed as written, not graded): "
            + "; ".join(escape(f"{a.host or 'VSiN'}: {a.description}") for a in ctx.vsin_other[:20])
            + "</p>"
        )
    return (
        "<p class='muted'>VSiN best bets, graded by us against official finals (incl. OT/SO): "
        f"{escape(total.line())}" + (f". {cells}" if cells else "") + ". A record is not "
        "evidence of an edge until 100+ graded bets; they never feed a price, gate or stake.</p>"
        + other
    )


def _pod_records(ctx: ReportContext) -> str:
    ours = tally(ctx.pod_history, "host")
    stated = {r.host: r for r in ctx.pod_stated}
    if not ours and not stated:
        return ""
    total = Tally()
    for p in ctx.pod_history:
        total.add(p)
    cells = []
    for host in sorted(set(ours) | set(stated), key=lambda h: (h == UNATTRIBUTED, h)):
        t, r = ours.get(host), stated.get(host)
        bits = [f"<b>{escape(host)}</b>"]
        if t:
            bits.append(f"our grade {escape(t.line())}")
        if r:
            bits.append(f"self-reported {r.wins}-{r.losses} {r.units:+g} pb")
        cells.append(bits[0] + (": " + "; ".join(bits[1:]) if len(bits) > 1 else ""))
    return (
        f"<p class='muted'>Podcast picks, graded by us against official finals: {escape(total.line())}. "
        + " · ".join(cells)
        + ". Names only where the transcript hands off to a host; self-reported = the puck-bucks record they read on air.</p>"
    )


def _pod_header(ctx: ReportContext) -> str:
    pod = ctx.podcast
    if pod is None:
        return ""
    if pod.status == "ok":
        return (
            f"<p class='muted'>Hockey Gambling Podcast (SGPN): {escape(pod.episode_title)}, "
            f"published {escape(pod.published[:16])} — their bets are listed under each game; "
            "it is not an input to the model.</p>" + _pod_records(ctx)
        )
    return f"<p class='muted'>Hockey Gambling Podcast: {escape(pod.detail or pod.status)}.</p>"


def _gates(r: LedgerRow, keep: int = 3) -> str:
    g = [x for x in r.gates if x != "stale_quote"] or list(r.gates)
    return ", ".join(g[:keep]) + (f" +{len(g) - keep}" if len(g) > keep else "")


def _rows_table(rows: list[LedgerRow], *, limit: int) -> str:
    if not rows:
        return "<p class='muted'>no readable quotes</p>"
    body = "".join(
        f"<tr><td>{escape(r.market)}</td><td class='sel'>{escape(_sel(r))}</td><td class='num'>{_american(r.american)}</td>"
        f"<td>{escape(r.book)}</td><td class='num'>{r.model_prob:.3f}</td><td class='num'>{r.consensus:.3f}</td>"
        f"<td class='num'>{_signed(r.edge)}</td><td class='num'>{_signed(r.ev)}</td>"
        f"<td class='st'>{'<span class=chip>buy</span>' if r.is_buy else escape(_gates(r))}</td></tr>"
        for r in rows[:limit]
    )
    return (
        "<table class='rows'><thead><tr><th>market</th><th>selection</th><th>price</th><th>book</th>"
        "<th>model</th><th>market</th><th>edge</th><th>EV</th><th>status</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _game_block(g: GameCard, ctx: ReportContext) -> str:
    game_rows = sorted((r for r in g.rows if not is_prop(r.market)), key=lambda r: -r.edge)
    buys = [r for r in game_rows if r.is_buy]
    reads = [r for r in game_rows if not r.is_buy and r.edge > 0]
    props = sorted((r for r in g.rows if is_prop(r.market)), key=lambda r: -abs(r.edge))
    parts = [
        "<div class='game'>",
        f"<h3>{escape(g.matchup)}<span class='kick'>{escape(g.away_in.code)} {g.away_in.games} GP · "
        f"{escape(g.home_in.code)} {g.home_in.games} GP · lineup {escape(g.away_in.lineup_source)} | {escape(g.home_in.lineup_source)}</span></h3>",
        _shape(g),
        _team_table(g, ctx),
        _out_line(g, ctx),
        _players_line(g, ctx),
        _pod_block(g, ctx),
        _vsin_block(g, ctx),
    ]
    if buys:
        parts.append("<p class='ctx'><b>Buys</b></p>" + _rows_table(buys, limit=len(buys)))
    parts.append(
        "<p class='ctx'><b>Best reads</b> <span class='muted'>(positive edge, gated)</span></p>"
    )
    parts.append(_rows_table(reads, limit=6))
    if props:
        parts.append(
            f"<p class='ctx'><b>Props read</b> <span class='muted'>({len(props)} rows, research only — not bettable)</span></p>"
        )
        parts.append(_rows_table(props, limit=6))
    parts.append("</div>")
    return "".join(parts)


def render_html(card: SlateCard, ctx: ReportContext) -> str:
    buys = sorted((r for r in card.rows if r.is_buy), key=lambda r: -r.edge)
    n_props = sum(1 for r in card.rows if is_prop(r.market))
    head = (
        "<div class='masthead'><div class='brand'><span class='pp'>Payoff Pitch</span> · NHL</div>"
        f"<h1>NHL card — {card.slate_date:%A, %B %-d, %Y}</h1>"
        f"<p class='sub'>{len(card.games)} games priced from one joint period sim each; {len(card.rows)} selections read, "
        f"{len(buys)} buy{'s' if len(buys) != 1 else ''}; {n_props} prop rows (research only).</p>"
        f"<div class='dateline'>priced {escape(card.priced_at)} · tag {escape(card.tag)} · prior {escape(card.prior_version)}</div></div>"
    )
    if buys:
        slate = (
            "<div class='slatebets'><h2>Tonight's buys</h2>"
            "<p class='sbnote'>pass every gate (edge, EV, odds cap, fresh quote, confirmed goalies, verified settlement).</p>"
            + _rows_table(buys, limit=len(buys))
            + "</div>"
        )
    else:
        slate = (
            "<div class='slatebets'><h2>No buys tonight</h2>"
            "<p class='sbnote'>every positive-edge row is held by a gate; the reads below are for the ledger.</p></div>"
        )
    games = (
        "<h2>Games</h2>"
        + _pod_header(ctx)
        + _vsin_header(ctx)
        + "".join(_game_block(g, ctx) for g in card.games)
    )
    unpriced = (
        f"<p class='muted'>Unpriced: {escape('; '.join(card.unpriced))}</p>"
        if card.unpriced
        else ""
    )
    fine = (
        "<div class='fine'>Form from MoneyPuck game logs before the slate (W/L by goals; a goal tie is a shootout, shown as SO). "
        f"Ranks are 1 = best among {ctx.n_ranked} teams on EB-shrunk rates (preseason prior carries weight early). "
        "Props are priced for the ledger and never a buy until graded history clears the research_only gate. "
        "Model preview, not investment advice.</div>"
    )
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{head}{slate}{games}{unpriced}{fine}</body></html>"
    )


def to_pdf(html: str) -> bytes:
    from weasyprint import HTML

    return bytes(HTML(string=html).write_pdf())


def pdf_path(out_dir: Path, slate: Date, tag: str) -> Path:
    return out_dir / f"card_{slate.isoformat()}_{tag}.pdf"


def write_pdf(card: SlateCard, ctx: ReportContext, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_pdf(render_html(card, ctx)))
    return path


__all__ = [
    "CODES",
    "RANKED",
    "ReportContext",
    "TeamForm",
    "build_context",
    "league_ranks",
    "pdf_path",
    "render_html",
    "team_form",
    "write_pdf",
]
