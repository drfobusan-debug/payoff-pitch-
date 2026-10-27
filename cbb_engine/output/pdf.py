"""CBB card PDF: the priced board per game with a Torvik team panel (display only).

Nothing here feeds pricing, gates or the ledger; it reads finished ledger rows and
Torvik's archived game logs.
"""

from __future__ import annotations

import html
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date as Date
from datetime import datetime
from zoneinfo import ZoneInfo

from cbb_engine.data.injuries import Injury
from cbb_engine.output.movement import Move
from cbb_engine.output.teamstats import Factors, Rated, Split, TeamStats, match_team
from engine_runtime import ledger

ET = ZoneInfo("America/New_York")

_STYLE = """
@page { size: letter; margin: 0.45in; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; color: #111; }
h1 { font-size: 14pt; margin: 0 0 2px; } .sub { color: #555; margin: 0 0 8px; }
.game { break-inside: avoid; border-top: 1.5px solid #333; padding-top: 4px; margin-bottom: 10px; }
.gh { font-size: 10.5pt; font-weight: bold; } .gh span { font-weight: normal; color: #555; font-size: 8.5pt; }
table { border-collapse: collapse; } td, th { padding: 1.5px 5px; text-align: right; }
th { font-weight: bold; color: #333; border-bottom: 1px solid #999; } td.l, th.l { text-align: left; }
table.board { width: 100%; margin: 3px 0; } table.board td { border-bottom: 0.5px solid #ddd; }
tr.buy td { background: #e6f4e6; font-weight: bold; }
table.panel { width: 100%; } table.panel > tbody > tr > td { vertical-align: top; padding: 0 6px 0 0; width: 50%; }
table.inner th { font-size: 7.5pt; } table.inner td { white-space: nowrap; }
table.inner { width: 100%; } .rk { color: #999; font-size: 6.5pt; } .on { background: #eef2fa; }
.ctx { margin: 2px 0; } .cap { font-weight: bold; text-align: left; vertical-align: bottom; } .note { color: #777; font-size: 7pt; margin: 2px 0 0; }
"""

_MARKETS = {"game_total": "Total", "game_ats": "Spread", "game_ml": "Moneyline"}


def _tip(tip_utc: str) -> str:
    try:
        t = datetime.fromisoformat(tip_utc.replace("Z", "+00:00")).astimezone(ET)
    except ValueError:
        return ""
    return t.strftime("%-I:%M %p ET")


def _rk(r: Rated, fmt: str) -> str:
    if r.value is None:
        return "—"
    rank = f" <span class='rk'>{r.rank}</span>" if r.rank else ""
    return f"{fmt.format(r.value)}{rank}"


def _board(rows: Sequence[ledger.Row]) -> str:
    """One line per market/side/number: the best-priced execution book."""
    best: dict[tuple[str, str, float | None], ledger.Row] = {}
    for r in rows:
        if r.market not in _MARKETS or r.american is None:
            continue
        k = (r.market, r.side, r.line)
        if k not in best or (r.edge or -9) > (best[k].edge or -9):
            best[k] = r
    if not best:
        return "<p class='note'>No full-game prices on this pass.</p>"
    order = list(_MARKETS)
    body = []
    for r in sorted(best.values(), key=lambda r: (order.index(r.market), r.side, r.line or 0)):
        line = (
            "" if r.line is None else (f"{r.line:+g}" if r.market == "game_ats" else f"{r.line:g}")
        )
        verdict = (
            r.tier
            if r.is_buy
            else ("shadow" if r.status != "live" else ", ".join(r.gates) or "pass")
        )
        body.append(
            f"<tr class='{'buy' if r.is_buy and r.status == 'live' else ''}'>"
            f"<td class='l'>{_MARKETS[r.market]}</td><td class='l'>{html.escape(r.side)} {line}</td>"
            f"<td class='l'>{html.escape(r.book)}</td><td>{r.american:+.0f}</td>"
            f"<td>{'' if r.p_fair is None else f'{r.p_fair:.1%}'}</td>"
            f"<td>{'' if r.edge is None else f'{r.edge:+.1%}'}</td><td class='l'>{html.escape(verdict)}</td></tr>"
        )
    return (
        "<table class='board'><tr><th class='l'>Market</th><th class='l'>Side</th><th class='l'>Book</th>"
        "<th>Price</th><th>Fair</th><th>Edge</th><th class='l'>Status</th></tr>"
        + "".join(body)
        + "</table>"
    )


def _ratings(away: TeamStats | None, home: TeamStats | None, an: str, hn: str) -> str:
    """Off/def/net per 100 possessions at home and on the road; tonight's venue shaded."""
    cols: list[tuple[str, Split | None, bool]] = [
        (f"{an} road", away.away if away else None, True),
        (f"{an} home", away.home if away else None, False),
        (f"{hn} home", home.home if home else None, True),
        (f"{hn} road", home.away if home else None, False),
    ]
    head = (
        f"<tr><th class='l cap' rowspan='2'>Ratings</th><th colspan='2'>{html.escape(an)}</th>"
        f"<th colspan='2'>{html.escape(hn)}</th></tr><tr>"
        + "".join(
            f"<th class='{'on' if on else ''}'>{c.rsplit(' ', 1)[1].title()}</th>"
            for c, _, on in cols
        )
        + "</tr>"
    )
    rows = []
    for label, attr in (("Off rtg", "off"), ("Def rtg", "dfn"), ("Net", "net")):
        fmt = "{:+.1f}" if attr == "net" else "{:.1f}"
        cells = "".join(
            f"<td class='{'on' if on else ''}'>{_rk(getattr(s, attr), fmt) if s else '—'}</td>"
            for _, s, on in cols
        )
        rows.append(f"<tr><td class='l'>{label}</td>{cells}</tr>")
    games = "".join(
        f"<td class='{'on' if on else ''}'>{s.games if s else 0}</td>" for _, s, on in cols
    )
    rows.append(f"<tr><td class='l'>Games</td>{games}</tr>")
    adj = "".join(
        f"<td colspan='2'>{_rk(t.adj_off, '{:.1f}')} / {_rk(t.adj_def, '{:.1f}')}</td>"
        if t
        else "<td colspan='2'>—</td>"
        for t in (away, home)
    )
    rows.append(f"<tr><td class='l'>Adj O / D</td>{adj}</tr>")
    return f"<table class='inner'>{head}{''.join(rows)}</table>"


def _factors(away: TeamStats | None, home: TeamStats | None, an: str, hn: str) -> str:
    """Each offense beside the defense it meets tonight."""
    cols: list[tuple[str, Factors | None]] = [
        (f"{an} O", away.offense if away else None),
        (f"{hn} D", home.defense if home else None),
        (f"{hn} O", home.offense if home else None),
        (f"{an} D", away.defense if away else None),
    ]
    head = (
        f"<tr><th class='l cap' rowspan='2'>Four factors</th><th colspan='2'>{html.escape(an)} offense</th>"
        f"<th colspan='2'>{html.escape(hn)} offense</th></tr><tr>"
        f"<th>{html.escape(an)} O</th><th>{html.escape(hn)} D</th><th>{html.escape(hn)} O</th>"
        f"<th>{html.escape(an)} D</th></tr>"
    )
    rows = []
    for label, attr in (("eFG%", "efg"), ("TOV%", "tov"), ("OREB%", "orb"), ("FT rate", "ftr")):
        cells = "".join(
            f"<td>{_rk(getattr(f, attr), '{:.1%}') if f else '—'}</td>" for _, f in cols
        )
        rows.append(f"<tr><td class='l'>{label}</td>{cells}</tr>")
    return f"<table class='inner'>{head}{''.join(rows)}</table>"


def _rest_days(t: TeamStats | None, day: Date) -> int | None:
    return None if t is None or t.last_game is None else (day - t.last_game).days - 1


def _rest(t: TeamStats | None, day: Date) -> str:
    n = _rest_days(t, day)
    if n is None:
        return "first game"
    return "back-to-back" if n == 0 else f"{n}d rest"


def rest_edge(away: TeamStats | None, home: TeamStats | None, day: Date, an: str, hn: str) -> str:
    """ "Duke +2 days rest"; a back-to-back is called out."""
    a, h = _rest_days(away, day), _rest_days(home, day)
    if a is None or h is None:
        return "rest unknown (a team has no game logged yet)"
    if a == h:
        return f"even ({a}d each)"
    team, diff = (an, a - h) if a > h else (hn, h - a)
    tired = [n for n, d in ((an, a), (hn, h)) if d == 0]
    return f"{team} +{diff} day{'s' if diff > 1 else ''} rest" + (
        f" ({', '.join(tired)} on a back-to-back)" if tired else ""
    )


_FACTORS = (("efg", "eFG%"), ("tov", "TOV%"), ("orb", "OREB%"), ("ftr", "FT rate"))


def unit_edge(
    off: TeamStats | None, dfn: TeamStats | None, off_venue: str, dfn_venue: str, on: str, dn: str
) -> str:
    """Which unit has the edge: offense rank vs defense rank (1 = best), tonight's venues."""
    if off is None or dfn is None:
        return f"{on} O vs {dn} D: no games logged yet"
    o: Split = getattr(off, off_venue)
    d: Split = getattr(dfn, dfn_venue)
    if o.off.rank is None or d.dfn.rank is None:
        rating = "venue split not played yet"
        lead = None
    else:
        lead = on if o.off.rank < d.dfn.rank else dn if d.dfn.rank < o.off.rank else None
        rating = f"off rtg #{o.off.rank} vs def rtg #{d.dfn.rank}"
    wins = [
        lab
        for k, lab in _FACTORS
        if (ra := getattr(off.offense, k).rank) is not None
        and (rb := getattr(dfn.defense, k).rank) is not None
        and ra < rb
    ]
    n = sum(
        1
        for k, _ in _FACTORS
        if getattr(off.offense, k).rank is not None and getattr(dfn.defense, k).rank is not None
    )
    who = f"edge {on} offense" if lead == on else f"edge {dn} defense" if lead == dn else "even"
    return f"{on} O vs {dn} D: {who} ({rating}); offense better in {len(wins)} of {n} factors" + (
        f" ({', '.join(wins)})" if wins else ""
    )


def _injuries(items: Sequence[tuple[str | None, Injury]], team: str | None, label: str) -> str:
    mine = [i for t, i in items if team is not None and t == team]
    if not mine:
        return f"{label}: none listed"
    return f"{label}: " + "; ".join(
        f"{i.player} ({i.position}, {i.injury}) {i.status}" for i in mine
    )


def _movement(ms: Sequence[Move], an: str, hn: str) -> str:
    if not ms:
        return "no movement yet (one pass or none captured)"
    parts = []
    for m in ms:
        if m.market == "game_total":
            parts.append(f"Total {m.opened:g} → {m.now:g} ({m.delta:+g})")
        elif m.market == "game_ats":
            parts.append(f"Spread {hn} {m.opened:+g} → {m.now:+g} ({m.delta:+g})")
        else:
            parts.append(f"ML {hn} {m.opened:+.0f} → {m.now:+.0f}")
    return (
        " · ".join(parts)
        + f" <span class='rk'>(median of {ms[0].books} books, open {_tip(ms[0].opened_at)})</span>"
    )


def team_panel(away: TeamStats | None, home: TeamStats | None, an: str, hn: str) -> str:
    if away is None and home is None:
        return (
            "<p class='note'><b>Team ratings</b> — no Torvik games logged yet for either team.</p>"
        )
    return (
        "<table class='panel'><tr>"
        f"<td>{_ratings(away, home, an, hn)}</td><td>{_factors(away, home, an, hn)}</td></tr></table>"
    )


KEY = (
    "Team panels (display only, from Torvik game logs): off/def rtg = points scored/allowed per 100 possessions,"
    " season to date, raw; Adj O / D = Torvik's opponent-adjusted ratings. Four factors pool every game: eFG%,"
    " TOV% = turnovers per possession, OREB% = share of own misses rebounded, FT rate = FTA/FGA; each offense sits"
    " beside the defense it meets tonight. Grey # = rank among D-I teams (1 = best); tonight's venue shaded."
    " Unit edge compares those ranks at tonight's venues. Early-season splits rest on few games, so ranks are noisy."
)


def build_html(
    day: Date,
    rows: Sequence[ledger.Row],
    teams: Mapping[str, tuple[str, str]],
    stats: Mapping[str, TeamStats],
    *,
    title: str = "",
    injuries: Sequence[Injury] = (),
    injuries_source: str = "",
    moves: Mapping[str, Sequence[Move]] | None = None,
) -> str:
    """``teams`` maps event_id -> (home, away) as named by the Odds API."""
    names = list(stats)
    listed = [(match_team(i.team, names), i) for i in injuries]
    unmatched = sorted({i.team for t, i in listed if t is None})
    by_event: dict[str, list[ledger.Row]] = defaultdict(list)
    for r in rows:
        by_event[r.event_id].append(r)
    buys = sum(1 for r in rows if r.is_buy and r.status == "live")
    blocks = []
    for eid in sorted(
        set(by_event) | set(teams), key=lambda e: (by_event[e][0].tip_utc if by_event[e] else "", e)
    ):
        home_name, away_name = teams.get(eid, ("", ""))
        if not home_name:
            continue
        away, home = (stats.get(match_team(n, stats) or "") for n in (away_name, home_name))
        an, hn = (t.team if t else n.split()[0] for t, n in ((away, away_name), (home, home_name)))
        tip = _tip(by_event[eid][0].tip_utc) if by_event[eid] else ""
        blocks.append(
            f"<div class='game'><div class='gh'>{html.escape(away_name)} at {html.escape(home_name)} "
            f"<span>{tip} · {an} {_rest(away, day)}, {hn} {_rest(home, day)}</span></div>"
            f"{_board(by_event[eid])}"
            f"<p class='ctx'><b>Line movement</b> — {_movement((moves or {}).get(eid, ()), an, hn)}</p>"
            f"{team_panel(away, home, an, hn)}"
            f"<p class='ctx'><b>Unit edge</b> — {html.escape(unit_edge(away, home, 'away', 'home', an, hn))}"
            f"<br>{html.escape(unit_edge(home, away, 'home', 'away', hn, an))}</p>"
            f"<p class='ctx'><b>Rest</b> — {html.escape(rest_edge(away, home, day, an, hn))}</p>"
            f"<p class='ctx'><b>Injuries</b> {html.escape(injuries_source)}— "
            f"{html.escape(_injuries(listed, away.team if away else None, an))} · "
            f"{html.escape(_injuries(listed, home.team if home else None, hn))}</p></div>"
        )
    return (
        f"<html><head><meta charset='utf-8'><style>{_STYLE}</style></head><body>"
        f"<h1>CBB card — {day.isoformat()}</h1><p class='sub'>{html.escape(title)} {buys} live buy(s) · "
        f"{len(blocks)} games · paper only</p><p class='note'>{KEY}{_unmatched(unmatched)}</p>{''.join(blocks)}</body></html>"
    )


def _unmatched(teams: Sequence[str]) -> str:
    if not teams:
        return ""
    return html.escape(
        f" Injury-list schools not matched to a Torvik team (not shown): {', '.join(teams)}."
    )


def render_pdf(html_body: str) -> bytes:
    from weasyprint import HTML

    return bytes(HTML(string=html_body).write_pdf())


__all__ = ["build_html", "render_pdf", "team_panel"]
