"""Podcast picks in a slate PDF: under each game, and their records at the back.

League-free HTML. Context only: nothing here reads a model probability into a
decision. Each pick prints as a bet, ``Name (record). SELECTION PRICE (edge)``,
the edge being the host's own reason for the bet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import escape
from typing import Generic, TypeVar

from engine_common.podcasts.extract import Pick
from engine_common.podcasts.picks import MARKET_LABEL, Placed, Record, Records, SlatePicks
from engine_common.podcasts.shows import VSIN_NAME, VSIN_SHOW

R = TypeVar("R")

CSS = """
.pod{page-break-inside:avoid;border:1px solid #e3d3a8;border-left:4px solid #8a6d1d;background:#fdfaf1;border-radius:5px;padding:5px 10px;margin:8px 0 4px;font-size:9.2pt;}
.pod h3{margin:0 0 2px;font-size:9.6pt;color:#6b5414;font-family:'DejaVu Sans',sans-serif;text-transform:uppercase;letter-spacing:.5px;}
.pod ul{margin:2px 0;padding-left:16px;}.pod li{margin:1px 0;}
.pod .k{display:inline-block;padding:0 6px;border-radius:8px;font-size:7.4pt;font-family:'DejaVu Sans',sans-serif;text-transform:uppercase;margin-right:3px;}
.pod .off{background:#6b5414;color:#fff;}.pod .lean{background:#e8dcb9;color:#4b3b0d;}
.pod .rec{color:#6b7280;font-size:8.2pt;}
.podrec{page-break-inside:avoid;border:1px solid #e3d3a8;border-left:4px solid #8a6d1d;background:#fdfaf1;border-radius:6px;padding:10px 14px;margin:14px 0 8px;font-size:9pt;}
.podrec h2{border:none;margin:0 0 4px;color:#6b5414;font-size:13pt;}
.podrec h4{margin:8px 0 2px;color:#6b5414;font-size:9.6pt;}
.podrec table{width:100%;border-collapse:collapse;font-family:'DejaVu Sans',sans-serif;font-size:8.2pt;}
.podrec th{text-align:left;font-weight:normal;color:#6b7280;border-bottom:1px solid #e3d3a8;padding:2px 4px;}
.podrec td{padding:2px 4px;border-bottom:1px solid #f1e8cf;}
.podrec .sbnote{color:#4b5563;font-style:italic;font-size:8.6pt;margin:2px 0 4px;}
"""


def reason(p: Pick) -> str:
    """The host's own reason for the bet, as extracted; never one of ours."""
    return p.edge or p.reason or "no reason given"


def bet_line(
    name: str, record: str, selection: str, price: float | None, edge: str, *, lean: bool = False
) -> str:
    """One pick as a bet: ``Name (record). SELECTION PRICE (edge)``; the price only if said."""
    px = f" {price:+.0f}" if price is not None else ""
    tag = "<span class='k lean'>Lean</span>" if lean else ""
    return (
        f"<li>{tag}<b>{escape(name)}</b> ({escape(record)}). "
        f"<b>{escape(selection)}{px}</b> ({escape(edge)})</li>"
    )


@dataclass
class PodcastView(Generic[R]):
    slate: SlatePicks[R]
    records: Records


def _wlp_units(r: Record) -> str:
    if not r.decided and not r.pushes:
        return "no graded picks yet"
    out = r.wlp
    if r.priced:
        out += f", {r.units:+.1f}u"
    return out


def _who(p: Pick) -> str:
    return escape(p.show_name) + (f" ({escape(p.host)})" if p.host else "")


def _price(p: Pick) -> str:
    return f" ({p.price:+.0f})" if p.price is not None else ""


def _item(pl: Placed[R], view: PodcastView[R]) -> str:
    p = pl.pick
    rec = view.records.official_for(p.show_name, p.host) or view.records.official_for(
        p.show_name, None
    )
    name = f"{p.host} · {p.show_name}" if p.host else p.show_name
    record = _wlp_units(rec) if rec else "no graded picks yet"
    return bet_line(name, record, pl.label, p.price, reason(p), lean=p.kind != "official")


def game_block(view: PodcastView[R] | None, matchup: str) -> str:
    """The game's podcast picks, or nothing when no show touched it."""
    if view is None:
        return ""
    picks = view.slate.for_game(matchup)
    if not picks:
        return ""
    picks.sort(key=lambda pl: (pl.pick.kind != "official", pl.pick.market, pl.pick.show_name))
    blocks = (
        ("Podcast picks", [pl for pl in picks if pl.pick.show != VSIN_SHOW]),
        ("VSiN best bets", [pl for pl in picks if pl.pick.show == VSIN_SHOW]),
    )
    return "".join(
        f"<div class='pod'><h3>{title}</h3><ul>"
        + "".join(_item(pl, view) for pl in mine)
        + "</ul></div>"
        for title, mine in blocks
        if mine
    )


def _pct(r: Record) -> str:
    if r.win_pct is None or r.se is None:
        return "—"
    return f"{r.win_pct:.0%} ± {r.se:.0%}"


def _row(r: Record, lean: Record | None = None) -> str:
    roi = f"{r.roi:+.0%} on {r.priced}" if r.roi is not None else "—"
    clv = f"{r.clv_mean:+.1f} on {r.clv_n}" if r.clv_mean is not None else "—"
    lean_cell = (
        f"<td>{lean.wlp if lean and (lean.decided or lean.pushes) else '—'}</td>"
        if lean is not None
        else ""
    )
    note = "underpowered" if r.underpowered else ""
    return (
        f"<tr><td>{escape(r.label)}</td><td>{r.wlp if r.decided or r.pushes else '—'}</td>"
        f"<td>{_pct(r)}</td><td>{r.units:+.1f}</td><td>{roi}</td><td>{clv}</td>"
        f"{lean_cell}<td>{note}</td></tr>"
    )


def _consensus(view: PodcastView[R], ordered: list[str]) -> str:
    items: list[str] = []
    for matchup in ordered:
        picks = [pl for pl in view.slate.for_game(matchup) if pl.pick.kind == "official"]
        if not picks:
            continue
        by_side: Counter[str] = Counter()
        for pl in picks:
            by_side[f"{MARKET_LABEL[pl.pick.market]} {pl.label}"] += 1
        sides = "; ".join(f"{n}× {escape(s)}" for s, n in by_side.most_common())
        items.append(f"<li><b>{escape(matchup)}</b>: {sides}</li>")
    if not items:
        return ""
    return "<h4>Consensus (official bets, by kickoff)</h4><ul>" + "".join(items) + "</ul>"


def _unmatched(view: PodcastView[R]) -> str:
    if not view.slate.unmatched:
        return ""
    rows = "".join(
        f"<li>{_who(p)} · {'Bet' if p.kind == 'official' else 'Lean'} · "
        f"{escape(p.description or p.team or '')}{_price(p)} "
        f"<span class='rec'>[{p.stamp}, {escape(p.episode_title[:60])}]</span></li>"
        for p in sorted(view.slate.unmatched, key=lambda p: (p.show_name, p.published, p.seconds))
    )
    return (
        "<h4>Not on this card</h4><p class='sbnote'>Picks that fit no game here, or more than one: "
        "futures, other slates, or a team the card could not place.</p><ul>" + rows + "</ul>"
    )


def records_block(view: PodcastView[R] | None, ordered: list[str]) -> str:
    """The league's podcast records, consensus in ``ordered`` game order, and the unplaced."""
    if view is None:
        return ""
    rec = view.records
    head = (
        "<tr><th>Show</th><th>Bets W-L-P</th><th>Win% ± SE</th><th>Units</th>"
        "<th>ROI (priced)</th><th>CLV pts</th><th>Leans</th><th></th></tr>"
    )
    shows = "".join(_row(off, lean) for off, lean in rec.shows)
    table = (
        f"<table>{head}{shows}</table>" if shows else "<p>No podcast pick has been graded yet.</p>"
    )
    hosts = ""
    if rec.hosts:
        hosts = (
            "<h4>By host (only picks the episode attributes)</h4><table>"
            + head.replace("Show", "Host")
            + "".join(_row(off, lean) for off, lean in rec.hosts)
            + "</table>"
        )
    markets = ""
    if rec.markets:
        mh = head.replace("<th>Leans</th>", "").replace("Show", "Show · market")
        markets = (
            "<h4>Bets by market</h4><table>"
            + mh
            + "".join(_row(m) for m in rec.markets)
            + "</table>"
        )
    note = (
        "<p class='sbnote'>Graded at the number the host said. Units only where a price is "
        "known: the host's own, or the board's at the host's number. A stake is 1u "
        "unless the host named one. CLV is points of line value against the close (spreads "
        "and totals). A bet's edge is the host's own reason for it, as they gave it. "
        "Underpowered: within 2 SE of the 52.4% needed at -110.</p>"
    )
    return (
        "<div class='podrec'><h2>"
        + (
            "Podcast & VSiN records"
            if any(off.label == VSIN_NAME for off, _ in rec.shows)
            else "Podcast records"
        )
        + "</h2>"
        + note
        + table
        + hosts
        + markets
        + _consensus(view, ordered)
        + _unmatched(view)
        + "</div>"
    )


__all__ = [
    "CSS",
    "PodcastView",
    "bet_line",
    "game_block",
    "reason",
    "records_block",
]
