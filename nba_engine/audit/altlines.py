"""Alternate spreads and totals, tracked beside each game's main number.

Every alternate the books post is a ledger row like any other (``alt_ats``,
``alt_total``), priced, closed and graded the same way and never bought. This
module only reads them back against the game's main number: an alternate that
gives the side more points than the main line has *bought* points (a worse
price), one that gives fewer has *sold* them (a better price).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from nba_engine.audit.ledger import LedgerRow

ALT: dict[str, str] = {"alt_ats": "game_ats", "alt_total": "game_total"}
MAX_BUCKET = 7

Row = tuple[str, int, int, int, int, float, float]  # label, n, W, L, P, units, ROI


def _home(r: LedgerRow) -> str:
    return r.matchup.rsplit(" @ ", 1)[-1]


def main_lines(rows: Iterable[LedgerRow], *, by_pass: bool = False) -> dict[tuple[str, ...], float]:
    """``(event_id, game_ats|game_total[, pass_tag])`` -> main number (spreads from home).

    The main number is the line most books pair, as on the card. ``by_pass``
    keys it per pricing pass, so a number the books moved off is not compared
    with alternates priced after the move.
    """
    groups: dict[tuple[str, ...], dict[float, list[LedgerRow]]] = defaultdict(dict)
    for r in rows:
        if r.market not in ALT.values() or r.line is None or r.entity:
            continue
        key = r.line if r.market == "game_total" or r.side == _home(r) else -r.line
        at = (r.event_id, r.market, r.pass_tag) if by_pass else (r.event_id, r.market)
        groups[at].setdefault(key, []).append(r)
    return {
        k: max(
            g.items(),
            key=lambda kv: (
                max(x.paired_books for x in kv[1]),
                max(x.books for x in kv[1]),
                len(kv[1]),
            ),
        )[0]
        for k, g in groups.items()
    }


def favour(r: LedgerRow, m: float | None) -> float | None:
    """Points the alternate gives this side over main number ``m`` (negative: sold)."""
    if m is None or r.line is None:
        return None
    if r.market == "alt_total":
        return round(m - r.line if r.side == "over" else r.line - m, 1)
    mine = m if r.side == _home(r) else -m
    return round(r.line - mine, 1)


def _bucket(pts: float) -> str:
    b = math.ceil(pts - 1e-9)
    return f"{MAX_BUCKET}+" if b >= MAX_BUCKET else str(b)


def report(rows: Iterable[LedgerRow], passes: Iterable[LedgerRow] = ()) -> list[Row]:
    """Graded alternates at a DraftKings/BetMGM price, by market, direction and points.

    ``passes`` (every pricing pass's rows) gives each alternate the main number
    of the pass that priced it; without them, the slate's of-record main is used.
    """
    rows = list(rows)
    main = main_lines(rows)
    at_pass = main_lines(passes, by_pass=True)
    groups: dict[str, list[LedgerRow]] = defaultdict(list)
    for r in rows:
        if r.market not in ALT or r.american is None or not r.graded:
            continue
        base = ALT[r.market]
        f = favour(r, at_pass.get((r.event_id, base, r.pass_tag), main.get((r.event_id, base))))
        if not f:
            continue
        way = "bought" if f > 0 else "sold"
        groups[f"{r.market} {way} (all)"].append(r)
        groups[f"{r.market} {way} {_bucket(abs(f))} pts"].append(r)
    out: list[Row] = []
    for label in sorted(groups):
        g = groups[label]
        units = sum(r.pnl or 0.0 for r in g)
        out.append(
            (
                label,
                len(g),
                sum(r.outcome == "win" for r in g),
                sum(r.outcome == "loss" for r in g),
                sum(r.outcome == "push" for r in g),
                round(units, 4),
                units / len(g),
            )
        )
    return out


__all__ = ["ALT", "favour", "main_lines", "report"]
