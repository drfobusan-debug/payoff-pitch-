"""Alternate puck lines and totals read against the main number (graded only).

The ledger prices every alternate the books post (``game_pl_alt``,
``game_total_alt``); this reads them back against the game's main puck line or
total for the shared alt-lines table. Nothing here prices or buys.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from engine_runtime import altlines as rt
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.market.board import EXEC_BOOKS

ALT: dict[str, str] = {"game_pl_alt": "game_pl", "game_total_alt": "game_total"}


def main_lines(rows: Iterable[LedgerRow]) -> dict[tuple[str, str, str], float]:
    """``(night, matchup, side)`` -> main puck line; ``(night, matchup, "total")`` -> main total."""
    seen: dict[tuple[str, str, str], Counter[float]] = defaultdict(Counter)
    for r in rows:
        if r.line is None or r.entity:
            continue
        if r.market == "game_pl":
            seen[(r.slate_date, r.matchup, r.side)][r.line] += 1
        elif r.market == "game_total":
            seen[(r.slate_date, r.matchup, "total")][r.line] += 1
    return {k: c.most_common(1)[0][0] for k, c in seen.items()}


def favour(r: LedgerRow, main: dict[tuple[str, str, str], float]) -> float | None:
    """Points the alternate gives this side over the main number (negative: sold)."""
    if r.line is None:
        return None
    if r.market == "game_total_alt":
        m = main.get((r.slate_date, r.matchup, "total"))
        if m is None:
            return None
        return round(m - r.line if r.side.lower() == "over" else r.line - m, 1)
    m = main.get((r.slate_date, r.matchup, r.side))
    return None if m is None else round(r.line - m, 1)


def alts(rows: Iterable[LedgerRow]) -> list[rt.Alt]:
    """Graded alternates at the best DraftKings/BetMGM price."""
    rows = [r for r in rows if not r.priced_late]
    main = main_lines(rows)
    buys = {(r.slate_date, r.matchup, r.market, r.side) for r in rows if r.is_buy and r.market in ALT.values()}
    keyed = []
    for r in rows:
        if r.market not in ALT or r.outcome not in ("win", "loss", "push") or r.book not in EXEC_BOOKS:
            continue
        f = favour(r, main)
        if f and r.american is not None:
            keyed.append(((r.slate_date, r.matchup, r.market, r.side, r.line), r.american, (r, f)))
    out: list[rt.Alt] = []
    for row in rt.best_price(keyed):
        r, f = row
        out.append(
            rt.Alt(
                game=f"{r.slate_date} {r.matchup}",
                kind="spread" if r.market == "game_pl_alt" else "total",
                favour=f,
                beside_buy=(r.slate_date, r.matchup, ALT[r.market], r.side) in buys,
                outcome=str(r.outcome),
                pnl=r.pnl or 0.0,
            )
        )
    return out


__all__ = ["ALT", "alts", "favour", "main_lines"]
