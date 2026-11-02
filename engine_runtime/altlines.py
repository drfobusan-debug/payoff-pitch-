"""Alternate spreads and totals read against the main number: buying vs selling points.

Engine-neutral. Each engine turns its graded alternate rows into ``Alt`` records
(points moved off the main number, whether the side is one the engine bought on
the main line) and this module tables them. Rows that share a game are not
independent, so the SE is clustered by game. Graded only: nothing here prices or buys.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

HEAD = ["cut", "games", "n", "W-L-P", "units", "ROI", "± SE (by game)"]
NOTE = (
    "Alternate numbers on the same side as the main line, one price per game, side and number "
    "(the best execution-book price). Bought = more points for the side (worse price); sold = fewer "
    "(better price). Graded only, never bought. Rows in one game move together, so judge by games, "
    "not n; a cut needs ~150 games and 2 SE before it says anything."
)
BUCKETS = ((1.0, "0.5-1"), (2.0, "1.5-2"), (3.0, "2.5-3"), (5.0, "3.5-5"), (math.inf, "5.5+"))


@dataclass(frozen=True)
class Alt:
    game: str  # clustering key: one game
    kind: str  # spread | total
    favour: float  # points the alternate gives the side over the main number (negative: sold)
    beside_buy: bool  # the engine bought this side on the main line
    outcome: str  # win | loss | push
    pnl: float  # 1u flat


def bucket(pts: float) -> str:
    a = abs(pts)
    return next(label for hi, label in BUCKETS if a <= hi + 1e-9)


def _cut(label: str, rows: list[Alt]) -> tuple[str, ...]:
    games: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        games[r.game].append(r.pnl)
    n = len(rows)
    units = sum(r.pnl for r in rows)
    roi = units / n
    var = sum((sum(p) - roi * len(p)) ** 2 for p in games.values())
    se = math.sqrt(var) / n if len(games) > 1 else float("nan")
    w = sum(r.outcome == "win" for r in rows)
    lo = sum(r.outcome == "loss" for r in rows)
    pu = sum(r.outcome == "push" for r in rows)
    return (
        label,
        str(len(games)),
        str(n),
        f"{w}-{lo}-{pu}",
        f"{units:+.1f}",
        f"{100 * roi:+.1f}%",
        "-" if math.isnan(se) else f"{100 * se:.1f}",
    )


def table(alts: Iterable[Alt]) -> list[tuple[str, ...]]:
    """Rows of ``HEAD``: per market, every alternate then the engine's own buys, by direction and size."""
    rows = [a for a in alts if a.favour and a.outcome in ("win", "loss", "push")]
    out: list[tuple[str, ...]] = []
    for kind in ("spread", "total"):
        for scope, keep in (("all", False), ("beside buys", True)):
            mine = [a for a in rows if a.kind == kind and (a.beside_buy or not keep)]
            for way, sign in (("bought", 1), ("sold", -1)):
                side = [a for a in mine if a.favour * sign > 0]
                if not side:
                    continue
                out.append(_cut(f"{kind} {scope}: {way}", side))
                for _, size in BUCKETS:
                    g = [a for a in side if bucket(a.favour) == size]
                    if g:
                        out.append(_cut(f"  {way} {size} pts", g))
    return out


def text(alts: Iterable[Alt]) -> list[str]:
    body = table(alts)
    if not body:
        return ["no graded alternate lines yet"]
    w = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(HEAD)]
    fmt = lambda r: " ".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return [fmt(tuple(HEAD)), *(fmt(r) for r in body)]


def best_price(rows: Iterable[tuple[tuple[object, ...], float, T]]) -> list[T]:
    """Keep one row per key, at the best American price (highest payout)."""
    best: dict[tuple[object, ...], tuple[float, T]] = {}
    for key, american, row in rows:
        dec = 1 + (american / 100 if american > 0 else 100 / -american)
        cur = best.get(key)
        if cur is None or dec > cur[0]:
            best[key] = (dec, row)
    return [v[1] for v in best.values()]


__all__ = ["HEAD", "NOTE", "Alt", "best_price", "bucket", "table", "text"]
