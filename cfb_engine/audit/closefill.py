"""Backfill the per-game close for graded buys that have none.

Buys graded before the per-game close went in carry an unstamped daily close, which
:func:`cfb_engine.audit.clv.timely_close` rightly refuses. The Odds API keeps
historical snapshots every five minutes, so one snapshot ``LEAD_MIN`` before each
kickoff window holds the real close for every game starting then. Only sides
without a timely close are filled; a timely close already on file always wins.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta

from cfb_engine.audit.clv import ClosingQuote, _utc, timely_close
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.priced import BUY_TIERS

#: Minutes before kickoff to read the historical snapshot (inside the 45-minute close window).
LEAD_MIN = 10.0
#: Odds API historical odds: 10 credits per market per region.
HIST_MULT = 10
MARKETS = 3


def missing(
    entries: Iterable[LedgerEntry], closing_for: Callable[[str], dict[str, ClosingQuote]]
) -> dict[str, set[str]]:
    """Per slate date, the matchups with a graded buy whose market has no timely close."""
    out: dict[str, set[str]] = defaultdict(set)
    timely: dict[str, set[str]] = {}
    for e in entries:
        if e.tier not in BUY_TIERS or e.result not in ("win", "loss", "push"):
            continue
        if e.date not in timely:
            timely[e.date] = {
                k.rsplit("|", 1)[0] for k, q in closing_for(e.date).items() if timely_close(q)
            }
        if f"{e.matchup}|{e.market}" not in timely[e.date]:
            out[e.date].add(e.matchup)
    return dict(out)


def windows(kickoffs: dict[str, str], lead_min: float = LEAD_MIN) -> dict[str, list[str]]:
    """Snapshot time (ISO UTC) -> the matchups kicking off ``lead_min`` after it."""
    out: dict[str, list[str]] = defaultdict(list)
    for matchup, stamp in sorted(kickoffs.items()):
        kick = _utc(stamp)
        if kick is not None:
            out[_iso(kick - timedelta(minutes=lead_min))].append(matchup)
    return dict(sorted(out.items()))


def fill(
    existing: dict[str, ClosingQuote], fresh: dict[str, ClosingQuote]
) -> dict[str, ClosingQuote]:
    """Fresh timely quotes replace stale or missing sides; a timely close on file is kept."""
    add = {
        k: q
        for k, q in fresh.items()
        if timely_close(q) and not (k in existing and timely_close(existing[k]))
    }
    return {**existing, **add}


def cost(snapshots: int, regions: str, lookups: int = 0) -> int:
    """Credits for ``snapshots`` historical boards plus ``lookups`` 1-credit event lists."""
    return snapshots * HIST_MULT * MARKETS * max(1, len(regions.split(","))) + lookups


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["HIST_MULT", "LEAD_MIN", "MARKETS", "cost", "fill", "missing", "windows"]
