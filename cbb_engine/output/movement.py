"""Consensus line movement per game from the day's quote shards (display only).

Consensus = median across books of each pass's main number: the total, the home
spread, and the home moneyline. "Open" is the first pass captured for the slate,
"now" the last pass at or before the card.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median

from engine_runtime.quotes import Quote

_KEYS = {"game_total": "over", "game_ats": "home", "game_ml": "home"}


@dataclass(frozen=True)
class Move:
    market: str
    opened: float
    now: float
    opened_at: str
    now_at: str
    books: int

    @property
    def delta(self) -> float:
        return self.now - self.opened


def _consensus(qs: list[Quote]) -> float | None:
    if not qs:
        return None
    vals = [q.american if q.market == "game_ml" else q.line for q in qs]
    have = [v for v in vals if v is not None]
    return float(median(have)) if have else None


def moves(quotes: Iterable[Quote], *, until: str) -> dict[str, list[Move]]:
    """event_id -> movement for each market with at least two passes."""
    by: dict[tuple[str, str], dict[str, list[Quote]]] = defaultdict(lambda: defaultdict(list))
    for q in quotes:
        if q.market in _KEYS and q.side == _KEYS[q.market] and q.captured_at <= until:
            by[(q.event_id, q.market)][q.captured_at].append(q)
    out: dict[str, list[Move]] = defaultdict(list)
    for (eid, market), passes in by.items():
        times = sorted(passes)
        if len(times) < 2:
            continue
        first, last = _consensus(passes[times[0]]), _consensus(passes[times[-1]])
        if first is None or last is None:
            continue
        out[eid].append(Move(market, first, last, times[0], times[-1], len(passes[times[-1]])))
    order = list(_KEYS)
    return {e: sorted(m, key=lambda x: order.index(x.market)) for e, m in out.items()}


__all__ = ["Move", "moves"]
