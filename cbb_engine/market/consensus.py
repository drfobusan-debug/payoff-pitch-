"""§3.7 fair price: power-devigged all-book median with the execution books left out.

One ``Anchor`` per (event, market, line): the fair probability of each side, the
books behind it, the execution quotes we can actually bet, and the stale flag —
"consensus moved ≥ ½ pt / 5¢ since the previous pass while this book's quote did
not". The flag is the market-only hypothesis; it never enters ``p_model``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import median

from cbb_engine.config import EXECUTION_BOOKS
from engine_common.odds import american_to_prob
from engine_runtime import devig
from engine_runtime.quotes import Quote

STALE_PTS = 0.5
STALE_CENTS = 5.0


@dataclass(frozen=True)
class ExecQuote:
    book: str
    side: str
    american: float
    opposite_american: float | None
    stale: bool = False


@dataclass
class Anchor:
    event_id: str
    tip_utc: str
    game_date: str
    market: str
    line: float | None
    sides: tuple[str, str]
    p_fair: dict[str, float]  # side -> devigged consensus probability
    books: int
    paired_books: int
    exec_quotes: list[ExecQuote] = field(default_factory=list)
    one_way: bool = False
    shard: str = ""

    @property
    def sharp_books(self) -> int:
        return self.paired_books

    @property
    def consensus_median_line(self) -> float | None:
        return self.line


def _sides(market: str, rows: list[Quote]) -> tuple[str, str]:
    names = sorted({r.side for r in rows})
    if market.endswith("total") or market.endswith("total_alt"):
        return ("over", "under")
    if "home" in names and "away" in names:
        return ("home", "away")
    return (names[0], names[1]) if len(names) == 2 else (names[0], "")


def build(rows: Iterable[Quote], *, shard: str = "") -> list[Anchor]:
    """Group a pass's quotes by (event, market, line) and anchor each."""
    groups: dict[tuple[str, str, float | None], list[Quote]] = defaultdict(list)
    for r in rows:
        groups[(r.event_id, r.market, r.line)].append(r)
    out: list[Anchor] = []
    for (eid, market, line), qs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        a, b = _sides(market, qs)
        by_side: dict[str, list[Quote]] = defaultdict(list)
        for q in qs:
            by_side[q.side].append(q)
        side_a = by_side.get(a, [])
        fair_a, seen, used = devig.consensus(
            ((q.book, q.american, q.opposite_american) for q in side_a),
            exclude_books=EXECUTION_BOOKS,
        )
        p_fair = {a: fair_a, b: 1.0 - fair_a} if fair_a is not None else {}
        execs = [
            ExecQuote(q.book, q.side, q.american, q.opposite_american)
            for q in qs
            if q.book in EXECUTION_BOOKS
        ]
        out.append(
            Anchor(
                event_id=eid,
                tip_utc=qs[0].tip_utc,
                game_date=qs[0].game_date,
                market=market,
                line=line,
                sides=(a, b),
                p_fair=p_fair,
                books=seen,
                paired_books=used,
                exec_quotes=execs,
                one_way=any(q.opposite_american is None for q in execs),
                shard=shard,
            )
        )
    return out


def consensus_line(anchors: Iterable[Anchor], event_id: str, market: str) -> float | None:
    """The median posted line across books for a market — the location before the devig shift."""
    lines = [a.line for a in anchors if a.event_id == event_id and a.market == market and a.line is not None and a.paired_books]
    return float(median(lines)) if lines else None


def flag_stale(current: list[Anchor], previous: list[Anchor]) -> None:
    """Mark execution quotes whose price held while the consensus moved (§3.7 market-only channel)."""
    prev_fair = {(p.event_id, p.market, p.line, s): v for p in previous for s, v in p.p_fair.items()}
    prev_line = defaultdict(list)
    for p in previous:
        if p.line is not None and p.paired_books:
            prev_line[(p.event_id, p.market)].append(p.line)
    prev_exec = {(p.event_id, p.market, p.line, e.book, e.side): e.american for p in previous for e in p.exec_quotes}
    cur_line = defaultdict(list)
    for a in current:
        if a.line is not None and a.paired_books:
            cur_line[(a.event_id, a.market)].append(a.line)
    for a in current:
        moved_pts = False
        if (a.event_id, a.market) in prev_line and (a.event_id, a.market) in cur_line:
            moved_pts = abs(median(cur_line[(a.event_id, a.market)]) - median(prev_line[(a.event_id, a.market)])) >= STALE_PTS
        flagged: list[ExecQuote] = []
        for e in a.exec_quotes:
            pf, pp = a.p_fair.get(e.side), prev_fair.get((a.event_id, a.market, a.line, e.side))
            moved_cents = pf is not None and pp is not None and abs(pf - pp) * 100 >= STALE_CENTS
            held = prev_exec.get((a.event_id, a.market, a.line, e.book, e.side)) == e.american
            flagged.append(ExecQuote(e.book, e.side, e.american, e.opposite_american, stale=held and (moved_pts or moved_cents)))
        a.exec_quotes = flagged


def implied(american: float) -> float:
    return american_to_prob(american)


__all__ = ["STALE_CENTS", "STALE_PTS", "Anchor", "ExecQuote", "build", "consensus_line", "flag_stale", "implied"]
