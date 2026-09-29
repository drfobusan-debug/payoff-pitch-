"""Collapse an archive day into one consensus + best price per selection.

A selection is ``(matchup, market, side, entity, line)``. For each we keep

* the latest quote per book (the archive holds every snapshot of the day),
* the devigged probability per book -- two-way from the recorded partner,
  three-way from the book's own three sides -- and the median across books as
  the consensus the model's edge is measured against,
* the best (longest) price on the board and where it is,
* how old that best quote is, and whether any book gave us a two-sided market
  (a lone side with no partner cannot be devigged: ``one_way_quote``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median

from engine_common.odds import american_to_prob, devig
from nhl_engine.data.capture import QuoteRow, last_quotes

SelKey = tuple[str, str, str, str, float | None]


@dataclass(frozen=True)
class Selection:
    matchup: str
    market: str
    side: str
    entity: str
    line: float | None
    consensus: float  # median devigged probability across books (raw implied if one-way)
    books: int  # books quoting this side
    two_sided: int  # books where a devig was possible
    best_american: float
    best_book: str
    best_captured_at: str
    per_book: dict[str, float] = field(default_factory=dict)  # book -> american

    @property
    def key(self) -> SelKey:
        return (self.matchup, self.market, self.side, self.entity, self.line)

    @property
    def one_way(self) -> bool:
        return self.two_sided == 0

    def age_minutes(self, now: datetime) -> float:
        try:
            seen = datetime.strptime(self.best_captured_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return float("inf")
        return (now - seen.replace(tzinfo=timezone.utc)).total_seconds() / 60.0


def _three_way_probs(rows: list[QuoteRow]) -> dict[str, dict[str, float]]:
    """book -> side -> devigged probability for a 3-way market's rows."""
    by_book: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_book[r.book][r.side] = r.american
    out: dict[str, dict[str, float]] = {}
    for book, sides in by_book.items():
        if len(sides) == 3:
            out[book] = devig(sides)
    return out


def selections(rows: list[QuoteRow]) -> list[Selection]:
    latest = list(last_quotes(rows).values())
    groups: dict[SelKey, list[QuoteRow]] = defaultdict(list)
    ml3_rows: dict[tuple[str, str], list[QuoteRow]] = defaultdict(list)
    for r in latest:
        groups[(r.matchup, r.market, r.side, r.entity, r.line)].append(r)
        if r.market.endswith("ml3"):
            ml3_rows[(r.matchup, r.market)].append(r)
    ml3_devig = {k: _three_way_probs(v) for k, v in ml3_rows.items()}

    out: list[Selection] = []
    for key, quotes in groups.items():
        matchup, market, side, entity, line = key
        fair: list[float] = []
        raw: list[float] = []
        for q in quotes:
            raw.append(american_to_prob(q.american))
            if market.endswith("ml3"):
                p = ml3_devig.get((matchup, market), {}).get(q.book, {}).get(side)
                if p is not None:
                    fair.append(p)
            elif q.opposite_american is not None:
                fair.append(devig({"a": q.american, "b": q.opposite_american})["a"])
        best = max(quotes, key=lambda q: (q.american, q.captured_at))
        out.append(
            Selection(
                matchup=matchup,
                market=market,
                side=side,
                entity=entity,
                line=line,
                consensus=float(median(fair)) if fair else float(median(raw)),
                books=len(quotes),
                two_sided=len(fair),
                best_american=best.american,
                best_book=best.book,
                best_captured_at=best.captured_at,
                per_book={q.book: q.american for q in quotes},
            )
        )
    out.sort(key=lambda s: (s.matchup, s.market, s.entity, s.line or 0.0, s.side))
    return out


def matchups(rows: list[QuoteRow]) -> dict[str, tuple[str, str, str]]:
    """matchup -> (away, home, event_id) for every game on the board."""
    out: dict[str, tuple[str, str, str]] = {}
    for r in rows:
        if r.matchup in out or " @ " not in r.matchup:
            continue
        away, home = r.matchup.split(" @ ", 1)
        out[r.matchup] = (away, home, r.event_id)
    return out


__all__ = ["Selection", "matchups", "selections"]
