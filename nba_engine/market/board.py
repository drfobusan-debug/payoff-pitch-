"""One consensus fair price and one execution price per selection.

A selection is ``(matchup, market, side, entity, line)``. From every book's latest
quote on it:

* ``fair`` -- the median across books of the **power** de-vigged probability,
  using only books that quote both sides of the same line (the NFL de-vig study:
  proportional de-vig manufactures a favourite-longshot slope, power does not).
  An unpaired side is counted but never priced.
* the execution price -- the better of DraftKings and BetMGM at that exact line,
  the only books a buy can be placed at (plan §10). ``exec_books`` says whether
  one or both posted it: the edge floor is fitted separately for each, because
  the better of two prices is a max-selection and a lone book's is not.

Every other book feeds the consensus and the archive only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import median

from engine_common.odds import american_to_prob
from nba_engine.data.capture import QuoteRow
from nfl_engine.market.fair import devig

EXEC_BOOKS: tuple[str, ...] = ("draftkings", "betmgm")
DEVIG_METHOD = "power"

SelKey = tuple[str, str, str, str, float | None]


def latest(rows: Iterable[QuoteRow]) -> list[QuoteRow]:
    """Latest quote per book on each selection of each game.

    Keyed on the event as well as the matchup: two teams meet three or four
    times a season, and a multi-day archive must not collapse those games.
    """
    out: dict[tuple[str, str, str, str, str, str, float | None, str], QuoteRow] = {}
    for r in sorted(rows, key=lambda r: r.captured_at):
        out[(r.game_date, r.event_id, *r.key)] = r
    return list(out.values())


@dataclass(frozen=True)
class Selection:
    game_date: str
    matchup: str
    event_id: str
    market: str
    side: str
    entity: str
    line: float | None
    fair: float | None  # None when no book paired both sides
    books: int
    paired_books: int
    exec_book: str | None
    exec_american: float | None
    exec_captured_at: str
    exec_prices: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> SelKey:
        return (self.matchup, self.market, self.side, self.entity, self.line)

    @property
    def exec_books(self) -> int:
        return len(self.exec_prices)

    @property
    def buyable(self) -> bool:
        """Priced (a paired consensus) and posted at an execution book."""
        return self.fair is not None and self.exec_american is not None


def novig(american: float, opposite: float, method: str = DEVIG_METHOD) -> float:
    return devig([american_to_prob(american), american_to_prob(opposite)], method)[0]


def selections(
    rows: Iterable[QuoteRow],
    *,
    exec_books: tuple[str, ...] = EXEC_BOOKS,
    method: str = DEVIG_METHOD,
) -> list[Selection]:
    groups: dict[tuple[str, str, SelKey], list[QuoteRow]] = defaultdict(list)
    for r in latest(rows):
        groups[(r.game_date, r.event_id, (r.matchup, r.market, r.side, r.entity, r.line))].append(r)
    out: list[Selection] = []
    for (_, _, (matchup, market, side, entity, line)), quotes in groups.items():
        fair = [
            novig(q.american, q.opposite_american, method)
            for q in quotes
            if q.opposite_american is not None
        ]
        ex = {q.book: q for q in quotes if q.book in exec_books}
        best = max(ex.values(), key=lambda q: (q.american, q.captured_at)) if ex else None
        out.append(
            Selection(
                game_date=quotes[0].game_date,
                matchup=matchup,
                event_id=quotes[0].event_id,
                market=market,
                side=side,
                entity=entity,
                line=line,
                fair=float(median(fair)) if fair else None,
                books=len(quotes),
                paired_books=len(fair),
                exec_book=best.book if best else None,
                exec_american=best.american if best else None,
                exec_captured_at=best.captured_at if best else "",
                exec_prices={b: q.american for b, q in ex.items()},
            )
        )
    out.sort(key=lambda s: (s.game_date, s.matchup, s.market, s.entity, s.line or 0.0, s.side))
    return out


def ev_per_unit(p: float, american: float) -> float:
    """Expected profit per unit staked at ``american`` if the true win chance is ``p``."""
    dec = 1.0 / american_to_prob(american)
    return p * (dec - 1.0) - (1.0 - p)


__all__ = [
    "DEVIG_METHOD",
    "EXEC_BOOKS",
    "Selection",
    "ev_per_unit",
    "latest",
    "novig",
    "selections",
]
