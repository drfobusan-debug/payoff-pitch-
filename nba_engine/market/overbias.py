"""How far the de-vigged close overstates the Over, fitted per market and book.

For every two-way Over/Under selection at the close the realised Over rate is
compared with the no-vig probability. The gap (hit minus no-vig, in probability)
is the Over bias; it is measured against the consensus fair price -- the number
the engine prices against -- and against each execution book's own no-vig
price, for the audit. Uncertainty is a bootstrap over games, since one game's
props are not independent draws.

A gap is only applied when its 95% interval excludes zero; otherwise the fair
price is left alone. The fitted values go to the versioned params store
(``params.py``), never into code or config (plan §4.7, §13).

    p_fair(Over)  = consensus + gap
    p_fair(Under) = consensus(Under) - gap
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from nba_engine.data.boxes import norm_name
from nba_engine.data.capture import QuoteRow, last_quotes
from nba_engine.market.board import EXEC_BOOKS, Selection, novig, selections
from nba_engine.schemas import GameResult

NAME = "over_bias"
CONSENSUS = "consensus"
PROP_STAT = {
    "pl_pts": "points",
    "pl_3pm": "threes",
    "pl_reb": "rebounds",
    "pl_ast": "assists",
    "pl_pra": "pra",
}
OU_MARKETS = ("game_total", "h1_total", *PROP_STAT)

GameKey = tuple[str, str]  # (slate date, matchup)


@dataclass(frozen=True)
class Graded:
    game: GameKey
    market: str
    book: str  # an execution book, or CONSENSUS
    p_over: float  # no-vig P(Over)
    hit: int


@dataclass(frozen=True)
class Gap:
    market: str
    book: str
    n: int
    games: int
    p_over: float
    hit_rate: float
    gap: float
    lo: float
    hi: float

    @property
    def applied(self) -> float:
        """The shift used in pricing: the gap if its interval excludes zero, else 0."""
        return self.gap if self.hi < 0.0 or self.lo > 0.0 else 0.0


def outcome(market: str, entity: str, game: GameResult) -> float | None:
    """The settled number an Over/Under on ``market`` is graded on; ``None`` = void."""
    if market == "game_total":
        return float(game.final_away + game.final_home)
    if market == "h1_total":
        return float(game.h1_away + game.h1_home)
    stat = PROP_STAT.get(market)
    if stat is None:
        return None
    key = norm_name(entity)
    for p in game.players:
        if norm_name(p.name) == key:
            if p.dnp or p.minutes <= 0:
                return None
            return float(p.pra if stat == "pra" else getattr(p, stat))
    return None


def _hit(value: float | None, line: float | None) -> int | None:
    if value is None or line is None or value == line:
        return None
    return int(value > line)


def grade(
    rows: Iterable[QuoteRow],
    finals: Mapping[GameKey, GameResult],
    *,
    books: tuple[str, ...] = EXEC_BOOKS,
) -> list[Graded]:
    """Every graded Over: one consensus row per selection, one per execution-book quote."""
    rows = [r for r in rows if r.market in OU_MARKETS]
    out: list[Graded] = []
    for sel in selections(rows):
        game = finals.get((sel.game_date, sel.matchup))
        if sel.side != "over" or sel.fair is None or game is None:
            continue
        hit = _hit(outcome(sel.market, sel.entity, game), sel.line)
        if hit is not None:
            out.append(Graded((sel.game_date, sel.matchup), sel.market, CONSENSUS, sel.fair, hit))
    for r in last_quotes(rows).values():
        if r.side != "over" or r.book not in books or r.opposite_american is None:
            continue
        game = finals.get((r.game_date, r.matchup))
        if game is None:
            continue
        hit = _hit(outcome(r.market, r.entity, game), r.line)
        if hit is not None:
            p = novig(r.american, r.opposite_american)
            out.append(Graded((r.game_date, r.matchup), r.market, r.book, p, hit))
    return out


def _gap(rows: list[Graded]) -> float:
    return sum(r.hit - r.p_over for r in rows) / len(rows)


def fit(graded: Iterable[Graded], *, draws: int = 400, seed: int = 1) -> list[Gap]:
    """Gap and game-clustered 95% interval per (market, book)."""
    cells: dict[tuple[str, str], list[Graded]] = defaultdict(list)
    for g in graded:
        cells[(g.market, g.book)].append(g)
    out: list[Gap] = []
    for (market, book), rows in sorted(cells.items()):
        by_game: dict[GameKey, list[Graded]] = defaultdict(list)
        for r in rows:
            by_game[r.game].append(r)
        games = sorted(by_game)
        rnd = random.Random(seed)
        sims = []
        for _ in range(draws):
            sample = [r for g in rnd.choices(games, k=len(games)) for r in by_game[g]]
            sims.append(_gap(sample))
        sims.sort()
        out.append(
            Gap(
                market=market,
                book=book,
                n=len(rows),
                games=len(games),
                p_over=sum(r.p_over for r in rows) / len(rows),
                hit_rate=sum(r.hit for r in rows) / len(rows),
                gap=_gap(rows),
                lo=sims[int(0.025 * draws)],
                hi=sims[min(draws - 1, int(0.975 * draws))],
            )
        )
    return out


def to_payload(gaps: list[Gap], **meta: object) -> dict:
    return {**meta, "method": "power", "gaps": [asdict(g) for g in gaps]}


def shifts(payload: Mapping | None, book: str = CONSENSUS) -> dict[str, float]:
    """market -> applied Over shift from a params payload (empty when never fitted)."""
    out: dict[str, float] = {}
    for raw in (payload or {}).get("gaps", []):
        if raw.get("book") == book:
            out[str(raw["market"])] = Gap(**raw).applied
    return out


def adjusted_fair(sel: Selection, shift: Mapping[str, float]) -> float | None:
    """The consensus fair price with the market's fitted Over bias taken out."""
    if sel.fair is None:
        return None
    s = shift.get(sel.market, 0.0)
    if sel.side == "over":
        return sel.fair + s
    if sel.side == "under":
        return sel.fair - s
    return sel.fair


__all__ = [
    "CONSENSUS",
    "NAME",
    "OU_MARKETS",
    "PROP_STAT",
    "Gap",
    "Graded",
    "adjusted_fair",
    "fit",
    "grade",
    "outcome",
    "shifts",
    "to_payload",
]
