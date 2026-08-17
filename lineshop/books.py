"""Which books to actually hold, measured off the board rather than reputation.

A single book's generosity is the wrong question, because what a shopper owns is
a *set*: two books that are individually strong but always best on the same
sides are one book. So the unit here is union coverage -- the share of priced
sides where somebody in the set holds the best number and price available
anywhere -- and the useful output is which two accounts add most to the ones you
already have.

Sides quoted by only a couple of books are dropped. Being the only quote on a
market is not being the best price on it, and without that filter a thin book
that hangs one lonely number outranks the market makers.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass

from lineshop.feed import Game
from lineshop.scan import H2H, offer_rank
from nfl_engine.market.odds import american_to_prob

MIN_BOOKS_PER_GAME = 5
MIN_QUOTES_PER_SIDE = 5


@dataclass(frozen=True)
class BookScore:
    book: str
    sides: int  # priced sides the book quoted
    best: int  # of those, how many it held the best offer on
    avg_cost: float  # price gap to the board's best price, when it is not best
    hold: float  # median two-sided vig on its own paired quotes

    @property
    def best_share(self) -> float:
        return self.best / self.sides if self.sides else 0.0


@dataclass(frozen=True)
class BookReport:
    sport: str
    games: int
    sides: int
    scores: tuple[BookScore, ...]
    coverage: dict[tuple[str, ...], float]  # set -> union coverage


def _offers(games: list[Game]) -> dict[tuple[str, str, str], dict[str, tuple[float, float]]]:
    """(game, market, side) -> book -> comparable (number, price) rank."""
    out: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = {}
    for game in games:
        if len(game.books) < MIN_BOOKS_PER_GAME:
            continue
        for (market, side), quotes in game.quotes.items():
            per_book: dict[str, tuple[float, float]] = {}
            for quote in quotes:
                scored = offer_rank(market, side, game, quote)
                if quote.book not in per_book or scored > per_book[quote.book]:
                    per_book[quote.book] = scored
            if len(per_book) >= MIN_QUOTES_PER_SIDE:
                out[(game.matchup, market, side)] = per_book
    return out


def _holds(games: list[Game]) -> dict[str, list[float]]:
    """Two-sided overround per book, from its own paired quotes.

    The two sides of a spread are quoted at opposite numbers (-3 and +3), so the
    pairing key takes the magnitude; totals share a number already and a
    moneyline has none.
    """
    out: dict[str, list[float]] = {}
    for game in games:
        markets: dict[tuple[str, str, float | None], list[float]] = {}
        for (market, _side), quotes in game.quotes.items():
            for quote in quotes:
                point = None if market == H2H or quote.point is None else abs(quote.point)
                key = (market, quote.book, point)
                markets.setdefault(key, []).append(american_to_prob(quote.american))
        for (_, book, _), probs in markets.items():
            if len(probs) == 2:
                out.setdefault(book, []).append(sum(probs) - 1.0)
    return out


def rank(
    sport: str, games: list[Game], *, sets_of: int = 4, fixed: tuple[str, ...] = ()
) -> BookReport:
    offers = _offers(games)
    holds = _holds(games)
    winners: dict[tuple[str, str, str], set[str]] = {}
    quoted: dict[str, int] = {}
    cost: dict[str, list[float]] = {}
    for key, per_book in offers.items():
        top = max(per_book.values())
        winners[key] = {b for b, r in per_book.items() if r == top}
        top_price = max(r[1] for r in per_book.values())
        for book, r in per_book.items():
            quoted[book] = quoted.get(book, 0) + 1
            if book not in winners[key]:
                # What the shopper pays for using this book instead of the best:
                # the extra implied probability its price demands. Price only --
                # a worse *number* shows up in the coverage share, not here.
                cost.setdefault(book, []).append(
                    american_to_prob(r[1]) - american_to_prob(top_price)
                )
    scores = tuple(
        sorted(
            (
                BookScore(
                    book=book,
                    sides=n,
                    best=sum(1 for w in winners.values() if book in w),
                    avg_cost=statistics.mean(cost.get(book, [0.0])),
                    hold=statistics.median(holds.get(book, [0.0])),
                )
                for book, n in quoted.items()
            ),
            key=lambda s: s.best_share,
            reverse=True,
        )
    )

    def coverage(books: tuple[str, ...]) -> float:
        if not winners:
            return 0.0
        chosen = set(books)
        return sum(1 for w in winners.values() if w & chosen) / len(winners)

    candidates = [s.book for s in scores]
    combos: dict[tuple[str, ...], float] = {}
    for combo in itertools.combinations(candidates, min(sets_of, len(candidates))):
        combos[tuple(sorted(combo))] = coverage(combo)
    have = tuple(b for b in fixed if b in candidates)
    if have:
        combos[have] = coverage(have)
        room = max(0, sets_of - len(have))
        others = [b for b in candidates if b not in have]
        for combo in itertools.combinations(others, min(room, len(others))):
            filled: tuple[str, ...] = tuple(sorted(have + combo))
            combos[filled] = coverage(filled)
    return BookReport(
        sport=sport,
        games=len({k[0] for k in offers}),
        sides=len(offers),
        scores=scores,
        coverage=combos,
    )
