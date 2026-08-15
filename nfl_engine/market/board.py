"""Multi-book prices for one game's three markets.

The NFL board is stored line by line rather than collapsed to a consensus,
because in this sport the *ladder* is the product. Books hang -2.5, -3 and -3.5
on the same game, and 10.2% of games with a 3-point spread land exactly on it
(n=433, 2015-2025), so which rung a bet is struck on is worth more than most
ratings disagreements. Nothing here forms a price; it keeps every quote
addressable so the EV layer can shop the ladder and the audit can grade the rung
that was actually taken.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from nfl_engine.market.odds import american_to_decimal, american_to_prob

# Pinnacle is the sharpest book The Odds API carries; its number counts for more
# when forming the fair price. Unlisted books default to 1.0.
BOOK_WEIGHTS = {"pinnacle": 2.0, "lowvig": 1.5, "betonlineag": 1.2}

OVER, UNDER = "over", "under"


@dataclass(frozen=True)
class MarketQuote:
    book: str
    american: float
    # The same book's price on the other side of the same line. Required to strip
    # the vig, and never fabricated: an unpaired quote is reported as unpaired
    # rather than assumed to be -110, which is the mistake the MLB engine paid
    # for on total bases.
    opposite_american: float | None = None

    @property
    def decimal(self) -> float:
        return american_to_decimal(self.american)

    @property
    def paired(self) -> bool:
        return self.opposite_american is not None

    @property
    def no_vig_prob(self) -> float:
        p = american_to_prob(self.american)
        if self.opposite_american is None:
            return p
        q = american_to_prob(self.opposite_american)
        total = p + q
        return p / total if total > 0 else p


@dataclass
class GameOdds:
    """Every quote on one game, indexed by market and line."""

    matchup: str
    ml: dict[str, list[MarketQuote]] = field(default_factory=lambda: defaultdict(list))
    # home handicap -> side abbrev -> quotes. Spreads are always stored on the
    # home axis so the two sides of a line pair up for de-vigging.
    spreads: dict[float, dict[str, list[MarketQuote]]] = field(default_factory=dict)
    totals: dict[float, dict[str, list[MarketQuote]]] = field(default_factory=dict)

    def add_ml(self, side: str, quote: MarketQuote) -> None:
        self.ml[side].append(quote)

    def add_spread(self, home_point: float, side: str, quote: MarketQuote) -> None:
        self.spreads.setdefault(home_point, defaultdict(list))[side].append(quote)

    def add_total(self, line: float, over: bool, quote: MarketQuote) -> None:
        self.totals.setdefault(line, defaultdict(list))[OVER if over else UNDER].append(quote)

    # -- selectors --------------------------------------------------------
    def main_spread(self) -> float | None:
        """The most-quoted home handicap, ties broken toward the shorter number."""
        if not self.spreads:
            return None
        return _mode_line({point: _count(sides) for point, sides in self.spreads.items()})

    def main_total(self) -> float | None:
        if not self.totals:
            return None
        return _mode_line({line: _count(sides) for line, sides in self.totals.items()})

    def consensus_home_spread(self) -> float | None:
        """Quote-weighted median home handicap."""
        return _weighted_median(self.spreads)

    def consensus_total(self) -> float | None:
        return _weighted_median(self.totals)

    def spread_ladder(self) -> list[float]:
        """Every quoted home handicap, shortest first -- the shopping ladder."""
        return sorted(self.spreads, key=abs)

    def total_ladder(self) -> list[float]:
        return sorted(self.totals)

    def best_spread(self, side: str) -> tuple[float, MarketQuote] | None:
        """The longest handicap available to ``side``, and the best price on it.

        Choosing by point first and price second is deliberate: crossing a key
        number is worth several cents of vig, and this is the only market where
        that trade is on the shelf.
        """
        best: tuple[float, MarketQuote] | None = None
        for home_point, sides in self.spreads.items():
            quotes = sides.get(side)
            if not quotes:
                continue
            # A side's own handicap: the home axis flips for the away team.
            own_point = home_point if side == self._home_side() else -home_point
            top = max(quotes, key=lambda q: q.decimal)
            if best is None or (own_point, top.decimal) > (best[0], best[1].decimal):
                best = (own_point, top)
        return best

    def _home_side(self) -> str:
        # matchup is "AWAY @ HOME"
        return self.matchup.split(" @ ")[-1]


def _count(sides: dict[str, list[MarketQuote]]) -> int:
    return sum(len(quotes) for quotes in sides.values())


def _mode_line(counts: dict[float, int]) -> float:
    best = max(counts.values())
    tied = [line for line, count in counts.items() if count == best]
    return min(tied, key=abs)


def _weighted_median(book: dict[float, dict[str, list[MarketQuote]]]) -> float | None:
    values: list[float] = []
    for line, sides in book.items():
        values.extend([line] * _count(sides))
    return statistics.median(values) if values else None
