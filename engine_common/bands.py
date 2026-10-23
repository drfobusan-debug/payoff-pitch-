"""One graded-only band's read: flat 1u on every bet in a slice of the board.

Shared by every engine's audit. Rows on one game move together, so the SE and
the halves are taken over per-game sums. Units are judged against what the
market itself expects of the same bets at the card's price (``expected``: the
no-vig consensus applied to the payout, so a band of heavy favourites expects
to lose the vig); a band earns a call only at >= ``MIN_N`` games: "wins" needs
the excess above ``CALL_SE`` standard errors with both halves of the ledger
(date order) positive; "loses" is the mirror. Many bands are read at once, so
about one in twenty sits past 2 SE by chance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MIN_N = 100
CALL_SE = 2.0


@dataclass
class BandRead:
    name: str
    pnls: list[float] = field(default_factory=list)  # one per bet, date order
    games: list[float] = field(default_factory=list)  # P&L summed per game, date order
    expected: list[float] = field(default_factory=list)  # market-expected P&L per bet, if priced

    @property
    def _by_game(self) -> list[float]:
        return self.games or self.pnls

    @property
    def n(self) -> int:
        return len(self.pnls)

    @property
    def units(self) -> float:
        return sum(self.pnls)

    @property
    def roi(self) -> float:
        return self.units / self.n if self.n else 0.0

    @property
    def se(self) -> float:
        """SE of the units total, over per-game sums."""
        g = self._by_game
        n = len(g)
        if n < 2:
            return float("nan")
        m = sum(g) / n
        return math.sqrt(sum((p - m) ** 2 for p in g) / (n - 1) * n)

    @property
    def market_units(self) -> float:
        """What the market expects these bets to make at the card's price."""
        return sum(self.expected)

    @property
    def excess(self) -> float:
        return self.units - self.market_units

    @property
    def halves(self) -> tuple[float, float]:
        g = self._by_game
        h = len(g) // 2
        return sum(g[:h]), sum(g[h:])

    @property
    def record(self) -> str:
        return f"{sum(p > 0 for p in self.pnls)}-{sum(p < 0 for p in self.pnls)}"

    @property
    def verdict(self) -> str:
        games = len(self._by_game)
        if games < MIN_N:
            return f"probation ({games}/{MIN_N} games)"
        h1, h2 = self.halves
        e1, e2 = self._expected_halves
        if self.excess > CALL_SE * self.se and min(h1 - e1, h2 - e2) > 0:
            return "wins: clears the bar"
        if -self.excess > CALL_SE * self.se and max(h1 - e1, h2 - e2) < 0:
            return "loses: refusal candidate"
        return "no edge"

    @property
    def _expected_halves(self) -> tuple[float, float]:
        """Market-expected units in each half; zero when the band carries no prices."""
        if not self.expected or len(self.expected) != len(self.pnls) or self.games:
            if self.games and self.expected:
                # Per-game halves: spread the expectation in bet order over the game split.
                h = len(self.games) // 2
                share = h / len(self.games) if self.games else 0.5
                return self.market_units * share, self.market_units * (1 - share)
            return 0.0, 0.0
        h = len(self.expected) // 2
        return sum(self.expected[:h]), sum(self.expected[h:])


__all__ = ["CALL_SE", "MIN_N", "BandRead"]
