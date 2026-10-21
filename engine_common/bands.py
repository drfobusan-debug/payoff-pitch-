"""One graded-only band's read: flat 1u on every bet in a slice of the board.

Shared by every engine's audit. Rows on one game move together, so the SE and
the halves are taken over per-game sums. A band earns a call only at >=
``MIN_N`` games: "wins" needs units above ``CALL_SE`` standard errors with both
halves of the ledger (date order) positive; "loses" is the mirror. Many bands
are read at once, so about one in twenty sits past 2 SE by chance.
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
        if self.units > CALL_SE * self.se and min(self.halves) > 0:
            return "wins: clears the bar"
        if -self.units > CALL_SE * self.se and max(self.halves) < 0:
            return "loses: refusal candidate"
        return "no edge"


__all__ = ["CALL_SE", "MIN_N", "BandRead"]
