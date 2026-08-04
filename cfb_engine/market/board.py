"""Structured per-game odds: the multi-book prices for one game's three markets,
organized so the pipeline can pull a side's quotes and the market's consensus
line without re-parsing selection strings.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from cfb_engine.market.ev import MarketQuote


@dataclass
class GameOdds:
    matchup: str
    # side abbrev -> quotes
    ml: dict[str, list[MarketQuote]] = field(default_factory=lambda: defaultdict(list))
    # home spread point -> {side abbrev -> quotes}
    spreads: dict[float, dict[str, list[MarketQuote]]] = field(default_factory=dict)
    # total line -> {"over"|"under" -> quotes}
    totals: dict[float, dict[str, list[MarketQuote]]] = field(default_factory=dict)

    def add_ml(self, side: str, quote: MarketQuote) -> None:
        self.ml[side].append(quote)

    def add_spread(self, home_point: float, side: str, quote: MarketQuote) -> None:
        self.spreads.setdefault(home_point, defaultdict(list))[side].append(quote)

    def add_total(self, line: float, over: bool, quote: MarketQuote) -> None:
        self.totals.setdefault(line, defaultdict(list))["over" if over else "under"].append(quote)

    # -- consensus selectors ---------------------------------------------
    def main_spread(self) -> float | None:
        """The most-quoted home spread (ties broken toward the line nearest 0)."""
        if not self.spreads:
            return None
        return _mode_line({pt: _count(sides) for pt, sides in self.spreads.items()})

    def main_total(self) -> float | None:
        if not self.totals:
            return None
        return _mode_line({ln: _count(sides) for ln, sides in self.totals.items()})

    def consensus_home_spread(self) -> float | None:
        """Book-count-weighted median home spread across all quoted lines."""
        points: list[float] = []
        for pt, sides in self.spreads.items():
            points.extend([pt] * _count(sides))
        return statistics.median(points) if points else None

    def consensus_total(self) -> float | None:
        lines: list[float] = []
        for ln, sides in self.totals.items():
            lines.extend([ln] * _count(sides))
        return statistics.median(lines) if lines else None


def _count(sides: dict[str, list[MarketQuote]]) -> int:
    return sum(len(v) for v in sides.values())


def _mode_line(counts: dict[float, int]) -> float:
    best = max(counts.values())
    tied = [ln for ln, c in counts.items() if c == best]
    return min(tied, key=abs)
