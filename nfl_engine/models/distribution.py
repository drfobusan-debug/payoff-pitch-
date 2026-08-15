"""The simulated score distribution, and the three markets read off it.

Every game price in the engine comes from one object: a pair of simulated score
arrays. Moneyline, spread and total are then the same distribution asked three
questions, which is what keeps them coherent -- a 3-point favourite cannot be
40% to win outright and 60% to cover -3 if both numbers come from here.

Pushes are returned, never absorbed. A bet on -3 that lands on 3 is refunded, so
its value is ``P(cover) / (1 - P(push))``: the 9.0% push rate on a 3-point spread
(n=1,153, 1999-2025) is the single largest reason NFL half-points are worth
shopping, and a model that folds pushes into losses prices -3 and -2.5 alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketProb:
    """Probability of a side winning, and of the line pushing.

    ``conditional`` is what an EV calculation wants: the chance of winning given
    the bet resolves at all.
    """

    win: float
    push: float

    @property
    def conditional(self) -> float:
        live = 1.0 - self.push
        return self.win / live if live > 1e-12 else 0.0


@dataclass(frozen=True)
class ScoreDistribution:
    """Simulated home and away scores, one entry per trial."""

    home: np.ndarray
    away: np.ndarray

    def __post_init__(self) -> None:
        if self.home.shape != self.away.shape:
            raise ValueError("home and away score arrays must be the same length")

    @property
    def n(self) -> int:
        return int(self.home.size)

    def margins(self) -> np.ndarray:
        """Home score minus away score."""
        return self.home - self.away

    def totals(self) -> np.ndarray:
        return self.home + self.away

    def mean_margin(self) -> float:
        return float(np.mean(self.margins()))

    def mean_total(self) -> float:
        return float(np.mean(self.totals()))

    # -- markets ----------------------------------------------------------
    def moneyline(self, *, home: bool) -> MarketProb:
        """Win probability for one side; a tie is the push."""
        margin = self.margins()
        tie = float(np.mean(margin == 0))
        wins = margin > 0 if home else margin < 0
        return MarketProb(win=float(np.mean(wins)), push=tie)

    def spread(self, home_point: float) -> MarketProb:
        """Cover probability for the *home* side laying/taking ``home_point``.

        ``home_point`` is the home team's handicap: -3.0 means the home team must
        win by more than 3. Pass the negated point for the away side.
        """
        adjusted = self.margins() + home_point
        return MarketProb(
            win=float(np.mean(adjusted > 0)),
            push=float(np.mean(adjusted == 0)),
        )

    def total(self, line: float, *, over: bool) -> MarketProb:
        totals = self.totals()
        wins = totals > line if over else totals < line
        return MarketProb(win=float(np.mean(wins)), push=float(np.mean(totals == line)))

    # -- diagnostics ------------------------------------------------------
    def margin_frequency(self, value: int) -> float:
        """Share of trials landing on exactly ``value`` points of margin.

        The validation hook for the key numbers: 2015-2025 actuals are 14.8% at
        a 3-point margin, 8.7% at 7 and 6.9% at 6.
        """
        return float(np.mean(np.abs(self.margins()) == value))
