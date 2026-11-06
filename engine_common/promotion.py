"""The bar a paper market must clear before it may be staked, shared by every engine.

A market's graded buys move it from ``paper`` to ``Lean (probation)`` at a half-unit
stake when, read as one series in date order:

* at least ``MIN_BUYS`` graded buys;
* ROI not negative, in both halves of the ledger;
* close-implied ROI lower 95% bound above zero, where close-implied ROI is
  ``closing no-vig prob of our side x decimal price taken - 1``: the prices we
  took were worth money at the close, juice included (Whelan: a little closing
  line value is not a winner; the move has to clear the hold);
* model Brier no worse than the market's on the same rows (a market with no
  model probability can never clear).

Results alone cannot confirm a +3-5% edge inside a season (1,600+ bets at 2 SE),
so this bar asks the close to vouch for the record. It labels a stake; it never
picks a bet or moves a price, and close-implied ROI is known only at tip, so it
never screens a bet before it is placed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_BUYS = 100
STAKE_UNITS = 0.5
PAPER, LEAN = "paper", "Lean (probation)"


def decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def close_roi(close_prob: float, american: float) -> float:
    """Expected return per unit of the price taken, under the closing no-vig probability."""
    return close_prob * decimal(american) - 1.0


@dataclass(frozen=True)
class Promotion:
    market: str
    n: int
    roi: float
    halves: tuple[float, float]  # ROI, older then newer half
    close_n: int  # buys with a timely close
    close_roi: float  # mean close-implied ROI
    close_roi_lo: float  # lower 95% bound on the mean close-implied ROI
    brier_model: float | None
    brier_market: float | None

    @property
    def cleared(self) -> bool:
        return (
            self.n >= MIN_BUYS
            and self.roi >= 0
            and min(self.halves) >= 0
            and self.close_n > 1
            and self.close_roi_lo > 0
            and self.brier_model is not None
            and self.brier_market is not None
            and self.brier_model <= self.brier_market
        )

    @property
    def needed(self) -> int:
        return max(0, MIN_BUYS - self.n)

    @property
    def status(self) -> str:
        return LEAN if self.cleared else PAPER

    @property
    def stake(self) -> float:
        return STAKE_UNITS if self.cleared else 0.0

    @property
    def why_not(self) -> str:
        """What still stands between the market and the bar; empty when cleared."""
        if self.cleared:
            return ""
        out = []
        if self.n < MIN_BUYS:
            out.append(f"{self.needed} more buys")
        if self.roi < 0 or min(self.halves) < 0:
            out.append("ROI")
        if self.close_n < 2 or self.close_roi_lo <= 0:
            out.append("close ROI")
        if self.brier_model is None or self.brier_market is None:
            out.append("no model")
        elif self.brier_model > self.brier_market:
            out.append("Brier")
        return ", ".join(out)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def read(
    market: str,
    pnls: list[float],
    close_rois: list[float],
    brier_model: float | None,
    brier_market: float | None,
) -> Promotion:
    """``pnls`` in date order (one per graded buy); ``close_rois`` for the buys with a timely close."""
    h = len(pnls) // 2
    n = len(close_rois)
    m = _mean(close_rois)
    se = math.sqrt(sum((c - m) ** 2 for c in close_rois) / (n - 1) / n) if n > 1 else float("inf")
    return Promotion(
        market,
        len(pnls),
        _mean(pnls),
        (_mean(pnls[:h]), _mean(pnls[h:])),
        n,
        m,
        m - 1.96 * se if n > 1 else float("-inf"),
        brier_model,
        brier_market,
    )


BAR_TEXT = "ROI ≥ 0 in both halves, close-implied ROI lower bound > 0, Brier ≤ market"

__all__ = [
    "BAR_TEXT",
    "LEAN",
    "MIN_BUYS",
    "PAPER",
    "STAKE_UNITS",
    "Promotion",
    "close_roi",
    "decimal",
    "read",
]
