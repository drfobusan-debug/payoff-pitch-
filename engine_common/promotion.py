"""The bar a paper market must clear before it may be staked, shared by every engine.

A market's graded buys move it from ``paper`` to ``Lean (probation)`` at a half-unit
stake when, read as one series in date order:

* at least ``MIN_BUYS`` graded buys;
* ROI not negative, in both halves of the ledger;
* CLV lower 95% bound above zero (the close has moved toward the bets);
* model Brier no worse than the market's on the same rows (a market with no
  model probability can never clear).

Results alone cannot confirm a +3-5% edge inside a season (1,600+ bets at 2 SE),
so this bar asks the close to vouch for the record. It labels a stake; it never
picks a bet or moves a price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_BUYS = 100
STAKE_UNITS = 0.5
PAPER, LEAN = "paper", "Lean (probation)"


@dataclass(frozen=True)
class Promotion:
    market: str
    n: int
    roi: float
    halves: tuple[float, float]  # ROI, older then newer half
    clv_n: int
    clv_mean: float
    clv_lo: float  # lower 95% bound on the mean CLV
    brier_model: float | None
    brier_market: float | None

    @property
    def cleared(self) -> bool:
        return (
            self.n >= MIN_BUYS
            and self.roi >= 0
            and min(self.halves) >= 0
            and self.clv_n > 1
            and self.clv_lo > 0
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
        if self.clv_n < 2 or self.clv_lo <= 0:
            out.append("CLV")
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
    clvs: list[float],
    brier_model: float | None,
    brier_market: float | None,
) -> Promotion:
    """``pnls`` in date order (one per graded buy); ``clvs`` for the buys with a timely close."""
    h = len(pnls) // 2
    n = len(clvs)
    m = _mean(clvs)
    se = math.sqrt(sum((c - m) ** 2 for c in clvs) / (n - 1) / n) if n > 1 else float("inf")
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


__all__ = ["LEAN", "MIN_BUYS", "PAPER", "STAKE_UNITS", "Promotion", "read"]
