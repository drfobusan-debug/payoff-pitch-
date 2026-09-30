"""Roll a pregame margin/total prior forward through the live score and clock.

The remaining margin from the home side's view is modelled as

    R ~ N(a_q * f * mu + b_q * f * surprise + c_q * EP,  sd(f)^2)

where ``f`` is the fraction of regulation left, ``mu`` the pregame expected
margin, ``surprise = current margin - (1 - f) * mu`` how far ahead of the line the
score already is and ``EP`` the possessing side's field-position expected points.
The final margin is ``current margin + R``.

The quarter coefficients and the ``sd(f)`` curve were fitted on 2024 FBS
play-by-play against closing consensus lines and checked out-of-sample on 2025
and 2026 (see docs/live_edge.md). Two things they say: first-half points carry no
information about team strength beyond the points themselves (b ~ 0), and the
leader coasts in the second half (b < 0, a < 1). The NFL curve is the CFB shape
scaled to the NFL pregame dispersion until an NFL fit replaces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

REGULATION = 3600
QUARTER = 900


@dataclass(frozen=True)
class Prior:
    """Pregame expectation, home minus away."""

    margin: float
    total: float
    margin_sd: float
    total_sd: float


@dataclass(frozen=True)
class LiveState:
    home_score: int
    away_score: int
    period: int
    clock_secs: int  # seconds left in the current period
    possession_home: bool | None = None
    yards_to_goal: int | None = None

    @property
    def seconds_left(self) -> int:
        if self.period > 4:
            return 0
        return (4 - self.period) * QUARTER + self.clock_secs

    @property
    def fraction_left(self) -> float:
        return max(0.0, min(1.0, self.seconds_left / REGULATION))

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    @property
    def total(self) -> int:
        return self.home_score + self.away_score


@dataclass(frozen=True)
class LiveDist:
    margin_mu: float
    margin_sd: float
    total_mu: float
    total_sd: float

    def p_home_win(self) -> float:
        return float(1.0 - norm.cdf(0.0, self.margin_mu, self.margin_sd))

    def p_home_cover(self, home_line: float) -> float:
        """P(home margin + home_line > 0); ``home_line`` is the book's home spread."""
        return float(1.0 - norm.cdf(-home_line, self.margin_mu, self.margin_sd))

    def p_over(self, total_line: float) -> float:
        return float(1.0 - norm.cdf(total_line, self.total_mu, self.total_sd))


# (a, b, c) per quarter: line carry, surprise carry, EP carry.
CFB_COEFS: dict[int, tuple[float, float, float]] = {
    1: (1.032, 0.071, 0.601),
    2: (1.016, -0.019, 0.445),
    3: (0.903, -0.213, 0.511),
    4: (0.764, -0.538, 0.419),
}
CFB_TOTAL_COEFS = (1.059, 0.088)
# Residual SD of the remaining margin by fraction of regulation left.
CFB_SD_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 3.0),
    (0.1, 5.60),
    (0.2, 7.03),
    (0.3, 8.27),
    (0.4, 9.69),
    (0.5, 10.89),
    (0.6, 11.91),
    (0.7, 12.70),
    (0.8, 13.50),
    (0.9, 14.50),
    (1.0, 15.41),
)
CFB_PREGAME_SD = 16.0
NFL_PREGAME_SD = 13.2
SD_FLOOR = 1.0
FINAL_SD = 1e-3


def expected_points(possession_home: bool | None, yards_to_goal: int | None) -> float:
    """Crude EP of the side with the ball, signed from the home view."""
    if possession_home is None or yards_to_goal is None or not 0 < yards_to_goal <= 100:
        return 0.0
    v = 6.3 - 0.072 * yards_to_goal
    return v if possession_home else -v


def _interp(curve: tuple[tuple[float, float], ...], f: float) -> float:
    if f <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:], strict=False):
        if f <= x1:
            return y0 + (y1 - y0) * (f - x0) / (x1 - x0)
    return curve[-1][1]


class Roller:
    def __init__(self, sport: str) -> None:
        self.sport = sport
        # NFL games are tighter than FBS ones; scale the fitted curve by the
        # ratio of pregame dispersions until an NFL play-by-play fit exists.
        self.scale = 1.0 if sport == "cfb" else NFL_PREGAME_SD / CFB_PREGAME_SD

    def margin_sd(self, f: float, prior_sd: float) -> float:
        fitted = _interp(CFB_SD_CURVE, f) * self.scale
        # Respect a prior that is wider than the league curve (a thin market).
        return max(SD_FLOOR, fitted * max(1.0, prior_sd / (CFB_PREGAME_SD * self.scale)))

    def roll(self, prior: Prior, state: LiveState) -> LiveDist:
        f = state.fraction_left
        if f <= 0.0:
            return LiveDist(state.margin, FINAL_SD, state.total, FINAL_SD)
        q = max(1, min(4, state.period))
        a, b, c = CFB_COEFS[q]
        surprise = state.margin - (1.0 - f) * prior.margin
        ep = expected_points(state.possession_home, state.yards_to_goal)
        rem_margin = a * f * prior.margin + b * f * surprise + c * ep
        at, bt = CFB_TOTAL_COEFS
        pace = state.total - (1.0 - f) * prior.total
        rem_total = at * f * prior.total + bt * f * pace
        return LiveDist(
            margin_mu=state.margin + rem_margin,
            margin_sd=self.margin_sd(f, prior.margin_sd),
            total_mu=state.total + rem_total,
            total_sd=max(SD_FLOOR, prior.total_sd * math.sqrt(f) + SD_FLOOR),
        )
