"""Ratings to expected points, and then the market's veto.

Turns a :class:`~nfl_engine.features.ratings.RatingBook` into the mean the phase-2
score distribution shapes: a margin, a total, and a drive count per side.

    margin = HOME_EDGE + MARGIN_OFF * (off_epa_h - off_epa_a)
                       + MARGIN_DEF * (def_epa_h - def_epa_a)
    total  = TOTAL_BASE + TOTAL_OFF * (off_epa_h + off_epa_a)
                        + TOTAL_DEF * (def_epa_h + def_epa_a)

**Success rate and PROE are in the rating book but not in this model**, which
reverses what the reliability table implied. Success rate allowed is indeed the
more repeatable defensive measurement (r=0.404 against 0.349 for EPA allowed),
but repeatable is not the same as incremental. Fitting the points model on the
walk-forward panel and scoring it out of sample by season, margin MAE is 10.329
on the EPA terms alone and 10.328 with both success terms added, and the fitted
defensive success coefficient comes out with the *wrong sign* (t = +1.30) once
EPA is in, which is collinearity rather than football. Pace, PROE and drive
counts are the same story on the total -- individually significant in-sample
(t = -2.63, +2.27, +2.50) and worth nothing out of it (MAE 10.720 against 10.752
with them). All of them stay in the rating book, because the props layer needs
pace and PROE to condition a game script; none of them moves a price.

The blend, and why it is not 0.55
---------------------------------
The plan said the ratings-implied margin would be pulled 55% of the way to the
market. Measured by ``scripts/nfl/ratings_study.py`` over 3,450 games, 2013-2025,
walk-forward:

    weight on market   0.00    0.25    0.55    0.75    0.90    1.00
    margin MAE        10.282  10.116   9.978   9.925   9.908   9.905

The curve is monotone all the way to the market. There is no interior optimum,
which is the arithmetic statement of the same fact the residual test gives
directly: the rating's disagreement with the closing line explains none of that
line's error (slope +0.016, t = +0.25). So ``MARKET_WEIGHT`` ships at 1.0 -- when
there is a market, the market is the mean -- and the ratings-implied number is
carried alongside as a *reported* disagreement, exactly like the MLB regression
arrows, with the situational block the only thing allowed to move the total away
from it.

The ratings still have two jobs. They are the mean when there is no market yet
(an early-week look, or a book that has not posted), and they are what conditions
game script for the props layer, where the market does not publish an answer.
Priced through the possession simulator they are a *well calibrated* forecast --
Brier 0.2221 against 0.2470 for always predicting the home team, and the deciles
above 0.5 land within 1.6pp of their prediction -- which is precisely the trap
this constant exists to close: being right about football is not the same as
being right about the price.
"""

from __future__ import annotations

from dataclasses import dataclass

from nfl_engine.features.adjustments import Adjustment, Situation, adjust
from nfl_engine.features.ratings import RatingBook
from nfl_engine.models.drives import LG_DRIVES, ExpectedGame

# Fitted by scripts/nfl/ratings_study.py --fit on 3,450 games, 2013-2025, every
# rating from prior weeks only (t = +19.7, -10.5, +11.5, +5.4).
HOME_EDGE = 2.04
MARGIN_OFF_EPA = 139.7
MARGIN_DEF_EPA = -95.7
TOTAL_BASE = 45.6
TOTAL_OFF_EPA = 82.1
TOTAL_DEF_EPA = 51.8

# 1.0 = the market is the mean. See the module docstring: every weight below this
# measured worse, monotonically.
MARKET_WEIGHT = 1.0


@dataclass(frozen=True)
class Forecast:
    """The engine's mean for a game, and the rating's opinion beside it."""

    home_points: float
    away_points: float
    home_drives: float
    away_drives: float
    rating_margin: float
    rating_total: float
    market_margin: float | None
    market_total: float | None
    adjustment: Adjustment

    def margin(self) -> float:
        return self.home_points - self.away_points

    def total(self) -> float:
        return self.home_points + self.away_points

    def rating_edge_margin(self) -> float | None:
        """How far the rating disagrees with the line. Reported, never priced."""
        if self.market_margin is None:
            return None
        return self.rating_margin - self.market_margin

    def rating_edge_total(self) -> float | None:
        if self.market_total is None:
            return None
        return self.rating_total - self.market_total

    def expected_game(self) -> ExpectedGame:
        return ExpectedGame(
            home_points=self.home_points,
            away_points=self.away_points,
            home_drives=self.home_drives,
            away_drives=self.away_drives,
        )


def rating_margin(book: RatingBook, home: str, away: str) -> float:
    """The ratings-implied home margin, before the market sees it."""
    h, a = book.rating(home), book.rating(away)
    return (
        HOME_EDGE
        + MARGIN_OFF_EPA * (h.off_epa - a.off_epa)
        + MARGIN_DEF_EPA * (h.def_epa - a.def_epa)
    )


def rating_total(book: RatingBook, home: str, away: str) -> float:
    h, a = book.rating(home), book.rating(away)
    return (
        TOTAL_BASE
        + TOTAL_OFF_EPA * (h.off_epa + a.off_epa)
        + TOTAL_DEF_EPA * (h.def_epa + a.def_epa)
    )


def drive_counts(book: RatingBook, home: str, away: str) -> tuple[float, float]:
    """Possessions per side, which is what gives the total its own variance.

    A drive rating is a delta from the league's own count, so a team with no
    history gets the league number rather than zero drives.
    """
    league = book.league.get("drives", LG_DRIVES)
    base = league if league > 0 else LG_DRIVES
    h, a = book.rating(home), book.rating(away)
    return (
        max(base + h.off_drives, 1.0),
        max(base + a.off_drives, 1.0),
    )


def forecast(
    book: RatingBook,
    home: str,
    away: str,
    *,
    situation: Situation | None = None,
    market_margin: float | None = None,
    market_total: float | None = None,
    market_weight: float = MARKET_WEIGHT,
) -> Forecast:
    """Blend the rating with the market, then apply the situational block.

    ``market_margin`` is the home team's expected margin (so a 3-point home
    favourite is +3.0, the sign ``games.csv`` already uses), and a missing market
    falls back to the rating alone.
    """
    r_margin = rating_margin(book, home, away)
    r_total = rating_total(book, home, away)
    weight = market_weight if book.is_usable() else 1.0
    margin = r_margin if market_margin is None else (
        weight * market_margin + (1.0 - weight) * r_margin
    )
    total = r_total if market_total is None else (
        weight * market_total + (1.0 - weight) * r_total
    )
    situation = situation or Situation()
    delta = adjust(situation)
    total += delta.total_points
    margin += delta.margin_points
    home_drives, away_drives = drive_counts(book, home, away)
    return Forecast(
        home_points=(total + margin) / 2.0,
        away_points=(total - margin) / 2.0,
        home_drives=home_drives,
        away_drives=away_drives,
        rating_margin=r_margin,
        rating_total=r_total,
        market_margin=market_margin,
        market_total=market_total,
        adjustment=delta,
    )
