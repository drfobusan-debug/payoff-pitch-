"""One player's single-game distribution for the markets a book quotes on him.

A prop is a count or a yardage total, not a score margin, so it gets its own
distribution rather than a rung read off the game simulator. Two decisions, both
measured by :mod:`scripts.nfl.props_study` on nflverse weekly lines (2016-2025,
fitted on 2016-2021 and scored on 2022-2025):

**The mean is the player's own usage, shrunk.** A player's mean over his prior
games this season, shrunk toward his prior-season mean (or, for a player without
one, his position's) by ``SHRINK`` pseudo-games. Out of time, over the usage floor,
this reads MAE 2.11 targets against 2.55 for the position mean, 4.24 carries
against 5.22, and 25.5 receiving yards against 28.2 -- so usage is projectable,
which is the *only* thing that table establishes.

**The spread grows with the mean.** Residual sd is close to linear in the
projection, so it is fitted that way per market rather than assumed Poisson:
targets sd = 1.33 + 0.271*mean, carries 0.94 + 0.438*mean, receiving yards
11.70 + 0.440*mean. Counts are then negative binomial matched to that (mean,
variance) pair -- Poisson is far too tight, since a running back's carries depend
on a game script that is itself random -- and yardage is lognormal, which is
positive and right-skewed the way a yardage total is.

What the same study says about *edge*, which is a different claim and the reason
:mod:`nfl_engine.props` ships this as research. Inside the usage range where a
book actually posts a line, scored on 2022-2025 at pseudo-lines placed on our own
projection (Brier against the base rate of the same rows):

    receptions        0.2416  base 0.2444   beats the base rate
    targets           0.2448  base 0.2466   tie
    carries           0.2484  base 0.2486   tie
    receiving yards   0.2452  base 0.2430   worse
    rushing yards     0.2463  base 0.2442   worse
    pass attempts     0.2537  base 0.2500   worse
    completions       0.2551  base 0.2498   worse
    passing yards     0.2575  base 0.2500   worse

So the yardage and quarterback markets are retired on measured evidence, and even
the one market that clears the bar clears it by 0.0028 Brier -- and against a line
drawn on our own projection, which is the friendliest test that exists. The number
a book would have hung is nowhere in this table, because no archive of prop closes
exists to put it there. That is what the forward archive is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nfl_engine.models.distribution import MarketProb

TARGETS = "targets"
RECEPTIONS = "receptions"
CARRIES = "carries"
ATTEMPTS = "attempts"
COMPLETIONS = "completions"
RECEIVING_YARDS = "receiving_yards"
RUSHING_YARDS = "rushing_yards"
PASSING_YARDS = "passing_yards"

COUNT_STATS = (TARGETS, RECEPTIONS, CARRIES, ATTEMPTS, COMPLETIONS)
YARD_STATS = (RECEIVING_YARDS, RUSHING_YARDS, PASSING_YARDS)
STATS = COUNT_STATS + YARD_STATS

# Pseudo-games of the shrinkage anchor mixed into the in-season mean, and the
# number of games this season before anything is projected at all. Both fitted on
# 2016-2021: below four games the in-season mean is mostly the anchor anyway, and
# a heavier shrink flattens the spread between roles the market prices.
SHRINK = 3.0
MIN_GAMES = 4


@dataclass(frozen=True)
class Spread:
    """Residual sd as a line in the projection: sd = ``a`` + ``b`` * mean."""

    a: float
    b: float

    def sd(self, mean: float) -> float:
        return max(self.a + self.b * mean, 1e-3)


# Fitted on 2016-2021 only, by quantile bin of the projection. The quarterback
# rows carry a negative slope, which is a symptom rather than a spread: their
# residual is essentially constant because the projection barely moves.
SPREADS = {
    TARGETS: Spread(1.33, 0.271),
    RECEPTIONS: Spread(1.02, 0.311),
    CARRIES: Spread(0.94, 0.438),
    ATTEMPTS: Spread(15.99, -0.198),
    COMPLETIONS: Spread(9.37, -0.133),
    RECEIVING_YARDS: Spread(11.70, 0.440),
    RUSHING_YARDS: Spread(5.47, 0.612),
    PASSING_YARDS: Spread(100.31, -0.088),
}

# Below this projection the player is not in the market: books post lines on
# roles, and a projection under the floor is a bench week being priced as a role.
# The floors are where the holdout sample stops thinning out, not opinions.
USAGE_FLOOR = {
    TARGETS: 2.5,
    RECEPTIONS: 2.0,
    CARRIES: 5.0,
    ATTEMPTS: 15.0,
    COMPLETIONS: 10.0,
    RECEIVING_YARDS: 25.0,
    RUSHING_YARDS: 20.0,
    PASSING_YARDS: 150.0,
}


@dataclass(frozen=True)
class Projection:
    """What one player is expected to do in one game, and how firm that is."""

    player: str
    player_id: str
    position: str
    team: str
    stat: str
    games: int  # prior games this season the mean was taken over
    mean: float
    prior_mean: float | None  # the shrinkage anchor, for audit

    @property
    def sd(self) -> float:
        return sd_for(self.stat, self.mean)

    def clears_floor(self) -> bool:
        return self.mean >= USAGE_FLOOR.get(self.stat, 0.0)


def sd_for(stat: str, mean: float) -> float:
    spread = SPREADS.get(stat)
    return max(mean, 1e-3) if spread is None else spread.sd(mean)


def shrunk_mean(
    prior_sum: float, prior_games: int, anchor: float, *, shrink: float = SHRINK
) -> float:
    """In-season mean pulled toward ``anchor`` by ``shrink`` pseudo-games."""
    return (prior_sum + shrink * anchor) / (prior_games + shrink)


def _nbinom_params(mean: float, sd: float) -> tuple[float, float]:
    """(r, p) for a negative binomial matched to ``mean`` and ``sd``.

    Variance below the mean is impossible for a negative binomial, so an
    over-tight fitted spread falls back to Poisson rather than to nonsense.
    """
    mean = max(mean, 1e-6)
    var = max(sd * sd, mean * (1.0 + 1e-6))
    r = mean * mean / (var - mean)
    return r, r / (r + mean)


def _nbinom_pmf(k: int, r: float, p: float) -> float:
    if k < 0:
        return 0.0
    log_pmf = (
        math.lgamma(k + r)
        - math.lgamma(r)
        - math.lgamma(k + 1)
        + r * math.log(p)
        + k * math.log1p(-p)
    )
    return math.exp(log_pmf)


def count_prob(mean: float, sd: float, line: float) -> MarketProb:
    """P(count > line) and the push weight, for a negative binomial count.

    A prop line on an integer pushes on that exact count -- books do hang
    ``receptions 4`` -- so the push is returned rather than folded into the loss,
    the same convention the game markets use.
    """
    r, p = _nbinom_params(mean, sd)
    ceiling = int(math.floor(line))
    below = 0.0
    for k in range(0, max(ceiling, 0) + 1):
        below += _nbinom_pmf(k, r, p)
    push = _nbinom_pmf(ceiling, r, p) if float(line).is_integer() and line >= 0 else 0.0
    over = max(1.0 - below, 0.0)
    return MarketProb(win=min(over, 1.0), push=min(push, 1.0))


def yards_prob(mean: float, sd: float, line: float) -> MarketProb:
    """P(yards > line) for a lognormal matched to ``mean`` and ``sd``.

    Lognormal rather than normal because a yardage total is positive and its
    upside tail is long: a receiver projected for 45 yards has a real chance of
    120 and none at all of -20.
    """
    mean = max(mean, 1e-6)
    sd = max(sd, 1e-6)
    sigma_sq = math.log1p((sd / mean) ** 2)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - sigma_sq / 2.0
    if line <= 0.0:
        return MarketProb(win=1.0, push=0.0)
    z = (math.log(line) - mu) / sigma
    over = 0.5 * math.erfc(z / math.sqrt(2.0))
    return MarketProb(win=min(max(over, 0.0), 1.0), push=0.0)


def prob_over(stat: str, mean: float, line: float, *, sd: float | None = None) -> MarketProb:
    """The over's probability for whichever family ``stat`` belongs to."""
    spread = sd if sd is not None else sd_for(stat, mean)
    if stat in YARD_STATS:
        return yards_prob(mean, spread, line)
    return count_prob(mean, spread, line)
