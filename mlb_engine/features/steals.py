"""A runner's steal rate per time on first base.

The books hang ~150 stolen-base props a night at a single 0.5 line and the
engine priced none of them. What it takes to price one is a per-runner rate and
the number of times he will be standing on first, and the simulator already
supplies the second: this module is only the rate.

Measured on 36,007 player-games of the 2026 season, reading each runner's steals
against his *times on first* -- singles plus walks plus hit by pitch -- rather
than against plate appearances, because that is the denominator a steal actually
has. League-wide: 33,888 times on first, 3,288 attempts (9.7%), 2,528 steals, so
**7.4% of times on first become a steal**.

Three things the fit settled, each tested by fitting on the season through
2026-07-05 and scoring the games after it (5,973 player-games with a runner on
first, 9.74% of which contained a steal):

**A runner's own rate is worth far more than the league's.** The league rate for
everybody scores 0.31531 log loss; his own rate, shrunk, scores **0.28885**. The
raw unshrunk rate is *worse* than the league (0.34622) -- an 0-for-12 runner is
not a runner who never steals -- so the shrinkage is not a formality.

**The prior should be sprint speed, not the league mean.** Their correlations
table claims sprint speed is the one Statcast metric that predicts stealing, and
it does: log(steal rate) against sprint speed fits r=+0.685 across 333 runners.
Regressing toward the speed-implied rate instead of the league mean improves the
held-out fit, and improves it most where the prior is what a runner is priced
on -- among runners with under 25 times on first, **0.27497 -> 0.26347**.

**The catcher and the pitcher are not worth pricing.** The opposing team's steals
allowed per time-on-first spans 0.693..1.363 of league, which looks like a lot,
but adding it to the runner's rate does not improve the held-out fit at any
shrinkage (best 0.28984 against 0.28992 without it), and it only reaches
break-even by being shrunk until it is the identity. It is real as a description
and empty as a prediction, so it is left out.

Held out, the shipped form predicts 0.0979 against a realised 0.0974, and by
predicted quintile reads .013/.019, .035/.046, .066/.076, .118/.124, .258/.222
predicted against actual. The top quintile is the one that matters -- it is where
the books hang the props -- and it is the one place the model is *over*, by three
points, which is an argument for the market being quote-only beyond the usual one.
"""

from __future__ import annotations

import math

# Steals per time on first, league-wide.
LEAGUE_SB_PER_OPP = 0.0740

# log(steal rate) = A + B * sprint_speed (ft/s), fit across 333 runners with 30+
# times on first. Being a fit of the log, it lands near the typical runner rather
# than near the league's opportunity-weighted 0.074: at league-average speed it
# reads 0.043, because most men on first are not stealing and the league rate is
# carried by the few who do. That is the right level for a *shrinkage target* and
# rescaling it up to 0.074 makes the held-out fit worse at every k (0.28917 v
# 0.28884) and turns an unbiased prediction into an over-prediction (0.1063
# against a realised 0.0974), so it is left as fitted.
SPRINT_A = -18.298
SPRINT_B = 0.555
SPRINT_DEFAULT = 27.3

# Times-on-first of prior mixed into every runner's rate. Chosen on the held-out
# fit as the compromise between its two ends: overall it wants 50 (0.28832 v
# 0.28884 at 30), thin-history runners want 20 (0.26346 v 0.26471), and the whole
# point of the speed prior is the thin-history runner, so it sits between them.
# Against a flat league prior at its own best k this is 0.28884 v 0.28992.
SHRINK_K = 30.0

# A rate outside this cannot be measured from one season of times on first.
MIN_RATE = 0.004
MAX_RATE = 0.45


def sprint_prior(sprint_speed: float | None) -> float:
    """The rate a runner of this speed steals at, before his own record.

    A runner with no sprint speed measured gets the league rate rather than the
    speed curve read at league-average speed: the two differ (0.074 v 0.043) and
    the held-out fit was scored this way, because a hitter Statcast has not timed
    is usually one who has barely played rather than one of average speed.
    """
    if sprint_speed is None or not math.isfinite(sprint_speed):
        return LEAGUE_SB_PER_OPP
    return min(max(math.exp(SPRINT_A + SPRINT_B * sprint_speed), MIN_RATE), MAX_RATE)


def steal_rate(
    steals: float,
    times_on_first: float,
    sprint_speed: float | None = None,
) -> float:
    """P(a given time on first becomes a steal) for one runner.

    ``steals`` and ``times_on_first`` are his season to date. With neither, the
    answer is what his legs imply; with a full season, almost entirely his own.
    """
    prior = sprint_prior(sprint_speed)
    if times_on_first <= 0:
        return prior
    rate = (steals + SHRINK_K * prior) / (times_on_first + SHRINK_K)
    return min(max(rate, MIN_RATE), MAX_RATE)
