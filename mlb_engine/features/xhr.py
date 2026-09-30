"""Expected home runs: what a batter's contact was worth, park noise removed.

The simulator's home-run rate starts from a batter's observed HR/PA. That number
answers "how many did he hit", which is not the same question as "how well did
he hit them": it carries the dimensions of the parks he visited, the wind that
blew that week, and the fence a ball missed by a foot. Statcast's expected-HR
idea replaces the counting stat with a physical one -- take each batted ball's
projected distance and spray angle, ask whether it clears the wall it was hit
toward, and sum the answers.

``xHR/PA`` computed this way is both more stable and more predictive of future
home runs than HR/PA, so :func:`mlb_engine.features.rolling.blend_hr_rate` uses
it as the prior a thin or lucky sample is pulled toward. A hitter with 15 home
runs on 8 expected ones is not a 15-home-run hitter.

Each ball is scored softly rather than as a yes/no: distance is a projection,
the wall is a piecewise approximation, and carry varies with the air. A ball
projected to land exactly on the wall counts as half a home run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from mlb_engine.data.fences import (
    LEAGUE_FENCE,
    Fence,
    fence_for_team,
    get_fence,
    wall_distance,
)
from mlb_engine.data.parks import get_park
from mlb_engine.data.statcast import batted_balls

# Home plate in Statcast's hit-coordinate frame.
HOME_X, HOME_Y = 125.42, 198.27

# Where the logistic sits and how wide it is, fitted to whether the ball was
# actually a home run rather than assumed (``scripts.xhr_wall_fit``). Both were
# hand-set -- a 14-foot spread centred on the wall itself -- and that curve gave
# a ball projected to land *20 to 40 feet short* an 11% chance of leaving the
# park, where the measured rate is 0.0%. Summed over a hitter's window it
# over-counted expected home runs by 42% out of sample (Brier .0438 -> .0264).
#
# The offset is physical: ``hit_distance_sc`` projects where the ball lands, and
# a ball has to clear the wall's *height* on the way, which the fit prices at
# roughly nine feet of extra carry.
WALL_OFFSET = 8.6
CARRY_SIGMA = 5.2

# The same curve, widened, for the *counterfactual* wall in the park re-score.
# Scoring the ball against the wall it was actually hit toward is a measured
# question, and the fit says it is nearly a step function. Asking what the same
# ball would have been worth in another park is not: it would have been hit in
# different air, off a different pitcher, into a different wind, none of which
# the projection carries. The counterfactual keeps the pre-fit 14-foot spread so
# a park ratio stays a tilt rather than becoming a step, which is also what stops
# a hitter whose distances differ from the league grid earning a park term he has
# not earned.
PARK_CARRY_SIGMA = 14.0

# A window is only worth reading if Statcast measured most of the contact in it.
# Distance and hit coordinates go missing in bulk -- whole date ranges of the
# cache carry no ``hit_distance_sc`` at all -- and the expected-HR sum is over
# the balls that *were* measured while the denominator is every plate
# appearance, so a hitter whose window is half-reported has his prior halved and
# the blend then crushes his home-run rate toward zero. Below this share of his
# batted balls the profile reports no data instead of a diluted number.
MIN_MEASURED_SHARE = 0.35

# A batted ball outside this launch-angle band cannot be a home run however far
# it is projected: too low and it is a line drive off the wall, too high and it
# is a pop-up that happens to travel.
MIN_HR_ANGLE = 15.0
MAX_HR_ANGLE = 50.0

# Below this a ball has no chance in any park, so it is skipped outright.
MIN_HR_DISTANCE = 280.0

# A park multiplier is a ratio of two small sums, so it needs a real spray chart
# underneath it to mean anything: enough balls in the air to describe where the
# hitter puts them, and enough expected home runs for the ratio to be stable.
# Below either, the batter falls back to a neutral park rather than betting on
# the shape of three fly balls.
MIN_PARK_BALLS = 25
MIN_PARK_XHR = 1.0

# Geometry alone cannot double or halve a hitter -- the extremes here are the
# real ones (a pull-happy lefty in the Bronx, the same bat in Detroit) and the
# clamp stops a thin or lopsided spray chart from running away with the price.
PARK_MULT_FLOOR = 0.80
PARK_MULT_CEILING = 1.25


@dataclass(frozen=True)
class XHRProfile:
    """A batter's expected vs actual home runs over a Statcast slice."""

    pa: int
    batted: int
    hr: int
    xhr: float
    has_data: bool
    # Share of his batted balls Statcast measured well enough to score. The xHR
    # sum is scaled up by its inverse, so this records how much of the number
    # is measured and how much is imputed from the balls that were.
    measured_share: float = 1.0

    @property
    def xhr_per_pa(self) -> float:
        """Expected home runs per plate appearance (NaN without PAs/data)."""
        if not self.has_data or self.pa <= 0:
            return float("nan")
        return self.xhr / self.pa

    @property
    def hr_per_pa(self) -> float:
        if self.pa <= 0:
            return float("nan")
        return self.hr / self.pa

    @property
    def luck(self) -> float:
        """Actual minus expected home runs: positive is park/weather fortune."""
        if not self.has_data:
            return float("nan")
        return self.hr - self.xhr


def spray_angle(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    """Spray angle in degrees; negative is the third-base side, 0 is center."""
    return pd.Series(
        np.degrees(
            np.arctan2(hc_x.astype(float) - HOME_X, HOME_Y - hc_y.astype(float))
        ),
        index=hc_x.index,
    )


def hr_probability(distance: float, wall: float, sigma: float = CARRY_SIGMA) -> float:
    """Probability a ball projected ``distance`` clears a wall at ``wall``.

    Half a home run is a ball projected ``WALL_OFFSET`` feet *past* the wall,
    not one landing on it: the wall has height, and the fit prices it.
    """
    return 1.0 / (1.0 + math.exp(-(distance - wall - WALL_OFFSET) / sigma))


def _fence_lookup(teams: pd.Series) -> dict[str, Fence]:
    return {str(t): fence_for_team(str(t)) for t in teams.dropna().unique()}


def batter_xhr(bdf: pd.DataFrame) -> XHRProfile:
    """Expected home runs over a batter's pitch-level Statcast slice.

    Requires ``hit_distance_sc`` plus hit coordinates; without them the profile
    reports ``has_data=False`` so callers leave the observed rate alone rather
    than blending toward a fabricated zero.
    """
    events = bdf["events"].dropna() if "events" in bdf else pd.Series(dtype=object)
    n_pa = int(len(events))
    n_hr = int(events.eq("home_run").sum())

    needed = {"hit_distance_sc", "hc_x", "hc_y", "launch_angle"}
    if not needed.issubset(bdf.columns):
        return XHRProfile(pa=n_pa, batted=0, hr=n_hr, xhr=0.0, has_data=False)

    balls = bdf.dropna(subset=["hit_distance_sc", "hc_x", "hc_y", "launch_angle"])
    n_batted = int(len(balls))
    if n_batted == 0:
        return XHRProfile(pa=n_pa, batted=0, hr=n_hr, xhr=0.0, has_data=False)

    # What share of his contact this rests on. The balls Statcast failed to
    # measure were still swings that could have left the park, so the measured
    # ones stand in for them -- but only while most of the contact is measured.
    share = _measured_share(bdf, n_batted)
    if share < MIN_MEASURED_SHARE:
        return XHRProfile(
            pa=n_pa, batted=n_batted, hr=n_hr, xhr=0.0,
            has_data=False, measured_share=share,
        )

    angle = balls["launch_angle"].astype(float)
    distance = balls["hit_distance_sc"].astype(float)
    live = balls[
        angle.between(MIN_HR_ANGLE, MAX_HR_ANGLE) & (distance >= MIN_HR_DISTANCE)
    ]
    if live.empty:
        return XHRProfile(
            pa=n_pa, batted=n_batted, hr=n_hr, xhr=0.0,
            has_data=True, measured_share=share,
        )

    teams = (
        live["home_team"]
        if "home_team" in live
        else pd.Series([None] * len(live), index=live.index)
    )
    fences = _fence_lookup(teams)
    sprays = spray_angle(live["hc_x"], live["hc_y"])

    # Positional, not label: a Statcast slice stitched from cached day-frames
    # repeats index labels, and ``sprays[idx]`` then hands back every row that
    # shares the label instead of one angle.
    xhr = 0.0
    for dist, spray, team in zip(
        live["hit_distance_sc"].astype(float).to_numpy(),
        sprays.to_numpy(),
        teams.to_numpy(),
        strict=True,
    ):
        fence = fences.get(str(team), fence_for_team(None))
        xhr += hr_probability(float(dist), wall_distance(fence, float(spray)))

    return XHRProfile(
        pa=n_pa, batted=n_batted, hr=n_hr, xhr=xhr / share,
        has_data=True, measured_share=share,
    )


def _measured_share(bdf: pd.DataFrame, n_measured: int) -> float:
    """Share of the slice's balls in play that carry distance and coordinates.

    Balls in play are counted the way the rest of the engine counts them, so a
    foul with an exit velocity is not mistaken for contact that could have left
    the park. With no way to count them, the measured balls are all there is and
    the share is 1.0 -- the previous behaviour.
    """
    bip = batted_balls(bdf)
    n_bip = int(len(bip))
    if n_bip <= 0:
        return 1.0
    return min(max(n_measured / n_bip, 0.0), 1.0)


def _live_balls(bdf: pd.DataFrame) -> pd.DataFrame | None:
    """Batted balls carrying far enough, and at an angle, to be home runs."""
    needed = {"hit_distance_sc", "hc_x", "hc_y", "launch_angle"}
    if not needed.issubset(bdf.columns):
        return None
    balls = bdf.dropna(subset=["hit_distance_sc", "hc_x", "hc_y", "launch_angle"])
    if balls.empty:
        return None
    angle = balls["launch_angle"].astype(float)
    distance = balls["hit_distance_sc"].astype(float)
    return balls[
        angle.between(MIN_HR_ANGLE, MAX_HR_ANGLE) & (distance >= MIN_HR_DISTANCE)
    ]


def xhr_at_fence(
    bdf: pd.DataFrame, fence: Fence, sigma: float = PARK_CARRY_SIGMA
) -> float:
    """Expected home runs if every one of these batted balls were hit here."""
    live = _live_balls(bdf)
    if live is None:
        return float("nan")
    if live.empty:
        return 0.0
    sprays = spray_angle(live["hc_x"], live["hc_y"])
    return sum(
        hr_probability(float(d), wall_distance(fence, float(spray)), sigma)
        for d, spray in zip(
            live["hit_distance_sc"].astype(float).to_numpy(),
            sprays.to_numpy(),
            strict=True,
        )
    )


def _league_spray() -> list[tuple[float, float, float]]:
    """A fixed (spray, distance, weight) grid standing in for an average hitter.

    Deterministic and league-generic: spray centred slightly to the pull side
    with a 20-degree spread, distance centred at 350 feet with a 45-foot spread.
    """
    grid = []
    for spray in range(-45, 50, 5):
        w_s = math.exp(-0.5 * (spray / 20.0) ** 2)
        for dist in range(280, 460, 10):
            w_d = math.exp(-0.5 * ((dist - 350.0) / 45.0) ** 2)
            grid.append((float(spray), float(dist), w_s * w_d))
    return grid


LEAGUE_SPRAY = _league_spray()


@lru_cache(maxsize=64)
def park_shape_baseline(venue_id: int | None) -> float:
    """What an *average* hitter gains or loses from this park's geometry alone."""
    here = sum(
        w * hr_probability(d, wall_distance(get_fence(venue_id), s), PARK_CARRY_SIGMA)
        for s, d, w in LEAGUE_SPRAY
    )
    neutral = sum(
        w * hr_probability(d, wall_distance(LEAGUE_FENCE, s), PARK_CARRY_SIGMA)
        for s, d, w in LEAGUE_SPRAY
    )
    return here / neutral if neutral > 0 else 1.0


def park_hr_multiplier(bdf: pd.DataFrame, venue_id: int | None) -> float:
    """How much tonight's park is worth to *this* hitter's batted-ball profile.

    :func:`batter_xhr` deliberately strips park effects out, scoring a hitter's
    contact against the walls he happened to face so the base rate measures the
    swing rather than the itinerary. This puts the park back -- but the one he
    is walking into, and only as it applies to him.

    Two separable things decide what a park is worth, and conflating them gets
    Coors Field exactly backwards. Its fences are among the deepest in baseball,
    so pure geometry scores it as a *pitcher's* park; it plays as the best home-
    run park in the league because of the altitude, which no wall diagram can
    see. So the level comes from the park's measured ``carry_factor`` -- an
    empirical number that already contains the air -- and geometry supplies only
    the part that is specific to this hitter:

        multiplier = carry_factor x  xHR(his contact @ here) / xHR(@ average)
                                    -------------------------------------------
                                     same ratio for a league-average spray chart

    The denominator is what makes it batter-specific rather than another scalar.
    An average hitter comes out at roughly ``carry_factor`` in every park; the
    deviation is earned by *his* spray chart, so Yankee Stadium's 314ft porch
    pays a pull-heavy left-handed hitter and does nothing for a right-handed bat
    who works the opposite field. Returns NaN when the sample is too thin or the
    distance data is missing, which callers read as a neutral park.
    """
    live = _live_balls(bdf)
    if live is None or len(live) < MIN_PARK_BALLS:
        return float("nan")
    neutral = xhr_at_fence(bdf, LEAGUE_FENCE)
    if neutral != neutral or neutral < MIN_PARK_XHR:  # NaN or too thin
        return float("nan")
    here = xhr_at_fence(bdf, get_fence(venue_id))
    if here != here:
        return float("nan")
    baseline = park_shape_baseline(venue_id)
    if baseline <= 0:
        return float("nan")
    park = get_park(venue_id) if venue_id is not None else None
    level = park.carry_factor if park is not None else 1.0
    mult = level * (here / neutral) / baseline
    return min(max(mult, PARK_MULT_FLOOR), PARK_MULT_CEILING)
