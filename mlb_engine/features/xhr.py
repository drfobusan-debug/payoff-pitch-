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

import numpy as np
import pandas as pd

from mlb_engine.data.fences import Fence, fence_for_team, wall_distance

# Home plate in Statcast's hit-coordinate frame.
HOME_X, HOME_Y = 125.42, 198.27

# Spread (feet) of the logistic around the wall. Absorbs projection error, wind,
# and the fact that the modelled wall is a five-point approximation of a curve.
CARRY_SIGMA = 14.0

# A batted ball outside this launch-angle band cannot be a home run however far
# it is projected: too low and it is a line drive off the wall, too high and it
# is a pop-up that happens to travel.
MIN_HR_ANGLE = 15.0
MAX_HR_ANGLE = 50.0

# Below this a ball has no chance in any park, so it is skipped outright.
MIN_HR_DISTANCE = 280.0


@dataclass(frozen=True)
class XHRProfile:
    """A batter's expected vs actual home runs over a Statcast slice."""

    pa: int
    batted: int
    hr: int
    xhr: float
    has_data: bool

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


def hr_probability(distance: float, wall: float) -> float:
    """Probability a ball projected ``distance`` clears a wall at ``wall``."""
    return 1.0 / (1.0 + math.exp(-(distance - wall) / CARRY_SIGMA))


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

    angle = balls["launch_angle"].astype(float)
    distance = balls["hit_distance_sc"].astype(float)
    live = balls[
        angle.between(MIN_HR_ANGLE, MAX_HR_ANGLE) & (distance >= MIN_HR_DISTANCE)
    ]
    if live.empty:
        return XHRProfile(pa=n_pa, batted=n_batted, hr=n_hr, xhr=0.0, has_data=True)

    teams = (
        live["home_team"]
        if "home_team" in live
        else pd.Series([None] * len(live), index=live.index)
    )
    fences = _fence_lookup(teams)
    sprays = spray_angle(live["hc_x"], live["hc_y"])

    xhr = 0.0
    for idx, dist in live["hit_distance_sc"].astype(float).items():
        fence = fences.get(str(teams.get(idx)), fence_for_team(None))
        xhr += hr_probability(dist, wall_distance(fence, float(sprays[idx])))

    return XHRProfile(pa=n_pa, batted=n_batted, hr=n_hr, xhr=xhr, has_data=True)
