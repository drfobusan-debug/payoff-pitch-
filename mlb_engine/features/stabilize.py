"""Sample-size stabilization for batter Statcast rate metrics.

A rate metric is noise until its sample is large enough for true skill to
override luck. The published stabilization points are well above what the
engine was requiring: every batter metric sat behind a single 15-batted-ball
floor (``regression.MIN_BBE``), so a 15-BBE barrel rate entered the HR
multiplier with the same authority as a 200-BBE one -- 3-5x looser than the
point where barrel% (~50 BBE) or hard-hit% (~80 BBE) stabilize. Strikeout and
walk rates had no plate-appearance minimum at all.

Rather than a threshold -- which flips a metric from ignored to fully trusted
on one extra batted ball -- each metric is shrunk toward its league baseline in
proportion to sample::

    w = n / (n + n_stabilize)
    stabilized = baseline + (value - baseline) * w

At ``n == n_stabilize`` the metric carries half its observed deviation from
league average, and it approaches the raw value asymptotically. A thin sample
therefore degrades gracefully to league-average rather than being read as
extreme skill.

Baselines are supplied by the caller (``regression.BL_*``) so this module stays
free of them and there is one definition of league average.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

# Sample counters a metric can be denominated in. ``build_batter_regression``
# supplies all four.
BBE = "bbe"  # batted-ball events
PA = "pa"  # plate appearances
SWINGS = "swings"
ZONE_SWINGS = "zone_swings"

# Batted-ball rates stabilize in balls-in-play; xBA/xwOBA/xSLG are quoted in PA
# (~100-150) but computed here per batted ball, so they are converted at the
# league rate of roughly 0.65 balls in play per plate appearance: 100 PA -> 65
# BBE.
_PA_TO_BBE = 0.65


@dataclass(frozen=True)
class StabPoint:
    counter: str
    n_stabilize: int


# Metric name -> where it stabilizes, in the units of its own denominator.
STAB_POINTS: dict[str, StabPoint] = {
    "barrel_rate": StabPoint(BBE, 50),
    "barrel_pa": StabPoint(PA, 77),  # 50 BBE of barrels expressed in PA
    "hard_hit": StabPoint(BBE, 80),
    "sweet_spot": StabPoint(BBE, 80),
    "xba": StabPoint(BBE, round(100 * _PA_TO_BBE)),
    "xslg": StabPoint(BBE, round(100 * _PA_TO_BBE)),
    "xwoba": StabPoint(BBE, round(100 * _PA_TO_BBE)),
    "woba": StabPoint(BBE, round(100 * _PA_TO_BBE)),
    "gb_rate": StabPoint(BBE, 50),
    "gb_pct": StabPoint(BBE, 50),
    "k_pct": StabPoint(PA, 60),
    "bb_pct": StabPoint(PA, 120),
    "whiff": StabPoint(SWINGS, 100),
    "zone_contact": StabPoint(ZONE_SWINGS, 100),
}

# Deliberately not stabilized:
#   max_ev      -- a maximum, not a rate; shrinking it toward a mean would pull
#                  genuine top-end power down, and it was the strongest HR
#                  separator in the graded backtest.
#   babip       -- read as a *luck* indicator, not skill; shrinking it toward
#                  league average is exactly what would erase the signal.
#   bat_speed,
#   sprint_speed -- physical measurements that stabilize in a handful of swings.


def shrink(value: float, baseline: float, n: int, n_stabilize: int) -> float:
    """Regress ``value`` toward ``baseline`` by its sample weight.

    Returns ``value`` unchanged when it is NaN, when the sample size is unknown
    (``n <= 0``) or when stabilization is disabled (``n_stabilize <= 0``), so a
    missing counter never collapses a metric to league average.
    """
    if math.isnan(value) or n <= 0 or n_stabilize <= 0:
        return value
    w = n / (n + n_stabilize)
    return baseline + (value - baseline) * w


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val not in ("0", "false", "False", "")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Stabilizer:
    """Config for the per-metric stabilization shrink.

    ``scale`` multiplies every stabilization point: 1.0 uses the published
    values, 0.5 trusts small samples twice as fast, 0.0 disables the shrink
    entirely (identical to ``enabled=False``).
    """

    enabled: bool = True
    scale: float = 1.0

    @classmethod
    def from_env(cls) -> Stabilizer:
        return cls(
            enabled=_env_flag("MLBE_STABILIZE", True),
            scale=_env_float("MLBE_STABILIZE_SCALE", 1.0),
        )

    def n_for(self, metric: str) -> int:
        point = STAB_POINTS.get(metric)
        if point is None or not self.enabled:
            return 0
        return round(point.n_stabilize * self.scale)

    def value(
        self, metric: str, value: float, baseline: float, counts: Mapping[str, int]
    ) -> float:
        """Stabilize one metric. Unknown metrics pass through untouched."""
        point = STAB_POINTS.get(metric)
        if point is None or not self.enabled:
            return value
        return shrink(value, baseline, counts.get(point.counter, 0), self.n_for(metric))

    def apply(
        self,
        values: Mapping[str, float],
        baselines: Mapping[str, float],
        counts: Mapping[str, int],
    ) -> dict[str, float]:
        """Stabilize every metric in ``values`` that has a baseline and a point."""
        out = dict(values)
        for metric, raw in values.items():
            baseline = baselines.get(metric)
            if baseline is None:
                continue
            out[metric] = self.value(metric, raw, baseline, counts)
        return out
