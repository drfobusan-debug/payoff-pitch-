"""How many plate appearances each hitter metric needs before it means anything.

The power screen scores eleven metrics one point apiece. That is only defensible
if the eleven stabilise at similar speeds, and the measurement says they do not:
bat speed agrees with itself inside fifteen plate appearances, and wOBA does not
agree with itself inside two hundred and fifty. Scoring them equally hands a
hitter the same credit for a real skill and for two weeks of batted-ball luck,
which is how a screen ends up selecting the hitters whose results have already
happened.

The numbers below are split-half reliabilities measured on the engine's own
Statcast window by ``scripts/measure_metric_reliability.py`` -- 145,707 plate
appearances over 638 hitters, 2026 season through 8/21. For each target sample
size every hitter with twice that many PA has his plate appearances shuffled and
split in two, the metric is computed on each half, and the correlation across
hitters is the reliability at that size.

Read the grid, because the spread is the whole point:

    metric        15 PA   60 PA   250 PA    r=.50 at
    bat speed      .67     .89      .98       15 PA
    O-Swing%       .33     .66      .90       27
    Brl%           .14     .42      .75       81
    EV90           .17     .50      .85       59
    HH%            .09     .29      .64      123
    xSLG           .12     .25      .62      185
    xwOBAcon       .10     .26      .55      190
    xBA            .06     .20      .50      249
    BA             .02     .07      .22      never
    SLG            .05     .13      .26      never
    OPS            .07     .08      .18      never
    wRC+           .10     .11      .23      never

Two more were measured for the batter regression article, which reads them as
levels rather than scoring them (2026 through 8/28, 162,464 PA over 1,397
hitters):

    metric        15 PA   60 PA   250 PA    r=.50 at   r=.70 at
    fastball whiff .26     .56      .85       51 PA      108 PA
    max EV         .18     .55      .75       49         195

Max EV half-repeats as quickly as fastball whiff% and then stops improving: it
is one batted ball out of the whole window, so the fifty-first plate appearance
buys as much of it as the two-hundredth. That is why the article quotes it over
the wider window and still calls a *move* in it unreadable -- see
``features.power_change``.

Four of the eleven scored metrics -- wRC+, OPS, BA and SLG -- never reach r=.50
at any sample the screen ever sees. They are not weak signals, they are the same
hitter disagreeing with himself, and the screen was paying a point for each.

Two honest caveats. Split-half is an *upper* bound on forecasting value: both
halves come from the same weeks, so a metric that cannot agree with itself
certainly cannot predict tonight, but agreeing with itself does not prove it
predicts anything either. And the pool thins as the sample size grows -- at 250
PA only regulars qualify, and regulars have a narrower true spread than the
league, which depresses the correlation. The far-right column is therefore
pessimistic and the ordering, which is what the screen uses, is the reliable
part.

Used two ways: metrics vote in proportion to their reliability at the hitter's
own sample size, and a metric below ``READABLE_R`` cannot promote a hitter
through a cut at all.
"""

from __future__ import annotations

import math

#: Reliability at which a metric is allowed to carry a screen decision by itself.
READABLE_R = 0.50

#: Measured split-half reliability curves: metric -> ((PA, r), ...), ascending.
#: Keys are ``HitterLine`` attributes; the four unscored diagnostics (contact,
#: K%, bat speed) are kept because the note quotes them and the same warning
#: applies.
CURVES: dict[str, tuple[tuple[int, float], ...]] = {
    "bat_speed": ((15, 0.67), (25, 0.76), (40, 0.85), (60, 0.89), (90, 0.93), (130, 0.96),
                  (180, 0.97), (250, 0.98)),
    "contact": ((15, 0.34), (25, 0.46), (40, 0.57), (60, 0.67), (90, 0.77), (130, 0.82),
                (180, 0.85), (250, 0.90)),
    "k": ((15, 0.28), (25, 0.33), (40, 0.46), (60, 0.56), (90, 0.62), (130, 0.73),
          (180, 0.77), (250, 0.85)),
    "osw": ((15, 0.33), (25, 0.48), (40, 0.59), (60, 0.66), (90, 0.74), (130, 0.79),
            (180, 0.84), (250, 0.90)),
    "ev90": ((15, 0.17), (25, 0.22), (40, 0.39), (60, 0.50), (90, 0.58), (130, 0.71),
             (180, 0.81), (250, 0.85)),
    "brl": ((15, 0.14), (25, 0.22), (40, 0.32), (60, 0.42), (90, 0.54), (130, 0.59),
            (180, 0.72), (250, 0.75)),
    "hh": ((15, 0.09), (25, 0.16), (40, 0.16), (60, 0.29), (90, 0.38), (130, 0.53),
           (180, 0.63), (250, 0.64)),
    "xslg": ((15, 0.12), (25, 0.15), (40, 0.18), (60, 0.25), (90, 0.27), (130, 0.39),
             (180, 0.49), (250, 0.62)),
    "xwoba_con": ((15, 0.10), (25, 0.08), (40, 0.19), (60, 0.26), (90, 0.29), (130, 0.40),
                  (180, 0.49), (250, 0.55)),
    "xba": ((15, 0.06), (25, 0.15), (40, 0.18), (60, 0.20), (90, 0.26), (130, 0.30),
            (180, 0.39), (250, 0.50)),
    "ba": ((15, 0.02), (25, 0.05), (40, 0.06), (60, 0.07), (90, 0.08), (130, 0.10),
           (180, 0.18), (250, 0.22)),
    "slg": ((15, 0.05), (25, 0.04), (40, 0.12), (60, 0.13), (90, 0.14), (130, 0.15),
            (180, 0.17), (250, 0.26)),
    "ops": ((15, 0.07), (25, 0.07), (40, 0.06), (60, 0.08), (90, 0.12), (130, 0.07),
            (180, 0.15), (250, 0.18)),
    "wrc": ((15, 0.10), (25, 0.04), (40, 0.11), (60, 0.11), (90, 0.14), (130, 0.11),
            (180, 0.13), (250, 0.23)),
    # Measured by the same method over 162,464 plate appearances and 1,397
    # hitters, 2026 through 8/28. Neither is scored by the screen; both are read
    # as levels by the batter regression article, which is why they are sized
    # here rather than by eye.
    "max_ev": ((15, 0.18), (25, 0.31), (40, 0.46), (60, 0.55), (90, 0.61), (130, 0.64),
               (180, 0.69), (250, 0.75)),
    "fb_whiff": ((15, 0.26), (25, 0.37), (40, 0.42), (60, 0.56), (90, 0.66), (130, 0.75),
                 (180, 0.80), (250, 0.85)),
}

#: Reliability at which a level is quoted as stable rather than as a thin read:
#: above it most of what separates two hitters is the hitters and not the sample.
STABLE_R = 0.70


def _monotone(curve: tuple[tuple[int, float], ...]) -> tuple[tuple[int, float], ...]:
    """The measured curve with its dips lifted to the running maximum.

    A metric cannot become less reliable as its sample grows -- reliability rises
    with the number of events by construction -- so every fall in the measured
    grid (OPS reads .07 at 25 PA and .06 at 40) is sampling noise in the
    measurement, not a property of the metric. Taking the running maximum keeps
    the estimate interpretable and monotone without touching the ordering, which
    is what the screen actually uses.
    """
    best = 0.0
    out: list[tuple[int, float]] = []
    for n, r in curve:
        best = max(best, r)
        out.append((n, best))
    return tuple(out)


CURVES = {metric: _monotone(curve) for metric, curve in CURVES.items()}

#: Metric -> the sample at which it first reaches ``READABLE_R``, or ``inf``.
PA_FOR_READABLE: dict[str, float] = {}

#: Metric -> the sample at which it first reaches ``STABLE_R``, or ``inf``. This
#: is the window a report should read the level over when it has the choice --
#: the optimum in the only sense a reliability curve can define one, the point
#: past which more plate appearances buy very little agreement.
PA_FOR_STABLE: dict[str, float] = {}


def _crossing(curve: tuple[tuple[int, float], ...], target: float) -> float:
    for (n0, r0), (n1, r1) in zip(curve, curve[1:], strict=False):
        if r0 < target <= r1:
            return n0 + (target - r0) / (r1 - r0) * (n1 - n0)
    if curve and curve[0][1] >= target:
        return float(curve[0][0])
    return math.inf


for _metric, _curve in CURVES.items():
    PA_FOR_READABLE[_metric] = _crossing(_curve, READABLE_R)
    PA_FOR_STABLE[_metric] = _crossing(_curve, STABLE_R)


def reliability(metric: str, pa: float) -> float:
    """Split-half reliability of ``metric`` at ``pa`` plate appearances.

    Interpolated in log PA, which is the scale a reliability curve is roughly
    linear on. Below the measured grid the first point is used rather than
    extrapolated to zero -- a hitter with 8 PA is not *more* readable than the
    curve says, and the screen's own PA floor keeps that case rare. Above it the
    last measured point holds, which understates a large sample and is the
    conservative direction. An unmeasured metric returns 1.0, so adding a metric
    to the screen without measuring it changes nothing about how it is scored.
    """
    curve = CURVES.get(metric)
    if curve is None:
        return 1.0
    if pa <= curve[0][0]:
        return curve[0][1]
    if pa >= curve[-1][0]:
        return curve[-1][1]
    for (n0, r0), (n1, r1) in zip(curve, curve[1:], strict=False):
        if n0 <= pa <= n1:
            span = math.log(n1) - math.log(n0)
            frac = (math.log(pa) - math.log(n0)) / span if span else 0.0
            return r0 + frac * (r1 - r0)
    return curve[-1][1]


def readable(metric: str, pa: float, target: float = READABLE_R) -> bool:
    """Whether ``metric`` at ``pa`` plate appearances may carry a decision."""
    return reliability(metric, pa) >= target


def never_readable(metric: str) -> bool:
    """Whether the metric fails ``READABLE_R`` at every sample the screen sees."""
    return math.isinf(PA_FOR_READABLE.get(metric, math.inf))
