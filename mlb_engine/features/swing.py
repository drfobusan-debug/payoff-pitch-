"""What a hitter's swing is, read at the sample each measure of it repeats at.

The screen's regression test was a residual: wOBA minus xwOBA, cut at 50 points.
That asks whether a hitter's results have outrun his contact, which is a
statement about which batted balls fell in. It says nothing about the swing that
produced them, and the threshold was hand-set from one morning's run.

Bat tracking measures the swing itself, and it does so on every competitive
swing rather than only on batted balls, which is why it repeats on a fraction of
the sample a batted-ball rate needs. Measured by
``scripts/measure_swing_reliability.py`` over 515,417 tracked swings from 685
hitters, split into adjacent equal blocks and correlated block to block:

    metric          10 sw   50 sw   150 sw   250 sw   r=.50 at
    bat speed         .73     .91      .95      .95       3 swings
    fast-swing%       .69     .89      .94      .95       4
    swing length      .69     .91      .95      .96       4
    blast%            .20     .53      .74      .80      48
    squared-up%       .15     .43      .71      .78      64

Bat speed is close to a physical constant of the hitter: it half-repeats inside
three swings and reaches .90 in fifty. The two contact-quality rates need about
fifty to sixty-five, roughly a fortnight of playing time -- still a quarter of
what the barrel rate needs (81 PA) and a fraction of xBA's 249.

**What that buys, out of time.** 3,175 batter-windows over 458 hitters across
2025 and 2026, thirteen anchor dates, each window's predictors read before the
anchor and the target being the fortnight after it, with standard errors
clustered on the hitter:

    added on top of trailing wOBA and xwOBA        TB/PA      HR/PA     H/PA
      blast% level                               t +6.6     t +8.4    t +0.3
      bat speed level                            t +5.4     t +8.9    t -1.9
      fast-swing% level                          t +4.4     t +7.4    t -2.1
      squared-up% level                          t -2.8     t -9.5    t +5.9
      swing length level                         t +3.5     t +6.0    t -2.3

Blast rate alone adds more to total bases (dR2 +.0168) than trailing wOBA and
xwOBA explain between them (R2 .0131). The split by market is sharp and is the
reason the five are not pooled into one index: **bat speed and blast rate predict
total bases and home runs; squared-up rate predicts hits and is negatively signed
on home runs.** A hitter who squares the ball up is not the same hitter as one
who blasts it, and one number for both hides that.

**The trends predict nothing, on either window size.** The same panel, using each
metric's recent block against the immediately preceding one:

    delta added on top of wOBA and xwOBA, target TB/PA
      bat speed t +1.4   fast-swing% t +1.7   squared-up% t +0.1
      blast% t -0.3      swing length t -0.2  five-metric composite t +1.0

That is the third trend this engine has tested and refused: the 3w-vs-6w barrel
trend (t -0.1, PR #109) and the CSW trend (PR #147) died the same way. A
biomechanical trend is a better-measured version of a thing that does not
forecast. Only levels are exposed here, deliberately.

**Where it relieves false negatives.** Of the 471 windows the luck-gap cut
removes, the half with the better swing went on to post .3801 TB/PA and the half
with the worse swing .3355 -- and the better half beat the .3708 average of every
hitter the cut *kept*. Half of what that cut throws away is not luck coming back;
it is a good swing that had a good fortnight. The rescue in
``output.power_screen`` is that finding, and the coefficient behind it is t +3.2
inside the flagged group.

**Two honest limits.** Savant's leaderboard publishes the authoritative rates but
cannot be sliced by swing count, which is what a reliability window is measured
in, so squared-up and blast are reconstructed here from the pitch-level collision
model with the cuts calibrated to the league rate. Per hitter that lands at
r +.86 and +.76 against the official figures over the same dates (bat speed
+.996, swing length +.997), and the reconstruction's spread is 40% too wide for
blast -- so the coefficients above are attenuated and the true effects are larger
than the t-statistics say, not smaller. And attack angle, the fifth measure
Franz asked for, is published by neither the leaderboard endpoints nor the
pitch-level feed; it is absent rather than approximated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

#: Reliability at which a swing measure may carry a decision by itself.
READABLE_R = 0.50

#: Descriptions that mark a competitive swing. Bunts and pitches the hitter never
#: offered at carry no bat-tracking row, so the filter is the tracked swing.
SWING_DESC = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
}

#: Savant's own definitions, on the pitch-level feed.
FAST_SWING_MPH = 75.0  # a "fast swing" is one at or above this bat speed
MIN_TRACKED_MPH = 50.0  # below this the tracker mis-fired, not a swing

#: Collision-model cuts, calibrated so the league squared-up and blast rates
#: reproduce Savant's published .246 and .100 over 7/12-8/21/2026.
SQUARED_UP_RATIO = 0.841  # exit speed as a share of the collision maximum
BLAST_BAT_SPEED = 72.9  # a blast is a squared-up ball struck at least this fast

#: Measured split-half reliability: metric -> ((swings, r), ...), ascending.
CURVES: dict[str, tuple[tuple[int, float], ...]] = {
    "bat_speed": ((3, 0.49), (5, 0.60), (10, 0.73), (25, 0.85), (50, 0.91), (100, 0.94),
                  (150, 0.95), (250, 0.95)),
    "fast": ((3, 0.44), (5, 0.55), (10, 0.69), (25, 0.83), (50, 0.89), (100, 0.92),
             (150, 0.94), (250, 0.95)),
    "swing_length": ((3, 0.46), (5, 0.56), (10, 0.69), (25, 0.84), (50, 0.91), (100, 0.94),
                     (150, 0.95), (250, 0.96)),
    "blast": ((3, 0.08), (5, 0.12), (10, 0.20), (25, 0.37), (50, 0.53), (100, 0.67),
              (150, 0.74), (250, 0.80)),
    "squared_up": ((3, 0.06), (5, 0.09), (10, 0.15), (25, 0.28), (50, 0.43), (100, 0.61),
                   (150, 0.71), (250, 0.78)),
}

#: League level and spread of each metric, read on ``window()`` -- so a z-score
#: compares a hitter with the league measured the same way he was. Same slice the
#: reliability curves were measured on.
LEAGUE: dict[str, tuple[float, float]] = {
    "bat_speed": (70.99, 3.12),
    "fast": (0.234, 0.209),
    "squared_up": (0.246, 0.062),
    "blast": (0.100, 0.048),
    "swing_length": (7.27, 0.470),
}

#: How much of each window a level is read over, as a multiple of the r=.50
#: crossing. The crossing itself is where a metric half-repeats, and for bat
#: speed that is three swings -- reliable in the split-half sense and still one
#: at-bat, so a game's worth of variance would drive the read. Four times the
#: crossing puts every measure above r=.75 and reproduced the same signs and
#: significance in the panel, so the level is read there and the crossing is kept
#: as the floor below which the metric is not read at all.
STABLE_MULT = 4


def _crossing(curve: tuple[tuple[int, float], ...], target: float) -> float:
    for (n0, r0), (n1, r1) in zip(curve, curve[1:], strict=False):
        if r0 < target <= r1:
            return n0 + (target - r0) / (r1 - r0) * (n1 - n0)
    if curve and curve[0][1] >= target:
        return float(curve[0][0])
    return math.inf


#: Metric -> swings at which it first reaches ``READABLE_R``.
SWINGS_FOR_READABLE: dict[str, float] = {m: _crossing(c, READABLE_R) for m, c in CURVES.items()}

#: Metric -> the window its level is read over.
WINDOW: dict[str, int] = {
    m: max(1, int(math.ceil(SWINGS_FOR_READABLE[m])) * STABLE_MULT) for m in CURVES
}


def reliability(metric: str, swings: float) -> float:
    """Split-half reliability of ``metric`` at ``swings`` competitive swings.

    Interpolated in log swings. Outside the measured grid the nearest endpoint
    holds rather than extrapolating, which understates a very large sample and is
    the conservative direction. An unmeasured metric returns 1.0.
    """
    curve = CURVES.get(metric)
    if curve is None:
        return 1.0
    if swings <= curve[0][0]:
        return curve[0][1]
    if swings >= curve[-1][0]:
        return curve[-1][1]
    for (n0, r0), (n1, r1) in zip(curve, curve[1:], strict=False):
        if n0 <= swings <= n1:
            span = math.log(n1) - math.log(n0)
            frac = (math.log(swings) - math.log(n0)) / span if span else 0.0
            return r0 + frac * (r1 - r0)
    return curve[-1][1]


def readable(metric: str, swings: float, target: float = READABLE_R) -> bool:
    """Whether ``metric`` at ``swings`` swings may carry a decision."""
    return reliability(metric, swings) >= target


@dataclass
class SwingProfile:
    """One hitter's swing, each measure read over its own window of swings.

    ``swings`` is what was tracked in the slice; each level is ``nan`` when the
    slice does not reach that metric's r=.50 floor, so a thin hitter reads as
    unmeasured rather than as league average.
    """

    swings: int
    bat_speed: float = math.nan
    fast: float = math.nan
    squared_up: float = math.nan
    blast: float = math.nan
    swing_length: float = math.nan

    def levels(self) -> dict[str, float]:
        """Metric -> level, ``nan`` where the slice never reached its floor."""
        return {
            "bat_speed": self.bat_speed,
            "fast": self.fast,
            "squared_up": self.squared_up,
            "blast": self.blast,
            "swing_length": self.swing_length,
        }

    def z(self, metric: str) -> float:
        """The level in league standard deviations, ``nan`` when unmeasured."""
        val = self.levels().get(metric, math.nan)
        mu, sd = LEAGUE.get(metric, (math.nan, math.nan))
        if val != val or not sd:
            return math.nan
        return (val - mu) / sd

    @property
    def power_z(self) -> float:
        """Bat speed and blast rate, the two that predict bases and home runs.

        Both are required rather than averaging whichever is present. Bat speed
        half-repeats in three swings and blast rate needs forty-eight, so falling
        back to bat speed alone would let a hitter be read on one at-bat's worth
        of tracking -- and it is the pair the panel measured. Kept separate from
        ``contact_z`` because the two carry opposite signs on the home-run line
        and pooling all five cancels both.
        """
        zs = (self.z("bat_speed"), self.z("blast"))
        if any(z != z for z in zs):
            return math.nan
        return sum(zs) / len(zs)

    @property
    def contact_z(self) -> float:
        """Squared-up rate: the measure that predicts hits rather than bases."""
        return self.z("squared_up")


#: Verdicts of the second stage, once the first has read the luck gap.
CONFIRMED = "confirmed"  # the swing agrees with what the gap says is coming
CONTRADICTED = "contradicted"  # the swing says the gap is about to be wrong
UNMEASURED = "unmeasured"  # too few tracked swings to read a level at all


def stage_two(luck_gap: float, prof: SwingProfile | None, *, min_power_z: float = 0.0) -> str:
    """Confirm or contradict a luck-gap read with the swing underneath it.

    ``luck_gap`` is wOBA minus xwOBA, positive when the results have outrun the
    contact. The gap is a statement about which batted balls fell in; it cannot
    say whether the swing producing them was good. So the two are crossed:

    * results above the contact and a below-league swing -> ``CONFIRMED``, the
      production has nothing holding it up;
    * results above the contact and an above-league swing -> ``CONTRADICTED``,
      the fortnight was lucky and the hitter is also good;
    * results below the contact and an above-league swing -> ``CONFIRMED``, the
      rebound has a swing behind it;
    * results below the contact and a below-league swing -> ``CONTRADICTED``,
      the low results are the level rather than the luck.

    Read on ``power_z`` -- bat speed and blast rate -- because those are the two
    the panel gives to bases and home runs. ``squared_up`` belongs to hits and is
    negatively signed on home runs, so it is reported beside this rather than
    folded into it. An unmeasured swing returns ``UNMEASURED`` and changes
    nothing: stage one stands on its own.
    """
    if prof is None:
        return UNMEASURED
    pz = prof.power_z
    if pz != pz:
        return UNMEASURED
    good = pz >= min_power_z
    if luck_gap >= 0:
        return CONTRADICTED if good else CONFIRMED
    return CONFIRMED if good else CONTRADICTED


def swings_of(rows: pd.DataFrame) -> pd.DataFrame:
    """The tracked competitive swings in a pitch-level slice, oldest first.

    Rows without a bat-speed reading are dropped rather than imputed: a swing the
    tracker missed is not a slow swing, and the windows below count tracked
    swings, so an untracked one must not consume one.
    """
    if rows.empty or "bat_speed" not in rows or "description" not in rows:
        return rows.iloc[0:0]
    sw = rows[
        rows["description"].isin(SWING_DESC)
        & rows["bat_speed"].notna()
        & (rows["bat_speed"] >= MIN_TRACKED_MPH)
    ]
    if "game_date" not in sw:
        return sw
    return sw.assign(_d=pd.to_datetime(sw["game_date"])).sort_values("_d").drop(columns="_d")


def _level(metric: str, values: pd.Series) -> float:
    """``metric``'s mean over its own window, ``nan`` below its r=.50 floor."""
    clean = values.dropna()
    if len(clean) < SWINGS_FOR_READABLE[metric]:
        return math.nan
    return float(clean.tail(WINDOW[metric]).mean())


def build_swing_profile(rows: pd.DataFrame) -> SwingProfile:
    """Read a hitter's swing off his pitch-level slice.

    Each level is the mean over that metric's own window of his most recent
    tracked swings -- ``WINDOW`` -- so bat speed is read on the last sixteen and
    squared-up rate on the last two hundred and eighty: the sample each of them
    needs, and no more, which is the same rule the scored metrics follow in
    ``features.reliability``.

    Deliberately not split by the hand the hitter will face. The panel that
    validated these levels pooled hands, and splitting would halve every window
    when the two contact rates already need the larger half of a season's swings.
    """
    sw = swings_of(rows)
    if sw.empty:
        return SwingProfile(swings=0)
    bat = sw["bat_speed"].astype(float)
    ev = sw["launch_speed"].astype(float) if "launch_speed" in sw else pd.Series(
        math.nan, index=sw.index
    )
    pitch = sw["release_speed"].astype(float) if "release_speed" in sw else pd.Series(
        math.nan, index=sw.index
    )
    # Speed at the plate is a share of release speed; the collision model wants
    # the former and the feed publishes the latter.
    ratio = ev / (1.23 * bat + 0.23 * pitch * 0.915)
    squared = (ratio.fillna(0.0) >= SQUARED_UP_RATIO).astype(float)
    series = {
        "bat_speed": bat,
        "fast": (bat >= FAST_SWING_MPH).astype(float),
        "squared_up": squared,
        "blast": (squared.astype(bool) & (bat >= BLAST_BAT_SPEED)).astype(float),
        "swing_length": (
            sw["swing_length"].astype(float) if "swing_length" in sw else pd.Series(dtype=float)
        ),
    }
    lvl = {m: _level(m, v) for m, v in series.items()}
    return SwingProfile(
        swings=int(len(sw)),
        bat_speed=lvl["bat_speed"],
        fast=lvl["fast"],
        squared_up=lvl["squared_up"],
        blast=lvl["blast"],
        swing_length=lvl["swing_length"],
    )
