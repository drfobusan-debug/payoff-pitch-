"""What a starter's arm is, read off the pitch and not off the batted ball.

The pitcher regression report ranks arms on a luck term -- BABIP against .290
and xwOBA-allowed against wOBA-allowed. That is a statement about which balls in
play found grass, and like the hitter version in ``features.swing`` it says
nothing about the delivery that produced them. Statcast measures the delivery on
every pitch: release speed, extension, the release point, spin, and the induced
break. This module reads those the way ``features.swing`` reads bat tracking,
and it is the one place the reports, the screen and the article get them from.

**Reliability is not the binding constraint here, and that is the finding.**
Measured over 1.44M fastballs from 2025 and 2026 by
``scripts/measure_arm_reliability.py`` -- adjacent equal blocks of *n* pitches
per pitcher-season, block means correlated across arms:

    metric              1 pitch   10   100   400   r=.50 at
    release speed          .91    .96   .97   .97   1 pitch
    perceived velocity     .91    .96   .97   .97   1
    extension              .94    .98   .98   .98   1
    release point x/z      .97    .99   .99   .97   1
    arm angle              .96    .99   .99   .99   1
    spin rate              .78    .93   .97   .97   1
    induced vertical break .76    .94   .98   .98   1
    horizontal break       .65    .90   .96   .96   1

Every one of them half-repeats on a single pitch, because a release point or a
radar reading is a physical property of the arm measured directly, not a rate
estimated from outcomes: between-arm spread is 2.4 mph of velocity against 1.1
mph of pitch-to-pitch scatter. So the r=.50 rule that sized the hitter windows
(3 swings for bat speed, 184 for blast rate) cannot size these -- it would read
a level off one pitch. The window below is chosen from the out-of-time panel
instead, which is the only thing that can choose it.

**What survives out of time.** 2,214 pitcher-windows over 316 starters, eighteen
anchors across 2025 and 2026, predictors read strictly before the anchor and the
target the fortnight after it, errors clustered on the pitcher. Stage one is
what the report already ranks -- trailing wOBA-allowed, xwOBA-allowed and BABIP:

    stage one alone            wOBA/PA   K/PA    H/PA    TB/PA
      xwOBA-allowed            t +3.5   t -6.7  t +3.9  t +3.4
      wOBA-allowed             t +0.2   t -1.1  t -0.9  t -0.5
      BABIP-allowed            t -0.1   t +2.3  t +0.1  t -0.0

Only the expected half of the luck term predicts anything; the results half and
the BABIP the report ranks on are indistinguishable from zero. Added on top of
that, and then on top of the CSW% and the pitch-shape grade (#219) already in
production, so nothing here is a second copy of a signal the engine has:

    level added on stage one + CSW + shape   wOBA/PA   K/PA    H/PA    HR/PA
      perceived velocity                     t -2.4   t +4.4  t -3.6  t -0.4
      release speed                          t -2.4   t +4.2  t -3.6  t -0.9
      induced vertical break                 t -1.6   t +3.9  t -3.0  t +5.0
      spin rate                              t -0.1   t +3.0  t -1.0  t +2.9
      extension                              t +0.1   t +0.7  t +0.2  t +2.2
      release point x / z, arm angle         t <1 everywhere
      horizontal break                       t +1.4   t -0.7  t +1.9  t -0.3
      release scatter                        t +0.1   t -1.0  t -1.4  t -0.2

Perceived velocity -- ``release_speed + 1.1 x release_extension - 6.0``, the
speed the hitter has to react to -- is the strongest single read and beats raw
velocity on every market, which is why extension is folded into it rather than
scored beside it. Induced vertical break is the arm's version of the hitter's
attack angle: it is the sharpest home-run read on the board (t +5.0) while
*subtracting* hits (t -3.0), so ride suppresses singles and pays for itself in
the seats. Together on top of stage one, CSW and shape they add dR2 +.016 to
strikeouts, +.013 to home runs and +.008 to hits.

The release point, the arm angle, the horizontal break and the release scatter
earn nothing anywhere. Scatter in particular is not a talent level -- within-game
release variance is a fatigue read and belongs to the removal model, not here.

**The deltas earn nothing, as on the hitter side.** Recent block against the
immediately preceding one, on top of stage one, target wOBA/PA: velocity t -1.0,
perceived velocity t -0.9, extension t +0.4, spin t -0.9, break t +1.3, arm angle
t +0.9, scatter t -1.1. That is the fourth trend this engine has tested and
refused, after the barrel trend (#109), the CSW trend (#147) and the swing
trends. No trend is exposed here on purpose.

**And it is a level, not a rescue.** Crossing the luck flag with the arm read,
the interaction is t -1.45 and the arm sorts the next fortnight by the same .022
wOBA in every cell -- hot results (.3188 strong arm against .3330 weak), cold
results (.3111 against .3339), neither (.3113 against .3337). Unlike the hitter
blast-rate rescue, a good arm inside the flagged rows is not specifically what
the flag gets wrong; a good arm is better everywhere. So the verdict here
qualifies the luck read rather than overturning a cut.

**What the verdict is worth, both signs of it.** The same panel, cells graded on
the fortnight after the anchor, intervals from 3,000 bootstrap resamples
clustered on the pitcher. Results ran hot means the report calls for a fade;
ran cold with a high BABIP means it calls for a bounce-back:

    cell                                   n    nxt wOBA   K/PA    TB/PA
      ran hot, below-league arm  CONFIRMED  148    .3382   .1865   .4031
      ran hot, above-league arm  CONTRAD.   177    .3155   .2359   .3653
      ran cold, above-league arm CONFIRMED  188    .3141   .2398   .3473
      ran cold, below-league arm CONTRAD.   196    .3361   .1977   .3923
      no flag either way (base rate)       1505    .3218   .2229   .3694

Within the fade rows the delivery is worth +.0225 of wOBA allowed
[+.0048, +.0402], -.0493 of strikeouts [-.0675, -.0315] and +.0384 of total
bases [+.0115, +.0654]; within the bounce-back rows an intact arm is worth
-.0217 of wOBA [-.0366, -.0068]. Both hold in 2025 and 2026 separately. So a
confirmed verdict is not decoration: a confirmed fade lands 16 points of wOBA
*above* the unflagged base rate and a confirmed bounce-back 8 points below it,
while a contradicted flag lands on the wrong side of the base rate in both
directions -- which is the four-way read stage one cannot make on its own.

**The confirmation has to be the level, and never the trend.** Read as a rising
or falling delivery it evaporates: inside the fade rows, falling perceived
velocity minus rising is +.0062 of wOBA [-.0112, +.0242], and once the level is
already in the cell it adds +.0075 [-.0177, +.0332]; inside the bounce-back rows
an intact *trend* is worth -.0105 [-.0247, +.0045]. Every one of those crosses
zero, which is why no delta is exposed anywhere in this module and why a starter
losing a mile off his fastball cannot flip a verdict here -- only where he sits
against the league can. ``scripts/arm_stage_study.py --cells`` rebuilds all of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

#: Reliability at which a measure may carry a decision by itself.
READABLE_R = 0.50

#: Fastball families. A velocity or a release point is only comparable inside a
#: pitch family, and the fastball is the one every starter throws.
FASTBALL_TYPES = ("FF", "FA", "SI", "FT")

#: Measured split-half reliability: metric -> ((pitches, r), ...), ascending.
CURVES: dict[str, tuple[tuple[int, float], ...]] = {
    "velo": ((1, 0.911), (10, 0.960), (100, 0.974), (400, 0.973)),
    "pvelo": ((1, 0.913), (10, 0.960), (100, 0.974), (400, 0.974)),
    "ext": ((1, 0.936), (10, 0.976), (100, 0.984), (400, 0.984)),
    "rel_x": ((1, 0.970), (10, 0.987), (100, 0.983), (400, 0.965)),
    "rel_z": ((1, 0.976), (10, 0.990), (100, 0.992), (400, 0.988)),
    "spin": ((1, 0.782), (10, 0.930), (100, 0.966), (400, 0.966)),
    "ivb": ((1, 0.756), (10, 0.944), (100, 0.979), (400, 0.980)),
    "hb": ((1, 0.648), (10, 0.904), (100, 0.959), (400, 0.961)),
}

#: League level and spread of each metric read on ``WINDOW`` fastballs, over the
#: same 2,214 pitcher-windows the panel used, so a z-score compares an arm with
#: the league measured the way he was.
LEAGUE: dict[str, tuple[float, float]] = {
    "velo": (94.13, 2.20),
    "pvelo": (95.27, 2.29),
    "ext": (6.49, 0.41),
    "rel_x": (1.89, 0.67),
    "rel_z": (5.80, 0.45),
    "spin": (2266.3, 145.3),
    "ivb": (13.04, 3.92),
    "hb": (10.35, 3.55),
    "scatter": (2.26, 0.84),
}

#: Fastballs a level is read over. Not derived from the reliability curves: they
#: saturate at one pitch, so they cannot size a window. The panel was built three
#: times, on 12, 100 and 400 fastballs, and every sign and significance held --
#: perceived velocity on the next fortnight's wOBA came in at t -6.4, -6.4 and
#: -3.8 -- so the middle one is used, a start and a half of fastballs.
WINDOW = 100

#: Fewer than this and no level is read at all, so a thin arm is unmeasured
#: rather than league average. Two of the smallest window the panel validated.
MIN_LEVEL_PITCHES = 24


def _crossing(curve: tuple[tuple[int, float], ...], target: float) -> float:
    for (n0, r0), (n1, r1) in zip(curve, curve[1:], strict=False):
        if r0 < target <= r1:
            return n0 + (target - r0) / (r1 - r0) * (n1 - n0)
    if curve and curve[0][1] >= target:
        return float(curve[0][0])
    return math.inf


#: Metric -> fastballs at which it first reaches ``READABLE_R``. One, for all of
#: them, which is the point: the constraint on an arm read is the outcome sample,
#: never the measurement.
PITCHES_FOR_READABLE: dict[str, float] = {m: _crossing(c, READABLE_R) for m, c in CURVES.items()}


def reliability(metric: str, pitches: float) -> float:
    """Split-half reliability of ``metric`` at ``pitches`` fastballs.

    Interpolated in log pitches; outside the measured grid the nearest endpoint
    holds rather than extrapolating. An unmeasured metric returns 1.0.
    """
    curve = CURVES.get(metric)
    if curve is None:
        return 1.0
    if pitches <= curve[0][0]:
        return curve[0][1]
    if pitches >= curve[-1][0]:
        return curve[-1][1]
    for (n0, r0), (n1, r1) in zip(curve, curve[1:], strict=False):
        if n0 <= pitches <= n1:
            span = math.log(n1) - math.log(n0)
            frac = (math.log(pitches) - math.log(n0)) / span if span else 0.0
            return r0 + frac * (r1 - r0)
    return curve[-1][1]


@dataclass
class ArmProfile:
    """One starter's delivery, each measure read over its last ``WINDOW`` fastballs.

    ``pitches`` is what the slice carried; a level is ``nan`` when the slice
    never reached ``MIN_LEVEL_PITCHES`` or when the feed carried no such column,
    so an arm we cannot see reads as unmeasured and never as average.
    """

    pitches: int
    velo: float = math.nan
    pvelo: float = math.nan
    ext: float = math.nan
    rel_x: float = math.nan
    rel_z: float = math.nan
    spin: float = math.nan
    ivb: float = math.nan
    hb: float = math.nan
    scatter: float = math.nan

    def levels(self) -> dict[str, float]:
        """Metric -> level, ``nan`` where the slice could not be read."""
        return {
            "velo": self.velo,
            "pvelo": self.pvelo,
            "ext": self.ext,
            "rel_x": self.rel_x,
            "rel_z": self.rel_z,
            "spin": self.spin,
            "ivb": self.ivb,
            "hb": self.hb,
            "scatter": self.scatter,
        }

    def z(self, metric: str) -> float:
        """The level in league standard deviations, ``nan`` when unmeasured."""
        val = self.levels().get(metric, math.nan)
        mu, sd = LEAGUE.get(metric, (math.nan, math.nan))
        if val != val or not sd:
            return math.nan
        return (val - mu) / sd

    @property
    def stuff_z(self) -> float:
        """Perceived velocity: the one level that survives CSW% and the shape grade.

        Raw velocity is not averaged in beside it. The two correlate at r +.99 by
        construction -- perceived velocity *is* velocity plus extension -- and
        perceived is the stronger of the two on every market the panel tested, so
        pooling them would only dilute it.
        """
        return self.z("pvelo")

    @property
    def ride_z(self) -> float:
        """Induced vertical break: home runs up, hits down, on the same pitch.

        Deliberately not folded into ``stuff_z``. Ride is the arm's attack angle:
        it is the sharpest home-run read here (t +5.0) while suppressing hits
        (t -3.0), and one number carrying both cancels both.
        """
        return self.z("ivb")


#: Verdicts of the second stage, once the first has read the luck term.
CONFIRMED = "confirmed"  # the arm agrees with what the luck term says is coming
CONTRADICTED = "contradicted"  # the arm says the luck read is about to be wrong
UNMEASURED = "unmeasured"  # too few fastballs to read a level at all


def stage_two(luck_gap: float, prof: ArmProfile | None, *, min_stuff_z: float = 0.0) -> str:
    """Confirm or contradict a starter's luck term with the delivery underneath it.

    ``luck_gap`` is xwOBA-allowed minus wOBA-allowed, positive when the contact
    he has allowed deserved *worse* than the line he posted -- the results ran
    hot and are due to correct upward. Crossed with the arm:

    * results ran hot and a below-league arm -> ``CONFIRMED``, nothing is holding
      the run prevention up;
    * results ran hot and an above-league arm -> ``CONTRADICTED``, the fortnight
      was lucky and the pitcher is also good;
    * results ran cold and an above-league arm -> ``CONFIRMED``, the rebound has
      a delivery behind it;
    * results ran cold and a below-league arm -> ``CONTRADICTED``, the ugly line
      is the level rather than the luck.

    All four cells are graded in the module docstring: out of time the confirmed
    pair lands on the far side of the unflagged base rate and the contradicted
    pair on the wrong side of it, which is what makes the verdict worth printing.
    The cut is where the arm sits against the league and never which way it is
    moving -- the trend form of every cell crosses zero.

    Read on ``stuff_z`` alone. Ride is reported beside this and not folded in: it
    points at home runs and against hits, so it cannot vote on a single verdict.
    An unmeasured arm returns ``UNMEASURED`` and changes nothing -- stage one
    stands on its own, which is also what the panel says it should do, since the
    arm read is a level everywhere rather than a rescue inside the flagged rows.
    """
    if prof is None:
        return UNMEASURED
    sz = prof.stuff_z
    if sz != sz:
        return UNMEASURED
    good = sz >= min_stuff_z
    if luck_gap >= 0:
        return CONTRADICTED if good else CONFIRMED
    return CONFIRMED if good else CONTRADICTED


def fastballs_of(rows: pd.DataFrame) -> pd.DataFrame:
    """The fastballs in a pitch-level slice, oldest first.

    Rows the tracker missed are dropped rather than imputed, and the slice is not
    split by the hand at the plate: the delivery is the pitcher's, and the panel
    that validated these levels pooled the two.
    """
    if rows.empty or "pitch_type" not in rows:
        return rows.iloc[0:0]
    fb = rows[rows["pitch_type"].isin(FASTBALL_TYPES)]
    if "game_date" not in fb:
        return fb
    return fb.assign(_d=pd.to_datetime(fb["game_date"])).sort_values("_d").drop(columns="_d")


def _series(rows: pd.DataFrame, column: str) -> pd.Series:
    """``column`` as floats, or an empty series when the feed never carried it."""
    if column not in rows:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _level(values: pd.Series) -> float:
    """The mean over the last ``WINDOW`` readings, ``nan`` below the floor."""
    clean = values.dropna()
    if len(clean) < MIN_LEVEL_PITCHES:
        return math.nan
    return float(clean.tail(WINDOW).mean())


def build_arm_profile(rows: pd.DataFrame) -> ArmProfile:
    """Read a starter's delivery off his pitch-level slice.

    Every level is the mean over his last ``WINDOW`` fastballs. Release point is
    mirrored so that the arm side is positive for either hand, which is what
    makes a left-hander and a right-hander comparable; horizontal break is
    mirrored with it. ``pfx_x`` arrived in the feed all along and our own
    ingestion dropped it until now, so a frame cached earlier carries no column:
    absent and unmeasured, never imputed.
    """
    fb = fastballs_of(rows)
    if fb.empty:
        return ArmProfile(pitches=0)
    hand = (
        fb["p_throws"].map({"L": -1.0, "R": 1.0}).fillna(1.0)
        if "p_throws" in fb
        else pd.Series(1.0, index=fb.index)
    )
    velo = _series(fb, "release_speed")
    ext = _series(fb, "release_extension")
    rel_x = _series(fb, "release_pos_x") * -hand
    rel_z = _series(fb, "release_pos_z")
    scatter = (
        float(math.hypot(float(rel_x.tail(WINDOW).std()), float(rel_z.tail(WINDOW).std())) * 12.0)
        if len(rel_x.dropna()) >= MIN_LEVEL_PITCHES
        else math.nan
    )
    return ArmProfile(
        pitches=int(len(fb)),
        velo=_level(velo),
        pvelo=_level(velo + 1.1 * ext - 6.0),
        ext=_level(ext),
        rel_x=_level(rel_x),
        rel_z=_level(rel_z),
        spin=_level(_series(fb, "release_spin_rate")),
        ivb=_level(_series(fb, "pfx_z") * 12.0),
        hb=_level(_series(fb, "pfx_x") * 12.0 * -hand),
        scatter=scatter,
    )
