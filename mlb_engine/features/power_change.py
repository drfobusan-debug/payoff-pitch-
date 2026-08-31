"""Peak power and the fastball whiff, each read over the window it stabilises on.

Two things the batter regression article was missing. **Max exit velocity** is
the ceiling of a hitter's contact -- the one batted ball that says what he is
capable of rather than what he averaged -- and **fastball whiff%** is whether he
can catch up to velocity, which is the half of the strikeout risk that a matchup
against a hard thrower actually turns on. Neither is scored anywhere; both are
levels the article quotes beside the luck gap.

**The window is measured, not chosen.** ``features.reliability`` sizes both by
split-half over 162,464 plate appearances and 1,397 hitters, 2026 through 8/28:
fastball whiff% half-repeats at 51 PA and reaches r=.70 at 108, max EV at 49 and
195. Levels are therefore read over the most recent ``PA_FOR_STABLE`` plate
appearances the hitter has, and a hitter under ``PA_FOR_READABLE`` gets no read
at all rather than a number the sample cannot support. The two windows differ by
a factor of two, so reading them over one shared look-back would either waste
half of the fastball evidence or print a max EV the sample has not earned.

**The change is a different question, and the answer is no.** Measured by
``scripts/measure_change_reliability.py`` on the same window -- adjacent equal
blocks of plate appearances per hitter, the block after them held out:

    block   hitters   noise sd   adjacent sd   true-change sd   t level   t move
    max EV
       25      476       3.15        3.21           0.63          +8.3     -0.1
       40      424       2.40        2.35           0.00         +13.2     -1.1
       60      359       2.02        1.99           0.00         +14.9     +0.8
       90      268       1.89        1.88           0.00         +15.6     -2.0
      130      171       1.79        1.85           0.48         +14.3     +0.3
    fastball whiff%
       25      476        .079        .087           .036        +10.1     +0.6
       40      424        .060        .067           .029        +10.6     +0.6
       60      359        .051        .059           .031        +13.1     +0.5
       90      268        .041        .044           .016        +16.4     +0.1
      130      171        .037        .036           .000        +14.8     +0.2

Noise sd is the same block split at random and differenced -- two reads of a
hitter who by construction did not change. Adjacent sd is the real move plus that
noise. For max EV the two are the same number at every block size: **there is no
detectable true change in a hitter's ceiling inside a season**, which is what you
would expect of a maximum, since a single new best batted ball moves it and
nothing moves it back. Fastball whiff% does carry a little real movement (true sd
.016--.036 against noise of .04--.08), and neither move predicts the next block
once the level is in the regression -- max EV t -2.0 to +0.8 with no stable sign,
fastball whiff +0.1 to +0.6, against the level's +8 to +16.

So the change is printed as a diagnostic with the band it has to clear, never as
a signal: ``BANDS`` is 1.96 standard deviations of a no-change delta at each
measured block size, and ``moved`` is false inside it. At the block the article
reads a move over -- fifty-two plate appearances, the narrowest sample either
level half-repeats at -- that band is about **6 mph of max EV** and **19 points of
fastball whiff%**, so a hitter "up 4 mph" has shown less than two identical
halves of his own season routinely differ by, and the article says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from mlb_engine.data.statcast import batted_balls
from mlb_engine.features.regression import SWING_DESC, WHIFF_DESC
from mlb_engine.features.reliability import PA_FOR_READABLE, PA_FOR_STABLE

#: Savant codes a hitter reads as one pitch. A cutter is thrown at fastball
#: velocity and belongs with the four-seam and the sinker; a splitter does not.
FASTBALL_CODES = frozenset({"FF", "FA", "SI", "FT", "FC"})

#: A plate appearance is keyed the way the reliability measurement keys it, which
#: is the finest grain the engine's Statcast cache supports.
PA_KEY = ("game_date", "inning", "inning_topbot")

#: Plate appearances a level is read over, and the floor below which it is not
#: read at all. Both come from the measured curves rather than from taste.
WINDOW: dict[str, int] = {
    m: int(math.ceil(PA_FOR_STABLE[m])) for m in ("max_ev", "fb_whiff")
}
FLOOR: dict[str, int] = {
    m: int(math.ceil(PA_FOR_READABLE[m])) for m in ("max_ev", "fb_whiff")
}

#: Plate appearances per block when the move is read. The level is quoted over
#: the wider window it stabilises on, but a move needs *two* blocks off the same
#: frame, so it is read on the narrowest sample the level half-repeats at -- the
#: floor. Reading the move over the stable window instead would need twice that
#: window and no hitter has it inside a look-back the article uses.
MOVE_BLOCK: dict[str, int] = dict(FLOOR)

#: 1.96 standard deviations of the delta between two equal blocks of a hitter who
#: did not change: metric -> ((block PA, band), ...) ascending. Measured, so a
#: move is compared against the noise of the sample it was read on.
BANDS: dict[str, tuple[tuple[int, float], ...]] = {
    "max_ev": ((25, 8.73), (40, 6.65), (60, 5.59), (90, 5.24), (130, 4.96)),
    "fb_whiff": ((25, 0.22), (40, 0.17), (60, 0.14), (90, 0.11), (130, 0.10)),
}

#: League baselines over the same 2026 window: the mean of the per-hitter max EV
#: among hitters with 50+ batted balls, and the league fastball whiff rate over
#: 91,024 fastball swings.
BL_MAX_EV = 109.4
BL_FB_WHIFF = 0.196

#: Fastball swings a whiff rate needs before it is a rate at all, on top of the
#: plate-appearance floor: a hitter can take a hundred plate appearances without
#: offering at many fastballs.
MIN_FB_SWINGS = 25


def band(metric: str, block: int) -> float:
    """The no-change band for ``metric`` at a block of ``block`` plate appearances."""
    grid = BANDS[metric]
    if block <= grid[0][0]:
        return grid[0][1]
    if block >= grid[-1][0]:
        return grid[-1][1]
    for (n0, b0), (n1, b1) in zip(grid, grid[1:], strict=False):
        if n0 <= block <= n1:
            span = math.log(n1) - math.log(n0)
            frac = (math.log(block) - math.log(n0)) / span if span else 0.0
            return b0 + frac * (b1 - b0)
    return grid[-1][1]


@dataclass(frozen=True)
class PowerChange:
    """Both levels over their own windows, and the move against its own noise."""

    pa: int
    #: Levels over the window each metric stabilises on.
    max_ev: float = math.nan
    max_ev_pa: int = 0
    fb_whiff: float = math.nan
    fb_whiff_pa: int = 0
    fb_swings: int = 0
    #: The same two read over adjacent equal blocks, recent against prior.
    max_ev_recent: float = math.nan
    max_ev_prior: float = math.nan
    fb_whiff_recent: float = math.nan
    fb_whiff_prior: float = math.nan
    block_pa: int = 0

    @property
    def d_max_ev(self) -> float:
        return self.max_ev_recent - self.max_ev_prior

    @property
    def d_fb_whiff(self) -> float:
        return self.fb_whiff_recent - self.fb_whiff_prior

    def delta(self, metric: str) -> float:
        return self.d_max_ev if metric == "max_ev" else self.d_fb_whiff

    def moved(self, metric: str) -> bool:
        """Whether the move clears the band two no-change reads would produce."""
        delta = self.delta(metric)
        if delta != delta:
            return False
        return abs(delta) > band(metric, self.block_pa)

    def stable(self, metric: str) -> bool:
        """Whether the level was read over the full window it stabilises at."""
        read = self.max_ev_pa if metric == "max_ev" else self.fb_whiff_pa
        return read >= WINDOW[metric]


def _pa_ordinal(rows: pd.DataFrame) -> pd.Series:
    """Each pitch labelled with the chronological number of its plate appearance."""
    ends = rows[rows["events"].notna()]
    if ends.empty:
        return pd.Series(dtype="float64", index=rows.index)
    keys = ends.assign(_d=pd.to_datetime(ends["game_date"]))
    keys = keys.sort_values(["_d", "inning"], kind="stable")
    ordered = zip(*(keys[c] for c in PA_KEY), strict=True)
    order = {k: i for i, k in enumerate(dict.fromkeys(ordered))}
    mine = list(zip(*(rows[c] for c in PA_KEY), strict=True))
    return pd.Series(mine, index=rows.index).map(order)


def _max_ev(rows: pd.DataFrame) -> float:
    bip = batted_balls(rows)
    speeds = pd.to_numeric(bip["launch_speed"], errors="coerce").dropna() if len(bip) else None
    return float(speeds.max()) if speeds is not None and len(speeds) else math.nan


def _fb_whiff(rows: pd.DataFrame) -> tuple[float, int]:
    if "pitch_type" not in rows or "description" not in rows:
        return math.nan, 0
    fb = rows[rows["pitch_type"].isin(FASTBALL_CODES)]
    swings = fb[fb["description"].isin(SWING_DESC)]
    n = len(swings)
    if n < MIN_FB_SWINGS:
        return math.nan, n
    return float(swings["description"].isin(WHIFF_DESC).mean()), n


def build_power_change(rows: pd.DataFrame) -> PowerChange:
    """Read both metrics off one hitter's pitches, each over its own window.

    The recent window is the most recent ``WINDOW`` plate appearances and the
    prior block is the ``WINDOW`` before it, so the two are the same size and the
    difference between them is not a difference in sample. A hitter short of
    ``FLOOR`` gets ``nan`` for the level and a hitter short of two blocks gets
    ``nan`` for the move: an unmeasured read is printed as unmeasured rather than
    filled in from the league.
    """
    if rows.empty or "events" not in rows:
        return PowerChange(pa=0)
    ordinal = _pa_ordinal(rows)
    total = int(ordinal.max()) + 1 if len(ordinal.dropna()) else 0
    if not total:
        return PowerChange(pa=0)

    fields: dict[str, float | int] = {"pa": total}
    for metric in ("max_ev", "fb_whiff"):
        window = min(WINDOW[metric], total)
        if window < FLOOR[metric]:
            continue
        level = rows[ordinal >= total - window]
        if metric == "max_ev":
            fields["max_ev"] = _max_ev(level)
            fields["max_ev_pa"] = window
        else:
            rate, swings = _fb_whiff(level)
            fields["fb_whiff"] = rate
            fields["fb_whiff_pa"] = window
            fields["fb_swings"] = swings

    block = max(MOVE_BLOCK.values())
    if total >= 2 * block:
        recent = rows[ordinal >= total - block]
        prior = rows[(ordinal >= total - 2 * block) & (ordinal < total - block)]
        fields["block_pa"] = block
        fields["max_ev_recent"] = _max_ev(recent)
        fields["max_ev_prior"] = _max_ev(prior)
        fields["fb_whiff_recent"] = _fb_whiff(recent)[0]
        fields["fb_whiff_prior"] = _fb_whiff(prior)[0]
    return PowerChange(**fields)  # type: ignore[arg-type]
