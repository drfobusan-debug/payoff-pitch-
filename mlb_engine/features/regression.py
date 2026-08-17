"""Statcast regression signals mapped to per-outcome multipliers.

Implements the sensitivity / PPV / NPV framework the user specified:

  HR   : bat speed & max EV (sensitive), barrel rate (PPV), hard-hit% / LA (NPV)
  XBH  : sweet-spot% (sensitive), xSLG (PPV), blast/bat-speed (NPV)
  1B   : whiff% / zone-contact% (sensitive), xBA + sprint speed (PPV),
         barrel rate (NPV -- power turns singles into extra-base hits)
  All  : BABIP and dxwOBA (xwOBA - wOBA) luck-regression signals.

Each raw metric is compared to a league baseline and squashed into a bounded
multiplier so no single signal dominates. Multipliers are applied to the matchup
outcome probabilities before simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import SupportsFloat, cast

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import batted_balls
from mlb_engine.features.xtb import NON_AB_EVENTS, TB_VALUE, LeagueXTB


def _safe_float(val: object, default: float) -> float:
    """Coerce a scalar to float, falling back when it is NA/NaN/None.

    Statcast columns arrive as pandas *nullable* dtypes, so an all-NA slice's
    ``.mean()`` / ``.max()`` returns ``pd.NA`` (not ``nan``), and ``float(pd.NA)``
    raises ``TypeError``. This guards every such reduction.
    """
    if val is None or pd.isna(val):
        return default
    return float(cast(SupportsFloat, val))


# League baselines (approximate, recalibratable).
BL_BARREL = 0.080
BL_HARD_HIT = 0.400
BL_SWEET_SPOT = 0.330
BL_BAT_SPEED = 71.5
BL_MAX_EV = 108.0
BL_WHIFF = 0.240
BL_ZONE_CONTACT = 0.820
BL_XBA = 0.250
BL_XSLG = 0.400
# Mean exit velocity on fly balls + line drives. Higher than the all-batted-ball
# mean because ground balls (which carry no home-run information) are excluded.
BL_FB_LD_EV = 92.5
# Infield-fly rate: pop-ups as a share of all fly balls (Statcast ``popup`` /
# (``popup`` + ``fly_ball``)).
BL_IFFB = 0.100
# Pulled-air rate: batted balls that are both pulled and in the air (LA >= 10),
# as a share of all batted balls.
BL_PULL_AIR = 0.220
# Barrels per plate appearance -- league barrel rate (~8% of batted balls) times
# the share of PAs that end in a batted ball (~0.62).
BL_BARREL_PA = 0.050
BL_BABIP = 0.290
BL_SPRINT = 27.0
BL_GB_RATE = 0.420

MIN_BBE = 15  # minimum batted-ball events for a stable signal

# A bullpen is not a small-sample pitcher. Pooled over 42 days a pen puts ~1,240
# batted balls on the board against a starter's ~100, so the spread between pens
# is mostly talent where the spread between starters is mostly noise -- which
# inverts how its contact profile should be read. See ``allowed_multipliers``.
BL_PEN_XWOBA = 0.311  # league bullpen xwOBA allowed on contact
PEN_XWOBA_SLOPE = 1.70  # fitted on hits per relief PA, t +4.5
PEN_XWOBA_CLIP = (-0.06, 0.06)

# Extra-base multiplier per ft/s of sprint speed, and the slow-bat conjunction.
# 26.5 ft/s is roughly the bottom third of the league; .060 2B+3B/PA is the top
# quarter, so the pair reads "can't run, yet the doubles are pouring in".
SPRINT_XBH_SLOPE = 0.025
SLOW_SPRINT = 26.5
XBH_SURGE = 0.060

# Singles fall as barrel rate rises: across 323 qualified batters (95k PA) the
# league goes from .175 singles/PA under 2% barrel to .124 at 10-12%, a slope of
# roughly -0.5 singles/PA per unit barrel -- about -3.5 in relative terms.
#
# Only the residual half of that belongs here. Regressing out K/PA drops the
# correlation from -.464 to -.207, i.e. half the effect is strikeouts, which the
# simulator already carries in each batter's own empirical K rate. The other half
# is hit conversion: a barrel-heavy hitter's share of hits that are singles falls
# from .756 to .568 while his total hits barely move. This slope prices that half
# only; the full -3.5 would charge for the strikeouts twice.
SINGLES_BARREL_SLOPE = 1.5

# Batted-ball mix on the singles line. A single is overwhelmingly a ground ball
# or a line drive -- measured over this cache, 45.6% of singles are ground balls
# and 48.7% are line drives, leaving 5.7% to the air. The two arrive there by
# opposite routes: line drives are hit less often (23.7% of contact) but land for
# a hit .623 of the time, while ground balls are hit far more often (42.4%) and
# land only .245 of the time -- but 91.4% of the ground balls that do land are
# singles, against 68.6% of line drives.
#
# Both slopes are fitted **out of time**, which matters more here than anywhere
# else in this file: features from a 42-day window predicting singles per PA over
# the *following* 21 days, stacked over four rolls (n=862 batter-windows),
# controlling for xBA, K% and sprint speed, with air contact as the omitted
# category. Fitting a rate against outcomes from the same window is circular --
# a line drive that falls in for a single raises both sides of the regression at
# once -- and it inflates these slopes by roughly a third.
#
#                same-window   OUT OF TIME    p (out of time)
#     GB%           +1.02         +0.86         <1e-4
#     LD%           +1.43         +1.23         <1e-4
#
# Both survive, which is the important result, and the line drive stays worth
# more per unit. Note the contrast with the home-run line, where GB% is a brake:
# the batted ball that cannot clear a fence is the one most likely to fall in for
# a single.
#
# One caveat on the line-drive term specifically. Line-drive rate is the least
# stable input in this file -- split-half reliability .21 over a 42-day window,
# against .68 for ground-ball rate -- so a hot line-drive stretch is mostly
# noise. The out-of-time design already charges the term for that (its slope
# falls further than GB's), but it is the more fragile of the two.
# Fitted jointly with the shape terms below, which share variance with them --
# hence smaller than the +0.86/+1.23 they take alone.
SINGLES_GB_SLOPE = 0.47
SINGLES_LD_SLOPE = 0.87

# Shape *within* those batted-ball classes. Same out-of-time design, same panel:
#
#     input          slope    p        what it says
#     GB pull%       -0.25   .046     the rollover, into a defence aligned for it
#     LD soft%       +0.25   .013     the flare that lands in front of the outfield
#     LD oppo+mid%   +0.22   .024     the line drive an outfielder can cut off
#
# Pulled grounders are a *negative*: a pulled ground ball is the hardest-hit and
# most-defended one, and the pull-side slugger rolling over an offspeed pitch
# outnumbers the speedster slashing one past a deep third baseman -- pull x
# sprint speed and pull x exit velocity were both tested as explicit
# interactions to separate the two archetypes and neither came close (p=.38,
# p=.36). The same logic runs the other way on line drives: the ones that stay
# singles are the soft, oppo, cut-off ones, because the hard pulled line drive
# is a double.
#
# Two candidates from the same family were tested and rejected, both for the
# same reason -- they do not persist for a hitter across windows:
#
#     GB BABIP          p=.91,  split-half reliability .14
#     LD launch angle   p=.41,  split-half reliability .13
#
# Ball-level, both look decisive: line drives at 10-15 degrees are singles .627
# of the time against .223 above 20 degrees. But a hitter's *rate* of them is
# almost pure noise over 42 days, so there is nothing to select on. GB BABIP is
# the clearest trap of the group: it is the natural sorting metric and it adds
# exactly zero out of time.
SINGLES_GB_PULL_SLOPE = -0.25
SINGLES_LD_SOFT_SLOPE = 0.25
SINGLES_LD_OPPOMID_SLOPE = 0.22

# League line-drive rate, as a share of all batted balls, and league shape rates
# within each class. Measured back through ``build_batter_regression`` over 1,262
# batter-windows rather than taken from the fitting panel, so an average hitter
# reads exactly 1.0 on the quantities the engine actually computes at runtime.
BL_LD_RATE = 0.234
BL_GB_PULL = 0.663
BL_LD_SOFT = 0.403
BL_LD_OPPOMID = 0.526

# Batted balls of each class needed before its shape is read at all. Below these
# the rate is NaN and the term drops out rather than pricing a handful of balls.
MIN_GB_SHAPE = 15
MIN_LD_SHAPE = 12

# Home-run NPV thresholds. Each describes contact that structurally cannot leave
# the park, so a hitter past one of them is braked rather than nudged:
#   * a hitter who cannot drive air contact hard lacks the force to clear a
#     fence, however hard he hits his ground balls;
#   * above a 50% ground-ball rate there are too few fly balls to sustain home
#     runs, even with elite raw power;
#   * pop-ups are fly balls with too *much* launch angle -- functionally dead.
# Mean exit velocity on air contact below which a bat cannot drive the ball out.
# League median air EV is 89.4 mph over balls in play, so the old 90.0 floor sat
# *above* the median and penalised 55% of hitters (99.8% before fouls were
# excluded from the pool). 86.5 is the ~18th percentile, which matches the
# incidence of the air-hard-hit brake beside it.
FB_LD_EV_FLOOR = 86.5
GB_RATE_CEILING = 0.50  # ~86th percentile of batter ground-ball rate
# Popups as a share of fly balls. Statcast's ``bb_type`` calls 21.5% of fly balls
# popups league-wide -- roughly twice the FanGraphs IFFB% convention this ceiling
# was borrowed from -- so 0.15 sat at the 24th percentile and braked three
# hitters in four for being ordinary. 0.285 is the ~80th percentile, a real tail.
IFFB_CEILING = 0.285


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class BatterRegression:
    bbe: int
    barrel_rate: float
    hard_hit: float
    sweet_spot: float
    bat_speed: float
    max_ev: float
    whiff: float
    zone_contact: float
    xba: float
    xslg: float
    babip: float
    woba: float
    xwoba: float
    sprint_speed: float = BL_SPRINT
    k_pct: float = float("nan")  # season PA strikeout rate (NaN when no PAs)
    bb_pct: float = float("nan")  # season PA walk rate (NaN when no PAs)
    gb_rate: float = BL_GB_RATE
    # Ground-ball rate and pulled-air rate: stamped for HR/PPV backtesting.
    gb_pct: float = float("nan")
    pull_air_pct: float = float("nan")
    # Plate appearances behind the slice, so barrel rate can be expressed per PA.
    pa: int = 0
    # Contact quality restricted to balls hit in the air. An unfiltered max/mean
    # exit velocity can be set by a scorched ground ball, which carries no
    # home-run information; these isolate the contact that can actually leave
    # the park. NaN when the launch-angle data is unavailable.
    fb_ld_ev: float = float("nan")
    fb_ld_max_ev: float = float("nan")
    fb_ld_hard_hit: float = float("nan")
    # Pop-ups as a share of fly balls: high launch angle that dies on the infield.
    iffb_pct: float = float("nan")
    # Shape within the two singles-producing batted-ball classes. See
    # ``_singles_shape``. NaN when that class is too thin to read.
    gb_pull_pct: float = float("nan")
    ld_soft_pct: float = float("nan")
    ld_oppomid_pct: float = float("nan")
    # Actual slugging over the slice, alongside doubles+triples per PA and line
    # drives per batted ball. Total bases is the numerator of slugging, so these
    # are the market's own surface stats: kept to measure the gap against the
    # expected versions rather than to score levels.
    slg: float = float("nan")
    xbh_per_pa: float = float("nan")
    ld_pct: float = float("nan")
    # Home total bases per PA over road, as a share of the road rate. Diagnostic:
    # the simulator already prices tonight's venue from the matching half of the
    # hitter's splits and multiplies by the park factor, so this is the read a
    # human wants beside those numbers -- "are his bases a ballpark artefact?" --
    # rather than a second adjustment. NaN when either half is too thin.
    tb_home_bias: float = float("nan")
    # Expected slugging *on contact*: the same launch parameters read per batted
    # ball, with no strikeout in the denominator. Runs high against real slugging
    # by construction; kept because the gap against it is what predicts decay.
    contact_slg: float = float("nan")

    @property
    def dxwoba(self) -> float:
        return self.xwoba - self.woba

    @property
    def slg_gap(self) -> float:
        """Slugging beyond what the hitter's per-ball contact quality supports.

        Total bases *is* the numerator of slugging, so this is the market's own
        luck term: bases collected off contact that does not carry them --
        bloops, misplays, wind. Measured against ``contact_slg`` rather than
        against ``xslg``, because netting strikeouts out of the expectation the
        way a calibrated xSLG does also nets out the discrepancy: the same
        threshold on the calibrated gap flags 556 batter-weeks that go on to
        produce +0.2% (p=.93), while this one flags 139 that give back 8.5%.
        Positive means over-performing; NaN when either side is unknown.
        """
        return self.slg - self.contact_slg

    @property
    def barrel_per_pa(self) -> float:
        """Barrels per plate appearance.

        Barrel rate per *batted ball* credits a hitter who barrels often but
        rarely puts the ball in play; per PA folds contact frequency in, which is
        what actually converts power into home runs. NaN when PAs are unknown.
        """
        if self.pa <= 0:
            return float("nan")
        return self.barrel_rate * self.bbe / self.pa

    @property
    def air_max_ev(self) -> float:
        """Max EV on air contact, falling back to all batted balls."""
        return self.fb_ld_max_ev if self.fb_ld_max_ev == self.fb_ld_max_ev else self.max_ev

    @property
    def air_hard_hit(self) -> float:
        """Hard-hit rate on air contact, falling back to all batted balls."""
        return (
            self.fb_ld_hard_hit
            if self.fb_ld_hard_hit == self.fb_ld_hard_hit
            else self.hard_hit
        )

    def multipliers(
        self,
        singles_barrel_slope: float = SINGLES_BARREL_SLOPE,
        singles_gb_slope: float = SINGLES_GB_SLOPE,
        singles_ld_slope: float = SINGLES_LD_SLOPE,
        singles_shape: bool = True,
    ) -> dict[str, float]:
        """Return bounded outcome multipliers for {1B,2B,3B,HR}.

        Any singles slope at 0 drops that term.
        """
        if self.bbe < MIN_BBE:
            return {}

        # --- Home runs ---
        # Barrel rate and max exit velocity are the two metrics that actually
        # separate true HRs from false positives in the graded backtest; max EV
        # was the single strongest separator yet historically carried one of the
        # smallest weights, so it is up-weighted here.
        hr = 1.0
        hr *= 1.0 + _clip((self.barrel_rate - BL_BARREL) * 2.5, -0.12, 0.15)  # PPV
        hr *= 1.0 + _clip((self.bat_speed - BL_BAT_SPEED) * 0.010, -0.06, 0.06)  # sensitive
        # Max EV over air contact only: an unfiltered max is often set by a
        # scorched ground ball, which cannot leave the park.
        hr *= 1.0 + _clip((self.air_max_ev - BL_MAX_EV) * 0.009, -0.06, 0.09)  # PPV
        # Pulled air contact has the shortest distance to the fence and the
        # highest HR conversion per batted ball.
        if self.pull_air_pct == self.pull_air_pct:  # not NaN
            hr *= 1.0 + _clip((self.pull_air_pct - BL_PULL_AIR) * 0.50, -0.05, 0.08)  # PPV
        if self.air_hard_hit < 0.30:  # NPV
            hr *= 0.80

        # --- Home-run NPV brakes ---
        # These describe contact that cannot become a home run, so they are
        # deliberately stronger than the PPV terms above: soft air contact and
        # ground balls are near-absolute negatives, not gentle tilts.
        if self.fb_ld_ev == self.fb_ld_ev and self.fb_ld_ev < FB_LD_EV_FLOOR:
            hr *= _clip(1.0 - (FB_LD_EV_FLOOR - self.fb_ld_ev) * 0.06, 0.70, 1.0)
        if self.gb_rate > GB_RATE_CEILING:
            hr *= _clip(1.0 - (self.gb_rate - GB_RATE_CEILING) * 2.0, 0.80, 1.0)
        if self.iffb_pct == self.iffb_pct and self.iffb_pct > IFFB_CEILING:
            hr *= _clip(1.0 - (self.iffb_pct - IFFB_CEILING) * 1.5, 0.85, 1.0)
        hr = _clip(hr, 0.50, 1.32)

        # --- Extra-base hits (2B/3B) ---
        # No contact-quality terms. There were three -- xSLG, sweet-spot rate and
        # bat speed, together worth up to 25% on a hitter's doubles rate -- and
        # none of them forward-predicts a double. Fitted out of time over 48,120
        # plate appearances in eight rolling blocks (features from a 42-day
        # window, outcome the games after it), each is indistinguishable from
        # zero, alone and beside the others:
        #
        #     sweet_spot  -0.024 (p=.26)    bat_speed  -0.014 (p=.51)
        #     xslg        -0.017 (p=.44)    hard_hit   +0.013 (p=.57)
        #     xwoba_contact -0.001 (p=.97)  gb_allowed -0.112 (p<.0001)
        #
        # and the signs are mostly *negative*: the top third of sweet-spot
        # hitters collected .0440 of a double per PA against the bottom third's
        # .0463. This is the same fact #129 measured from the other end -- a
        # hitter's doubles rate carries 2.8x more spread than his talent and
        # correlates 0.20 with it -- so contact quality read over six weeks is
        # sorting noise, and pricing it manufactures exactly the dispersion the
        # total-bases market has been losing money on.
        #
        # What survives is on the arm's side of the matchup (``gb_allowed``, in
        # ``allowed_multipliers``) and the runner's legs, below.
        xbh = 1.0
        # Sprint speed belongs here and nowhere else in the power stack: it turns
        # a single into a double and a gap shot into a triple, and does nothing
        # for a home run. Over 2,609 batter-weeks it predicts forward 2B+3B/PA
        # holding prior extra-base rate and contact quality fixed (beta +0.0031
        # per SD, p=.001); the slow third averages .0407 against the fast third's
        # .0483. Sized at half the measured slope, since 42 days of sprint speed
        # is a season-long tool being read on a short window.
        xbh *= 1.0 + _clip((self.sprint_speed - BL_SPRINT) * SPRINT_XBH_SLOPE, -0.07, 0.07)
        # A slow bat whose doubles have spiked is the third false positive: among
        # hitters over .060 2B+3B/PA, the slow third keeps .0420 of a .0778 rate
        # while the fast third keeps .0568 of .0766. The continuous term above
        # covers about half that spread, so the conjunction adds the rest.
        if self.sprint_speed < SLOW_SPRINT and self.xbh_per_pa > XBH_SURGE:
            xbh *= 0.96
        xbh = _clip(xbh, 0.82, 1.20)

        # --- Singles ---
        one = 1.0
        # No xBA term. There was one -- ``(xba - BL_XBA) * 0.60`` -- and it was
        # wrong twice over. It was **mis-centred**: ``BL_XBA`` is .250, a league
        # batting average per plate appearance, while ``self.xba`` is expected BA
        # over *batted balls only* and averages .332, so the median hitter was
        # collecting a free +4.9% on his singles rate and the whole distribution
        # sat at 1.034 instead of 1.0. And it was **not predictive**: fitted out
        # of time against singles per PA it comes back at -0.10, p=.78. The
        # apparent in-sample strength was circularity -- a ball that falls in for
        # a single is also a ball with a high xBA.
        #
        # Removing it does two things at once: the distribution re-centres on
        # 1.0, and the terms that *are* predictive stop being clipped away by a
        # constant offset that was pushing everyone into the ceiling.
        one *= 1.0 + _clip((BL_WHIFF - self.whiff) * 0.30, -0.06, 0.06)  # sensitive
        one *= 1.0 + _clip((self.zone_contact - BL_ZONE_CONTACT) * 0.30, -0.05, 0.05)  # sensitive
        one *= 1.0 + _clip((self.sprint_speed - BL_SPRINT) * 0.010, -0.04, 0.05)  # PPV speed
        one *= 1.0 + _clip(
            (BL_BARREL - self.barrel_rate) * singles_barrel_slope, -0.06, 0.06
        )  # NPV power
        # Batted-ball mix. Clamps sit at the 10th/90th percentile of each rate
        # so a real tail is priced but a thin-window outlier cannot run away
        # with the number.
        one *= 1.0 + _clip(
            (self.gb_rate - BL_GB_RATE) * singles_gb_slope, -0.05, 0.05
        )  # PPV batted-ball mix
        if self.ld_pct == self.ld_pct:  # not NaN
            one *= 1.0 + _clip(
                (self.ld_pct - BL_LD_RATE) * singles_ld_slope, -0.05, 0.05
            )  # PPV batted-ball mix
        # Shape within those classes: which grounders and which line drives.
        # Each drops out on its own when that batted-ball class is too thin.
        if singles_shape:
            if self.gb_pull_pct == self.gb_pull_pct:  # not NaN
                one *= 1.0 + _clip(
                    (self.gb_pull_pct - BL_GB_PULL) * SINGLES_GB_PULL_SLOPE,
                    -0.04,
                    0.04,
                )  # NPV rollover
            if self.ld_soft_pct == self.ld_soft_pct:
                one *= 1.0 + _clip(
                    (self.ld_soft_pct - BL_LD_SOFT) * SINGLES_LD_SOFT_SLOPE,
                    -0.04,
                    0.04,
                )  # PPV flare
            if self.ld_oppomid_pct == self.ld_oppomid_pct:
                one *= 1.0 + _clip(
                    (self.ld_oppomid_pct - BL_LD_OPPOMID) * SINGLES_LD_OPPOMID_SLOPE,
                    -0.04,
                    0.04,
                )  # PPV cut-off
        # The floor is deliberately further from 1.0 than the ceiling, and was
        # widened from 0.85. At 0.85 a genuinely poor contact hitter could not
        # be marked down more than 15% while the home-run line allows 50% -- and
        # the graded ledger shows the cost of that asymmetry lands exactly here:
        # the realised hit gap between the worst and best quartile of bats was
        # 9.3 points and the model priced 3.7. Weak bats were the false
        # positives, so the room to price them down is the half that mattered.
        one = _clip(one, 0.72, 1.18)

        # --- BABIP / dxwOBA luck regression (nudges contact outcomes) ---
        # Positive dxwOBA (unlucky) -> nudge up; high BABIP + neg dxwOBA -> down.
        luck = _clip(self.dxwoba * 1.5, -0.06, 0.06)
        if self.babip > 0.330 and self.dxwoba < 0:
            luck -= 0.03
        elif self.babip < 0.260 and self.dxwoba > 0:
            luck += 0.03
        contact_luck = 1.0 + luck

        return {
            "1B": one * contact_luck,
            "2B": xbh * contact_luck,
            "3B": xbh * contact_luck,
            "HR": hr,
        }


def _rate(series: pd.Series, cond) -> float:
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float(cond(s).mean())


def build_batter_regression(
    bdf: pd.DataFrame,
    sprint_speed: float = BL_SPRINT,
    league_xtb: LeagueXTB | None = None,
) -> BatterRegression:
    """Compute regression metrics from a batter's pitch-level Statcast slice."""
    batted = batted_balls(bdf)
    swings = bdf[
        bdf["description"].isin(
            ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"]
        )
    ]
    whiffs = bdf["description"].isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])

    n_bbe = int(len(batted))
    lsa = batted["launch_speed_angle"].dropna() if "launch_speed_angle" in batted else pd.Series([])
    barrel = float((lsa == 6).mean()) if len(lsa) else 0.0
    hard_hit = _safe_float((batted["launch_speed"] >= 95).mean(), 0.0) if n_bbe else 0.0
    la = batted["launch_angle"].dropna() if "launch_angle" in batted else pd.Series([])
    sweet = float(la.between(8, 32).mean()) if len(la) else 0.0
    gb_pct = float((la < 10).mean()) if len(la) else float("nan")
    pull_air = _pull_air_rate(batted)
    bat_speed = (
        float(bdf["bat_speed"].dropna().mean()) if bdf["bat_speed"].notna().any() else BL_BAT_SPEED
    )
    max_ev = _safe_float(batted["launch_speed"].max(), BL_MAX_EV) if n_bbe else BL_MAX_EV
    n_sw = int(len(swings))
    whiff = float(whiffs.sum() / n_sw) if n_sw else BL_WHIFF
    zc_swings = swings["zone"].between(1, 9) if "zone" in swings else pd.Series(dtype=bool)
    zc_contact = swings["description"].eq("hit_into_play") | swings["description"].isin(
        ["foul", "foul_tip"]
    )
    n_zsw = int(zc_swings.sum()) if len(zc_swings) else 0
    zone_contact = (
        float((zc_swings & zc_contact).sum() / n_zsw) if n_zsw else BL_ZONE_CONTACT
    )
    xba = (
        _safe_float(batted["estimated_ba_using_speedangle"].dropna().mean(), BL_XBA)
        if n_bbe and "estimated_ba_using_speedangle" in batted
        else BL_XBA
    )
    xwoba = (
        _safe_float(batted["estimated_woba_using_speedangle"].dropna().mean(), float("nan"))
        if n_bbe and "estimated_woba_using_speedangle" in batted
        else float("nan")
    )
    # actual wOBA over the same batted balls (contact-only comparison for dxwOBA)
    woba = (
        _safe_float(batted["woba_value"].dropna().mean(), float("nan"))
        if n_bbe and "woba_value" in batted
        else float("nan")
    )
    contact_slg = _estimate_xslg(batted)
    babip = _babip(bdf)
    bb_type = batted["bb_type"].dropna() if "bb_type" in batted else pd.Series(dtype=object)
    gb_rate = float(bb_type.eq("ground_ball").mean()) if len(bb_type) else BL_GB_RATE
    # Pop-ups as a share of all fly balls: the over-corrected swing whose extra
    # launch angle dies on the infield rather than clearing a fence.
    n_fb = int(bb_type.isin(["fly_ball", "popup"]).sum())
    iffb = float(bb_type.eq("popup").sum() / n_fb) if n_fb else float("nan")
    fb_ld_ev, fb_ld_max_ev, fb_ld_hard_hit = _air_contact(batted)
    gb_pull_pct, ld_soft_pct, ld_oppomid_pct = _singles_shape(batted)

    ld_pct = float(bb_type.eq("line_drive").mean()) if len(bb_type) else float("nan")

    ev = bdf["events"].dropna()
    n_pa = int(len(ev))
    n_ab = int((~ev.isin(NON_AB_EVENTS)).sum())
    # Expected slugging on slugging's own scale: expected bases over at-bats.
    # Falls back to the contact-quality rescale when no league lookup is given.
    xslg = contact_slg
    if league_xtb is not None and n_bbe and n_ab > 0:
        from_grid = league_xtb.xslg(batted, n_ab)
        if not np.isnan(from_grid):
            xslg = from_grid
    total_bases = float(ev.map(TB_VALUE).fillna(0.0).sum())
    slg = total_bases / n_ab if n_ab else float("nan")
    xbh_per_pa = float(ev.isin(["double", "triple"]).sum() / n_pa) if n_pa else float("nan")
    tb_home_bias = _tb_home_bias(bdf)
    k_pct = float(ev.isin(["strikeout", "strikeout_double_play"]).sum() / n_pa) if n_pa else float("nan")
    bb_pct = float(ev.eq("walk").sum() / n_pa) if n_pa else float("nan")

    if np.isnan(xwoba):
        xwoba = BL_XBA
    if np.isnan(woba):
        woba = xwoba

    return BatterRegression(
        bbe=n_bbe,
        barrel_rate=barrel,
        hard_hit=hard_hit,
        sweet_spot=sweet,
        bat_speed=bat_speed,
        max_ev=max_ev,
        whiff=whiff,
        zone_contact=zone_contact,
        xba=xba,
        xslg=xslg,
        babip=babip,
        woba=woba,
        xwoba=xwoba,
        sprint_speed=sprint_speed,
        k_pct=k_pct,
        bb_pct=bb_pct,
        gb_rate=gb_rate,
        gb_pct=gb_pct,
        pull_air_pct=pull_air,
        pa=n_pa,
        fb_ld_ev=fb_ld_ev,
        fb_ld_max_ev=fb_ld_max_ev,
        fb_ld_hard_hit=fb_ld_hard_hit,
        iffb_pct=iffb,
        gb_pull_pct=gb_pull_pct,
        ld_soft_pct=ld_soft_pct,
        ld_oppomid_pct=ld_oppomid_pct,
        slg=slg,
        xbh_per_pa=xbh_per_pa,
        ld_pct=ld_pct,
        tb_home_bias=tb_home_bias,
        contact_slg=contact_slg,
    )


# Plate appearances needed on *each* side before a venue gap is worth printing.
# A total-base rate over 30 PA is still noisy, but below that the split says more
# about which pitchers happened to be scheduled than about the ballpark.
MIN_VENUE_PA = 30


def _tb_home_bias(bdf: pd.DataFrame) -> float:
    """Home total bases per PA over road, as a share of the road rate."""
    if "inning_topbot" not in bdf:
        return float("nan")
    pa_rows = bdf[bdf["events"].notna()]
    home = pa_rows[pa_rows["inning_topbot"].eq("Bot")]["events"]
    away = pa_rows[pa_rows["inning_topbot"].eq("Top")]["events"]
    if min(len(home), len(away)) < MIN_VENUE_PA:
        return float("nan")
    road = float(away.map(TB_VALUE).fillna(0.0).mean())
    if road <= 0:
        return float("nan")
    return float(home.map(TB_VALUE).fillna(0.0).mean()) / road - 1.0


def _air_contact(batted: pd.DataFrame) -> tuple[float, float, float]:
    """Return (mean EV, max EV, hard-hit rate) over balls hit in the air.

    Air contact is launch angle >= 10 degrees -- fly balls and line drives. An
    exit velocity measured over *all* batted balls can be set by a scorched
    ground ball, which cannot become a home run, so the home-run terms read these
    instead. Returns NaNs when there is no launch-angle data to split on.
    """
    nan = float("nan")
    if "launch_angle" not in batted or "launch_speed" not in batted:
        return nan, nan, nan
    air = batted[batted["launch_angle"] >= 10.0]
    speed = air["launch_speed"].dropna()
    if speed.empty:
        return nan, nan, nan
    return (
        float(speed.mean()),
        float(speed.max()),
        float((speed >= 95.0).mean()),
    )


def barrel_rate(bdf: pd.DataFrame) -> tuple[float | None, int]:
    """Barrel rate over a pitch-level Statcast slice, plus its batted-ball count.

    Barrels are ``launch_speed_angle == 6`` (Statcast's optimal EV+LA class).
    Returns ``(None, n)`` when there are no batted balls / no classification, so
    callers can decide whether a window is thick enough to trust rather than
    reading 0.0 as "no power".
    """
    batted = batted_balls(bdf)
    n = int(len(batted))
    if n == 0:
        return None, 0
    lsa = (
        batted["launch_speed_angle"].dropna()
        if "launch_speed_angle" in batted
        else pd.Series([], dtype=float)
    )
    if len(lsa) == 0:
        return None, n
    return float((lsa == 6).mean()), n


def _pull_air_rate(batted: pd.DataFrame) -> float:
    """Share of batted balls that are pulled *and* in the air (LA >= 10).

    Spray angle from hit coordinates (home plate ~(125.42, 198.27)); negative =
    third-base/left-field side. A pull for a RHB is to left field (negative
    angle), for a LHB to right field (positive). Returns NaN when the hit-
    location or stance columns are unavailable.
    """
    need = {"hc_x", "hc_y", "launch_angle", "stand"}
    if not need.issubset(batted.columns) or batted.empty:
        return float("nan")
    df = batted.dropna(subset=["hc_x", "hc_y", "launch_angle", "stand"])
    if df.empty:
        return float("nan")
    spray = np.degrees(
        np.arctan2(df["hc_x"].astype(float) - 125.42, 198.27 - df["hc_y"].astype(float))
    )
    is_rhb = df["stand"].astype(str) == "R"
    pulled = (is_rhb & (spray < -10)) | (~is_rhb & (spray > 10))
    in_air = df["launch_angle"].astype(float) >= 10
    return float((pulled & in_air).mean())


def _singles_shape(batted: pd.DataFrame) -> tuple[float, float, float]:
    """Spray and contact quality *within* a hitter's grounders and line drives.

    Returns ``(gb_pull_pct, ld_soft_pct, ld_oppomid_pct)``:

    * pulled grounders as a share of grounders -- the "rollover" swing, which
      hits into the teeth of a defence aligned for exactly that;
    * line drives under 93 mph as a share of line drives -- the flare that is
      too high to field and too soft to reach the gap;
    * line drives to centre or the opposite field as a share of line drives --
      the ones an outfielder can cut off, holding the hitter to first.

    Each is NaN when the underlying batted-ball class is too thin to read.
    """
    need = {"hc_x", "hc_y", "launch_angle", "launch_speed", "stand", "bb_type"}
    if not need.issubset(batted.columns) or batted.empty:
        return (float("nan"), float("nan"), float("nan"))
    df = batted.dropna(subset=list(need))
    if df.empty:
        return (float("nan"), float("nan"), float("nan"))
    spray = np.degrees(
        np.arctan2(df["hc_x"].astype(float) - 125.42, 198.27 - df["hc_y"].astype(float))
    )
    is_rhb = df["stand"].astype(str) == "R"
    pulled = (is_rhb & (spray < -10)) | (~is_rhb & (spray > 10))
    gb = df["bb_type"] == "ground_ball"
    ld = df["bb_type"] == "line_drive"
    n_gb, n_ld = int(gb.sum()), int(ld.sum())
    gb_pull = float(pulled[gb].mean()) if n_gb >= MIN_GB_SHAPE else float("nan")
    if n_ld >= MIN_LD_SHAPE:
        ld_soft = float((df.loc[ld, "launch_speed"].astype(float) < 93.0).mean())
        ld_oppomid = float((~pulled[ld]).mean())
    else:
        ld_soft = ld_oppomid = float("nan")
    return (gb_pull, ld_soft, ld_oppomid)


def _estimate_xslg(batted: pd.DataFrame) -> float:
    """Expected slugging *on contact*, rescaled from expected wOBA on contact.

    An average over batted balls with no strikeout in the denominator, so it runs
    ~86 points above the slugging hitters actually post (league .486 against
    .400). That makes it the right side of the over-performance gap and the wrong
    thing to compare against a league slugging baseline -- for the level, see
    ``LeagueXTB``, which this only stands in for when no lookup is available.
    """
    if len(batted) == 0 or "estimated_woba_using_speedangle" not in batted:
        return BL_XSLG
    ewoba = batted["estimated_woba_using_speedangle"].dropna()
    if ewoba.empty:
        return BL_XSLG
    # rough affine map from xwOBAcon to xSLG scale
    return float(_clip(ewoba.mean() * 1.35, 0.10, 1.20))


def _babip(bdf: pd.DataFrame) -> float:
    ev = bdf["events"].dropna()
    if ev.empty:
        return BL_BABIP
    hits = ev.isin(["single", "double", "triple"]).sum()
    hr = ev.eq("home_run").sum()
    k = ev.isin(["strikeout", "strikeout_double_play"]).sum()
    bb = ev.isin(["walk", "hit_by_pitch"]).sum()
    sf = ev.eq("sac_fly").sum()
    ab_in_play = len(ev) - bb - hr - k  # balls in play (excl HR)
    denom = ab_in_play + sf
    if denom <= 0:
        return BL_BABIP
    return float((hits) / denom)


# Pitcher baselines.
BL_CSW = 0.280
BL_K_PCT = 0.220
BL_BB_PCT = 0.080
BL_K_MINUS_BB = 0.140
BL_BARREL_ALLOWED = 0.080
# Whiffs per two-strike PITCH, which is what ``build_pitcher_regression``
# measures. It was 0.280 -- a put-away rate per two-strike *swing* -- against a
# rate whose league value is .1448 for starters and .1483 for relief, so the
# term it feeds sat on its -0.06 clip for 98% of the 201 starters with 400+
# pitches and for every bullpen: an unconditional strikeout haircut applied to
# both sides of the ball, priced as if every arm in the league had the worst
# put-away stuff in it. See ``scripts/pen_stuff_study.py``.
BL_TWO_STRIKE_WHIFF = 0.145
# The same statistic off a bullpen, which is a different population: pooled over
# a dozen arms a pen posts .1196 K-BB against a starter's .140, so charging a pen
# the starter's baseline docked every bullpen in the league ~3% of its
# strikeouts before any of them was told apart from another.
BL_PEN_K_MINUS_BB = 0.120
BL_STUFF_PLUS = 100.0
BL_LOCATION_PLUS = 100.0
BL_SWSTR = 0.110  # swinging strikes / pitches
BL_WHIFF_PITCHER = 0.240  # swinging strikes / swings induced
BL_GB_ALLOWED = 0.420
# Half the fitted +0.77; see ``allowed_multipliers`` for why it is discounted.
OPP_GB_SINGLES_SLOPE = 0.39
# The other half of the same fact, and much better evidenced: a grounder that
# falls in is a single, so the extra-base channel loses what the singles channel
# gains. Fitted on 14,957 (game, batter) pairs against the game's starter over
# 201 starters, weighted by plate appearances and holding the hitter's own
# extra-base multiplier fixed: -1.47 of the league 2B+3B rate per unit of GB%
# allowed, t = -7.0, and monotone across all five quintiles of GB% allowed
# (2B+3B per PA .0528 .0495 .0467 .0420 .0411). It replicates out of time --
# -1.51 fitting on the first 60% of dates, -1.39 on the held out 40% (t = -4.1)
# -- which is why this ships at the fitted slope where the singles term ships at
# half of one.
OPP_GB_XBH_SLOPE = -1.45
OPP_GB_XBH_CLIP = (-0.14, 0.14)
# The same fact read off a bullpen, which is a different measurement. Pooling a
# corps of ~24 arms halves the spread in GB% allowed (sd 0.038 against a
# starter's 0.065) and it halves the slope with it: fitted on 13,681
# (game, batter) pairs against relief, holding the hitter's own rate and the
# pen's xwOBA allowed fixed, the extra-base coefficient is -0.76 where the
# starter's is -1.47, and the singles coefficient +0.51 where the starter's
# fitted +0.77. Neither reaches significance on one season of relief data
# (t -1.33 and +1.66), so the pen slopes are the fitted values rather than
# anything stronger, and the clip is tightened to match the narrower spread.
# Applying the starter's slope here overstated the term roughly two-fold.
PEN_GB_SINGLES_SLOPE = 0.25
PEN_GB_XBH_SLOPE = -0.75
PEN_GB_XBH_CLIP = (-0.07, 0.07)
BL_FB_ALLOWED = 0.360  # fly balls + pop-ups / batted balls

# You cannot hit a home run on the ground, and that is as true of the arm that
# induced the grounder as of the bat that hit it. Above this share of batted
# balls a starter is keeping the ball down as a matter of profile rather than
# luck; same ceiling and slope as the batter-side ground-ball brake.
GB_ALLOWED_CEILING = 0.50
GB_ALLOWED_SLOPE = 2.0
GB_ALLOWED_FLOOR = 0.78

# Fly-ball volume on its own is not a liability -- a starter can give up all the
# fly balls he likes if they are hit softly. It becomes a home-run problem only
# in combination with hard contact, so the term is the *product* of the two
# excesses rather than either alone, and stays dormant unless both are above
# baseline. The gain is large because it multiplies two small differences.
FB_ALLOWED_FLOOR = 0.420
FB_HARD_GAIN = 20.0
FB_HARD_CAP = 0.10

# Barrel rate allowed, on the home-run line. Measured against the next start's
# HR/PA over 2,426 starts / 56,072 PA, every feature read from pitches thrown
# strictly before the start, K% controlled, chronological 60/40 holdout:
#
#     term(s)                    coef      t   holdout dev
#     none (K only)                --     --      0.28406
#     barrel allowed            +2.12  +2.93      0.28391
#     hard-hit allowed          +0.86  +0.72      0.28404
#     GB% allowed               -1.33  -4.07      0.28362
#     FB% allowed               +1.61  +4.75      0.28374
#     GB + FB + barrel          +1.13  +1.46      0.28391
#
# Two things follow. The slope belongs where it is -- the fitted coefficient is
# +2.12 against the 2.0 that ships, and a weekly walk-forward over the whole
# multiplier is flat from 1.5 to 4.0 (0.28619/0.28617/0.28616) and worse at 0
# (0.28635), so the term is not the #195 case of something better deleted.
#
# But it is the *weakest* of the three batted-ball reads, not the strongest: it
# forward-predicts the next start's HR/PA at r=0.059 against fly-ball rate's
# 0.090 and ground-ball rate's -0.079, and marginal to the pair added after it
# (#87) it keeps only half its coefficient at t +1.46. Widening the clip was
# measured too and changes nothing (0.28617 at every bound tried), so the 4.8%
# of starts that reach it are reaching a bound the data does not mind.
# See ``scripts/starter_hr_terms.py``.
BARREL_ALLOWED_HR_SLOPE = 2.0

# Induced vertical break of the four-seamer, in inches: the usable proxy for a
# flat vertical approach angle. A high-ride fastball at the top of the zone is
# the pitch a high-launch hitter turns into a souvenir; a heavy sinking one is
# not. Weakest evidence of the matchup terms, so the least authority.
BL_IVB = 15.0
IVB_SLOPE = 0.008
IVB_CLIP = (-0.04, 0.06)
FOUR_SEAM_TYPES = {"FF", "FA"}

# Four-seam velocity, the one shape metric that pays on the strikeout side.
#
# What one start measures, correlated across a pitcher's consecutive starts:
# release height .97, extension .95, velocity .93, spin .91, IVB .84 -- then a
# cliff to whiff/swing .20, K/PA .20, CSW% .15, xwOBA allowed .10. One outing is
# ~90 radar-measured fastballs and ~22 results, so a velocity read off a single
# start is legitimate where every result-based read is not.
#
# Scored the way the engine uses it, 2,082 starts / 48,120 PA, binomial deviance
# per PA on strikeouts, six-week K%/CSW%/xwOBAcon controlled, chronological
# 60/40 holdout:
#
#                             coef      t   train    holdout
#     priced levels only        --     --  1.05786    1.05839
#     + velocity level      0.0478   7.88  1.05642    1.05732
#     + last-start dev      0.0972   5.32  1.05736    1.05767
#     + level and dev       0.0995   5.44  1.05588    1.05661
#
# Slopes are those logit coefficients on the strikeout scale (x0.78 at a .22
# league rate). A one-sided fit -- a dip counted differently from a spike --
# was tried because the velocity itself carries asymmetrically (30% of a dip
# survives to the next start, 55% of a spike) and it did not beat the linear
# term (.05684 / .05677 against .05661), so the simple version ships.
#
# It buys nothing on contact. On hits per NON-strikeout PA the level is t=-2.15
# for .0002 of deviance and the last-start deviation makes the holdout worse,
# which is the same verdict every starter contact instrument has drawn here.
BL_VFA = 94.7
VFA_K_LEVEL_SLOPE = 0.037  # per mph above the league four-seamer
VFA_K_DEV_SLOPE = 0.078  # per mph of last start away from his own level
VFA_K_LEVEL_CLIP = (-0.12, 0.12)
VFA_K_DEV_CLIP = (-0.08, 0.08)
MIN_VFA_PITCHES = 60  # four-seamers in the window before the level is read
MIN_VFA_START = 15  # four-seamers in the last start before its deviation is read

# Batted balls against one side of the plate before a starter's platoon contact
# split is trusted. Higher than the K split's PA floor because contact quality
# is the noisier measurement.
MIN_SPLIT_BBE = 40

# Stuff-based expected-K% fit: xK% is a linear function of the two fastest-
# stabilizing whiff signals (CSW% and SwStr%), anchored so a league-average arm
# (CSW .280, SwStr .110) maps to ~.220 K%. Used as the small-sample K prior so a
# pitcher regresses toward his stuff, not the flat league mean.
#
# The slopes were 2.6 and 1.4, never fitted -- only the anchor was ever chosen.
# Measured (``scripts.xk_refit_study``) on 2,936 starter-starts, each predicted
# from pitches thrown strictly before it and scored on the start that followed,
# they are 0.34 and 0.71: the hand-set line was about three times too steep, and
# a prior that steep is not a prior. Regressing the realised next-start K rate on
# it gave a slope of 0.286 where a calibrated prior gives 1.0 -- the arms it put
# at .100 struck out .181, the ones it put at .373 struck out .258, 27 points of
# prediction across 8 points of reality. Out of sample it lost to the league
# mean (wRMSE 0.1217 vs 0.1083) and to the pitcher's own raw 42-day rate
# (0.1027), which is the whole argument: a prior meant to rescue a thin sample
# was more extreme than the sample it was pulling on. It also pinned 27 of 259
# starters to a clip; the fitted line pins none.
XK_CSW_COEF = 0.34
XK_SWSTR_COEF = 0.71
XK_INTERCEPT = BL_K_PCT - XK_CSW_COEF * BL_CSW - XK_SWSTR_COEF * BL_SWSTR
MIN_SPLIT_PA = 25  # min PA vs a handedness before trusting a pitcher's platoon K%

# Walk (plate-discipline) baselines: Zone%, chase (O-Swing%), first-pitch strike%.
BL_ZONE = 0.480
BL_CHASE = 0.310
BL_FSTRIKE = 0.605

# Stuff-based expected-BB% fit: walks rise when a pitcher throws fewer strikes,
# induces fewer chases, and falls behind more often — these command signals
# stabilize far faster than observed BB%. Each term is (baseline - value), so a
# league-average arm maps to ~.085 and a high-Zone/high-chase arm (NPV screen)
# maps below it. Used as the small-sample BB prior.
#
# Fitted alongside xK% and hand-set in the same way beforehand: 0.50/0.40/0.30
# measure 0.21/0.09/0.11 against the next start's walk rate, a calibration slope
# of 0.369, and out of sample the hand-set line lost to the league mean as well
# (wRMSE 0.0647 vs 0.0621). Milder than the strikeout prior -- walks are noisier,
# and even the refit line only just beats doing nothing -- but the same sign and
# the same cause, so it is corrected rather than left to be found later.
XBB_ZONE_COEF = 0.21
XBB_CHASE_COEF = 0.09
XBB_FSTRIKE_COEF = 0.11

# Empirical-Bayes prior strengths for the contact-quality signals a starter
# allows, in batted balls. Measured by splitting the season into adjacent,
# non-overlapping six-week blocks (112 pitcher-pairs, ~106 batted balls a block)
# and correlating one block against the next: xwOBA r=0.31, hard-hit r=0.24,
# BABIP r=0.10, barrel r=0.09. Solving n/(n+k) = r at n=106 gives k, so a
# starter with a league-average sample keeps exactly his measured reliability
# and thin samples keep less. Contrast the command signals on the same blocks,
# which are left alone: K% r=0.52, whiff r=0.52, CSW r=0.50, velocity r=0.95.
# Scaled by the ball-in-play share of the old pool (78,107 / 147,827 = 0.53):
# the block sizes those k values were solved at counted foul balls as batted
# balls, so every prior was ~1.9x too strong. Re-measuring the block-to-block
# correlations on balls in play alone leaves the reliabilities essentially
# unchanged (hard-hit .274 -> .255, barrel .214 -> .211), so only n moves.
STARTER_PRIOR_BBE = {
    "xwoba": 123.0,
    "babip": 527.0,
    "hard_hit": 181.0,
    "barrel": 595.0,
}


def shrink_starter_rate(raw: float, baseline: float, bbe: int, prior_bbe: float,
                        strength: float = 1.0) -> float:
    """Pull an observed contact-quality rate toward its league baseline.

    ``strength`` scales the whole correction: 1.0 applies the measured
    empirical-Bayes weight, 0.0 leaves the raw rate untouched.
    """
    s = min(max(strength, 0.0), 1.0)
    if s == 0.0 or bbe <= 0:
        return raw
    keep = bbe / (bbe + prior_bbe)
    keep = keep + (1.0 - s) * (1.0 - keep)
    return baseline + keep * (raw - baseline)


CALLED_OR_WHIFF = {"called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING_DESC = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
K_EVENTS_P = ["strikeout", "strikeout_double_play"]


@dataclass
class PitcherRegression:
    bbe: int
    pitches: int
    babip_allowed: float
    woba_allowed: float
    xwoba_allowed: float
    hard_hit_allowed: float
    barrel_allowed: float
    csw: float
    k_pct: float
    bb_pct: float
    two_strike_whiff: float
    swstr: float = BL_SWSTR
    whiff: float = BL_WHIFF_PITCHER
    zone_pct: float = BL_ZONE
    chase: float = BL_CHASE
    fstrike: float = BL_FSTRIKE
    k_pct_vs_l: float = float("nan")
    k_pct_vs_r: float = float("nan")
    gb_allowed: float = BL_GB_ALLOWED
    fb_allowed: float = BL_FB_ALLOWED
    barrel_allowed_vs_l: float = float("nan")
    barrel_allowed_vs_r: float = float("nan")
    hard_hit_allowed_vs_l: float = float("nan")
    hard_hit_allowed_vs_r: float = float("nan")
    ivb: float = float("nan")
    extension: float = float("nan")
    release_var: float = float("nan")
    spin: float = float("nan")
    # Four-seam velocity over the window, and how far his most recent start sat
    # from it. ``vfa_k`` is the share of the fitted K term to charge (0 = off).
    vfa: float = float("nan")
    vfa_dev: float = float("nan")
    vfa_k: float = 0.0
    # Optional FanGraphs pitch-modeling metrics (None if no subscription data).
    stuff_plus: float | None = None
    location_plus: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    # Pre-shrinkage contact-quality rates, kept for reporting.
    raw_contact: dict[str, float] = field(default_factory=dict)
    # Whether these rates pool a whole bullpen rather than describing one arm.
    bullpen: bool = False

    @property
    def dxwoba(self) -> float:
        return self.xwoba_allowed - self.woba_allowed

    @property
    def k_minus_bb(self) -> float:
        return self.k_pct - self.bb_pct

    def expected_k_pct(self) -> float:
        """Stuff-based expected K% (xK%) from CSW% and SwStr%.

        These whiff signals stabilize in far fewer pitches than observed K%, so
        xK% is the right small-sample prior: a hard-to-hit arm with a thin PA
        sample should regress toward his stuff, not the flat league mean.

        The slopes are fitted against the next start's strikeout rate, so the
        line is a forecast and not a restatement of the arm's stuff on the K
        scale. It moves less than the raw rate it regularises, which is what a
        prior is for.
        """
        xk = XK_INTERCEPT + XK_CSW_COEF * self.csw + XK_SWSTR_COEF * self.swstr
        return _clip(xk, 0.08, 0.42)

    def expected_bb_pct(self) -> float:
        """Command-based expected BB% from Zone%, chase (O-Swing%), and F-strike%.

        These discipline signals stabilize far faster than observed BB%, so they
        are the right small-sample prior: fewer strikes / fewer chases / more
        first-pitch balls all push walks up; the reverse (the NPV screen) pushes
        them below the league mean. Slopes fitted against the next start's walk
        rate, as for :meth:`expected_k_pct`.
        """
        xbb = (
            BL_BB_PCT
            + XBB_ZONE_COEF * (BL_ZONE - self.zone_pct)
            + XBB_CHASE_COEF * (BL_CHASE - self.chase)
            + XBB_FSTRIKE_COEF * (BL_FSTRIKE - self.fstrike)
        )
        return _clip(xbb, 0.02, 0.20)

    def platoon_k_multiplier(self, bats: str | None) -> float:
        """K-rate multiplier for a batter of a given handedness (vs-L / vs-R split).

        Returns 1.0 when handedness is unknown or the split sample is too thin.
        """
        if bats not in ("L", "R") or self.k_pct <= 0:
            return 1.0
        split = self.k_pct_vs_l if bats == "L" else self.k_pct_vs_r
        if split != split:  # NaN -> insufficient split sample
            return 1.0
        return _clip(split / self.k_pct, 0.85, 1.18)

    def platoon_power_multiplier(self, bats: str | None) -> float:
        """HR multiplier for a batter of a given handedness (contact-quality split).

        A starter's overall home-run rate routinely hides a severe vulnerability
        to one side of the plate, and the K split cannot see it -- missing bats
        and suppressing contact quality are different skills. This reads the
        barrels and hard contact he actually allows to this handedness against
        his own overall rate, so a reverse-split arm stops looking ordinary.

        Returns 1.0 when handedness is unknown or the split sample is too thin.
        """
        if bats not in ("L", "R"):
            return 1.0
        barrel = self.barrel_allowed_vs_l if bats == "L" else self.barrel_allowed_vs_r
        hard = (
            self.hard_hit_allowed_vs_l if bats == "L" else self.hard_hit_allowed_vs_r
        )
        m = 1.0
        if barrel == barrel and self.barrel_allowed > 0:  # not NaN
            m *= _clip(barrel / self.barrel_allowed, 0.80, 1.25)
        if hard == hard and self.hard_hit_allowed > 0:
            m *= _clip(hard / self.hard_hit_allowed, 0.90, 1.12)
        return _clip(m, 0.85, 1.25)

    def batted_ball_hr_mult(self) -> float:
        """HR effect of *where* the pitcher lets the ball go, not how hard.

        Two opposite profiles, and the multiplier stack had neither: the sinker/
        ground-ball starter who suppresses the long ball against even elite
        power, and the fly-ball starter whose air contact is also hard.
        """
        m = 1.0
        if self.gb_allowed > GB_ALLOWED_CEILING:
            m *= _clip(
                1.0 - (self.gb_allowed - GB_ALLOWED_CEILING) * GB_ALLOWED_SLOPE,
                GB_ALLOWED_FLOOR,
                1.0,
            )
        if self.fb_allowed > FB_ALLOWED_FLOOR and self.hard_hit_allowed > BL_HARD_HIT:
            excess = (self.fb_allowed - FB_ALLOWED_FLOOR) * (
                self.hard_hit_allowed - BL_HARD_HIT
            )
            m *= 1.0 + _clip(excess * FB_HARD_GAIN, 0.0, FB_HARD_CAP)
        return m

    def k_multiplier(self) -> float:
        """Reported only: how this arm's stuff compares with the league's.

        Driven by CSW% and K-BB% (fast-stabilizing K predictors), the 2-strike
        put-away whiff rate, and Stuff+ when a FanGraphs feed is present. Each
        term is a deviation from a league baseline, so the product is a
        comparison between arms and reads well as one.

        It is **not** applied to a projected strikeout rate any more, because it
        cannot improve one. The rate it used to multiply is the arm's observed
        window K% blended toward xK% at 150 PA, and that rate is already
        calibrated: over 2,777 starts, each predicted from pitches thrown
        strictly before it, the blended rate's own quintiles land within a point
        of what the arm went on to do (.1856 -> .1811 at the bottom, .2674 ->
        .2658 at the top), while multiplying stretches them to .1473 and .3321.
        Out of sample the multiplier is worse than not having it -- weekly
        walk-forward wRMSE 0.10400 against 0.09763 -- and a dose search over the
        exponent picks 0.0. Refitting the terms does not rescue it (0.10122), and
        every term is individually harmful. #190 reached the same verdict on the
        bullpen half, where the pen's own pooled K% beat its stuff outright.

        CSW% is the reason: it is already inside xK%, so applying it again on top
        prices the same variable twice and re-inflates the spread the blend was
        built to shrink. See ``scripts/k_multiplier_study.py``.
        """
        if self.pitches < 100:
            return 1.0
        k_bb_baseline = BL_PEN_K_MINUS_BB if self.bullpen else BL_K_MINUS_BB
        m = 1.0
        m *= 1.0 + _clip((self.csw - BL_CSW) * 2.5, -0.15, 0.20)  # highest baseline PPV
        m *= 1.0 + _clip((self.k_minus_bb - k_bb_baseline) * 1.5, -0.12, 0.15)
        m *= 1.0 + _clip((self.two_strike_whiff - BL_TWO_STRIKE_WHIFF) * 0.8, -0.06, 0.08)
        if self.stuff_plus is not None:
            m *= 1.0 + _clip((self.stuff_plus - BL_STUFF_PLUS) * 0.004, -0.10, 0.15)
        return _clip(m, 0.75, 1.30)

    def velocity_k_multiplier(self) -> float:
        """How much of his strikeout rate his fastball is worth.

        Two terms: how hard he throws relative to the league, and how his most
        recent start sat against his own window. The second is the point -- one
        start measures velocity at r=.93 while measuring nothing else about him,
        so it is the only same-week form read the engine can honestly take.

        Applied to the blended strikeout rate in the pipeline, not folded into
        ``k_multiplier``: that one is reported only, and multiplying there would
        show the velocity read twice while pricing it once. As a rate forecast it
        earns that place -- weekly walk-forward over 3,086 starts, wRMSE 0.09749
        on the blended rate alone against 0.09634 with this term, and the dose
        search keeps ~0.8 of it where it kept none of stuff -- but replayed
        through the simulator on graded slates it prices slightly worse, so
        ``vfa_k`` ships at 0. Both studies are in ``scripts/``.

        Returns 1.0 unless ``vfa_k`` is set, and for a reliever always: this was
        fitted on starts.
        """
        if self.vfa_k <= 0.0 or self.bullpen:
            return 1.0
        m = 1.0
        if self.vfa == self.vfa:  # not NaN
            m *= 1.0 + self.vfa_k * _clip(
                (self.vfa - BL_VFA) * VFA_K_LEVEL_SLOPE, *VFA_K_LEVEL_CLIP
            )
        if self.vfa_dev == self.vfa_dev:
            m *= 1.0 + self.vfa_k * _clip(
                self.vfa_dev * VFA_K_DEV_SLOPE, *VFA_K_DEV_CLIP
            )
        return m

    def allowed_multipliers(self) -> dict[str, float]:
        """Multipliers on outcomes the pitcher ALLOWS (hits/xbh/hr)."""
        if self.bbe < MIN_BBE:
            return {}
        base = 1.0
        if self.bullpen:
            # A pen's rates pool ~1,240 batted balls, twelve times a starter's,
            # so the luck correction below is the wrong instrument on it: what
            # reads as a lucky starter is a good bullpen, and treating it as
            # luck prices the pen backwards. Measured on 16,547 game-batter rows
            # of relief plate appearances, held out on the last 40% of the
            # season:
            #
            #                              hits/PA deviance
            #     no pen contact term          .41972
            #     inverse BABIP (was shipped)  .41997   <- worse than nothing
            #     xwOBA allowed level          .41921
            #
            # The quintile table is blunter still: the pens allowing the most
            # hits (.2245/PA) were the ones handed a 0.989 suppression.
            base *= 1.0 + _clip(
                (self.xwoba_allowed - BL_PEN_XWOBA) * PEN_XWOBA_SLOPE,
                *PEN_XWOBA_CLIP,
            )
        # A single starter gets no contact-*level* term at all, and the reason is
        # sample size rather than baseball. The pen term above works because a pen
        # pools ~1,240 batted balls; a starter's 42-day window is a median of 95,
        # where the measurement error on a rate is the size of the whole spread
        # between pitchers. Measured on 2,311 starts / 53,353 PA against starters,
        # binomial deviance per PA, K% controlled, chronological 60/40 holdout:
        #
        #                                    coef       t   train    holdout
        #     no contact term                  --      --  1.06259   1.05355
        #     inverse BABIP (was shipped)   -0.089   -0.44  1.06255   1.05364
        #     dxwOBA allowed (was shipped)  -0.122   -0.61  1.06253   1.05369
        #     xwOBAcon level                 0.053    0.24  1.06259   1.05355
        #
        # Every instrument is indistinguishable from noise and every one made the
        # holdout worse, while the two that shipped swung the allowed-hit rate over
        # a 0.865..1.166 range -- more than 5% in 54% of starts. What a starter
        # repeats is the trajectory, not the outcome, so the ground-ball term below
        # is the only allowed-contact read he earns.
        # Ground balls allowed. A grounder that gets through is a single 91% of
        # the time, so the starter who keeps the ball down concedes singles in
        # place of extra bases -- and unlike the rest of his allowed-contact
        # profile, this one is his to control. Split-half reliability over 42
        # days, measured on this cache:
        #
        #     GB% allowed     .658      <- stable, and the term below
        #     hard% allowed   .488
        #     xBA allowed     .158
        #     BABIP allowed   .126
        #
        # which is McCracken's DIPS result reproduced directly: what a pitcher
        # repeats is the trajectory, not the outcome.
        #
        # Sized at *half* the fitted slope (+0.77 out of time, K% controlled,
        # p=.018) and this is the weakest-evidenced term in this file. Three
        # reasons to discount it rather than ship it at full strength: n=128
        # pitcher-windows; it was one of nine candidates tested, so p=.018 does
        # not survive a multiplicity correction; and sinker share (p=.045) is
        # very nearly the same variable, so the two cannot both be believed.
        # Rejected alongside it, all failing outright: the xBA-minus-xwOBA gap
        # (p=.71), xBA allowed (p=.49), extension (p=.55), sweet-spot allowed
        # (p=.17) and hard-hit allowed (p=.60) -- every one of them a stat from
        # the low-reliability family above.
        #
        # Applied to 1B only, never through ``base``: the whole point of the term
        # is that a grounder is a single *instead of* an extra-base hit, so
        # letting it lift 2B/3B/HR would assert the opposite of what it means.
        base = _clip(base, 0.88, 1.14)
        gb_singles_slope = (
            PEN_GB_SINGLES_SLOPE if self.bullpen else OPP_GB_SINGLES_SLOPE
        )
        one = base * (
            1.0
            + _clip(
                (self.gb_allowed - BL_GB_ALLOWED) * gb_singles_slope, -0.035, 0.035
            )
        )  # PPV opposing grounders

        # Hard-hit rate allowed is the most reliable contact signal a starter
        # carries (block-to-block r=0.24, vs barrel's 0.09) and it separates
        # extra-base contact rather than singles, so it lifts 2B/3B/HR only.
        hard = 1.0 + _clip((self.hard_hit_allowed - BL_HARD_HIT) * 1.0, -0.08, 0.10)

        # Ground balls on the extra-base line: the mirror of the singles term
        # above, and the half of it that was missing. GB% allowed is the one
        # batted-ball rate a starter genuinely repeats -- it forward-predicts his
        # next three weeks at r=0.42 and stabilizes by ~50 batted balls, against
        # barrel allowed's r=0.05 and a ~500-BBE stabilization point -- so it is
        # the only member of this family strong enough to move a rate rather than
        # merely inform a veto.
        gb_slope, gb_clip = (
            (PEN_GB_XBH_SLOPE, PEN_GB_XBH_CLIP)
            if self.bullpen
            else (OPP_GB_XBH_SLOPE, OPP_GB_XBH_CLIP)
        )
        gb_xbh = 1.0 + _clip((self.gb_allowed - BL_GB_ALLOWED) * gb_slope, *gb_clip)

        # Barrel rate allowed, read *unshrunk* because that is the scale
        # ``BARREL_ALLOWED_HR_SLOPE`` was fitted on. ``starter_contact_shrink``
        # ships at 0, so this is what production already prices; the point is
        # that enabling the knob must not silently gut the term. A 42-day window
        # is a median 95 batted balls against a 595-BBE prior, so shrinkage keeps
        # 14% of the excess -- at a raw-fitted slope the whole term would shrink
        # to about 1% of a home-run rate, an arithmetic accident rather than a
        # decision. The shrunk-scale equivalent is a slope near 9, which is not
        # what this constant is.
        barrel_allowed = self.raw_contact.get("barrel", self.barrel_allowed)
        hr = base * hard * (
            1.0
            + _clip(
                (barrel_allowed - BL_BARREL_ALLOWED) * BARREL_ALLOWED_HR_SLOPE,
                -0.10,
                0.18,
            )
        )
        # Where he lets the ball go, which is a separate skill from how hard it
        # is hit and the one the stack was missing entirely.
        hr *= self.batted_ball_hr_mult()
        # Ride on the four-seamer: the flat-approach fastball at the letters.
        if self.ivb == self.ivb:  # not NaN
            hr *= 1.0 + _clip((self.ivb - BL_IVB) * IVB_SLOPE, *IVB_CLIP)
        # Floor drops with the ground-ball brake so a sinkerballer can actually
        # suppress; the ceiling is unchanged.
        hr = _clip(hr, 0.78, 1.35)
        xbh = _clip(base * hard * gb_xbh, 0.85, 1.30)
        return {"1B": one, "2B": xbh, "3B": xbh, "HR": hr}


def _four_seam_velocity(pdf: pd.DataFrame) -> tuple[float, float]:
    """Window four-seam velocity, and where his most recent start sat against it.

    Both are NaN until there are enough four-seamers to read: a start's mean is
    only a measurement because it averages ~90 pitches, so a start he barely
    threw the pitch in says nothing.
    """
    if "pitch_type" not in pdf or "release_speed" not in pdf:
        return float("nan"), float("nan")
    fb = pdf[pdf["pitch_type"].isin(FOUR_SEAM_TYPES)].dropna(subset=["release_speed"])
    if len(fb) < MIN_VFA_PITCHES:
        return float("nan"), float("nan")
    level = float(fb["release_speed"].mean())
    if "game_date" not in fb:
        return level, float("nan")
    last = fb[fb["game_date"] == fb["game_date"].max()]["release_speed"]
    if len(last) < MIN_VFA_START:
        return level, float("nan")
    return level, float(last.mean()) - level


def build_pitcher_regression(
    pdf: pd.DataFrame,
    stuff_plus: float | None = None,
    location_plus: float | None = None,
    shrink: float = 0.0,
    bullpen: bool = False,
    vfa_k: float = 0.0,
) -> PitcherRegression:
    batted = batted_balls(pdf)
    n_bbe = int(len(batted))
    n_pitches = int(len(pdf))
    babip = _babip(pdf)
    hard_hit = float((batted["launch_speed"] >= 95).mean()) if n_bbe else BL_HARD_HIT
    barrel = (
        float((batted["launch_speed_angle"] == 6).mean())
        if n_bbe and "launch_speed_angle" in batted
        else BL_BARREL_ALLOWED
    )
    xwoba = (
        float(batted["estimated_woba_using_speedangle"].dropna().mean())
        if n_bbe and "estimated_woba_using_speedangle" in batted
        else BL_XBA
    )
    woba = (
        float(batted["woba_value"].dropna().mean())
        if n_bbe and "woba_value" in batted
        else xwoba
    )

    # CSW% = (called strikes + whiffs) / pitches
    if n_pitches and "description" in pdf:
        csw = float(pdf["description"].isin(CALLED_OR_WHIFF).mean())
    else:
        csw = BL_CSW

    # SwStr% (whiffs / pitches) and whiff-per-swing: fastest-stabilizing K signals.
    if n_pitches and "description" in pdf:
        n_whiff = int(pdf["description"].isin(WHIFF_DESC).sum())
        n_swings = int(pdf["description"].isin(SWING_DESC).sum())
        swstr = n_whiff / n_pitches
        whiff = n_whiff / n_swings if n_swings else BL_WHIFF_PITCHER
    else:
        swstr, whiff = BL_SWSTR, BL_WHIFF_PITCHER

    # Command signals for the expected-BB% prior: Zone%, chase (O-Swing%), F-strike%.
    if n_pitches and "zone" in pdf and pdf["zone"].notna().any():
        in_zone = pdf["zone"].between(1, 9)
        zone_pct = float(in_zone.mean())
        out_zone = pdf[~in_zone]
        n_out = len(out_zone)
        chase = (
            float(out_zone["description"].isin(SWING_DESC).sum() / n_out)
            if n_out
            else BL_CHASE
        )
    else:
        zone_pct, chase = BL_ZONE, BL_CHASE

    # First-pitch strike% = share of 0-0 pitches that are not balls (type != "B").
    if n_pitches and "balls" in pdf and "strikes" in pdf and "type" in pdf:
        first = pdf[(pdf["balls"] == 0) & (pdf["strikes"] == 0)]
        fstrike = float((first["type"] != "B").mean()) if len(first) else BL_FSTRIKE
    else:
        fstrike = BL_FSTRIKE

    # K% / BB% over plate appearances the pitcher ended.
    ev = pdf["events"].dropna()
    pa = len(ev)
    if pa:
        k_pct = float(ev.isin(K_EVENTS_P).sum() / pa)
        bb_pct = float(ev.isin(["walk", "hit_by_pitch"]).sum() / pa)
    else:
        k_pct, bb_pct = BL_K_PCT, BL_BB_PCT

    # Platoon K% split (vs LHB / vs RHB) from the batter's stance on each PA.
    k_pct_vs_l = k_pct_vs_r = float("nan")
    if pa and "stand" in pdf:
        pa_rows = pdf[pdf["events"].notna()]
        for hand, target in (("L", "l"), ("R", "r")):
            side = pa_rows[pa_rows["stand"] == hand]["events"]
            if len(side) >= MIN_SPLIT_PA:
                rate = float(side.isin(K_EVENTS_P).sum() / len(side))
                if target == "l":
                    k_pct_vs_l = rate
                else:
                    k_pct_vs_r = rate

    # 2-strike put-away whiff rate.
    if "strikes" in pdf and n_pitches:
        two_strike = pdf[pdf["strikes"] == 2]
        n2 = len(two_strike)
        two_strike_whiff = (
            float(two_strike["description"].isin(WHIFF_DESC).sum() / n2)
            if n2
            else BL_TWO_STRIKE_WHIFF
        )
    else:
        two_strike_whiff = BL_TWO_STRIKE_WHIFF

    # Induced vertical break of the FOUR-SEAMER (pfx_z in feet -> inches). The
    # mean over every pitch is not a ride measurement -- breaking balls carry
    # negative pfx_z, so it mostly reported arsenal composition.
    ivb = float("nan")
    if "pfx_z" in pdf and "pitch_type" in pdf:
        four_seam = pdf[pdf["pitch_type"].isin(FOUR_SEAM_TYPES)]["pfx_z"].dropna()
        if len(four_seam) >= 20:
            ivb = float(four_seam.mean() * 12.0)

    vfa, vfa_dev = _four_seam_velocity(pdf)

    ext = float(pdf["release_extension"].dropna().mean()) if pdf["release_extension"].notna().any() else float("nan")
    rel_var = (
        float(np.sqrt(pdf["release_pos_x"].dropna().var() + pdf["release_pos_z"].dropna().var()))
        if pdf["release_pos_x"].notna().any()
        else float("nan")
    )
    spin = float(pdf["release_spin_rate"].dropna().mean()) if pdf["release_spin_rate"].notna().any() else float("nan")

    # Contact quality is the least reliable thing a six-week starter sample
    # measures, and it drives the hit/HR multipliers, so it is the one group
    # that gets pulled toward league. xwOBA and wOBA share a weight because
    # they are the same batted balls and only their difference is consumed.
    raws = {
        "babip": babip,
        "woba": woba,
        "xwoba": xwoba,
        "hard_hit": hard_hit,
        "barrel": barrel,
    }
    if shrink > 0.0 and n_bbe:
        babip = shrink_starter_rate(babip, BL_BABIP, n_bbe, STARTER_PRIOR_BBE["babip"], shrink)
        xwoba = shrink_starter_rate(xwoba, BL_XBA, n_bbe, STARTER_PRIOR_BBE["xwoba"], shrink)
        woba = shrink_starter_rate(woba, BL_XBA, n_bbe, STARTER_PRIOR_BBE["xwoba"], shrink)
        hard_hit = shrink_starter_rate(
            hard_hit, BL_HARD_HIT, n_bbe, STARTER_PRIOR_BBE["hard_hit"], shrink
        )
        barrel = shrink_starter_rate(
            barrel, BL_BARREL_ALLOWED, n_bbe, STARTER_PRIOR_BBE["barrel"], shrink
        )

    # Where the batted balls he allows actually go. The hitter side has carried
    # a ground-ball rate since the batted-ball work; the pitcher side had none.
    if n_bbe and "bb_type" in batted:
        bbt = batted["bb_type"].dropna()
        n_bbt = len(bbt)
        gb_allowed = float(bbt.eq("ground_ball").sum() / n_bbt) if n_bbt else (
            BL_GB_ALLOWED
        )
        fb_allowed = (
            float(bbt.isin(["fly_ball", "popup"]).sum() / n_bbt)
            if n_bbt
            else BL_FB_ALLOWED
        )
    else:
        gb_allowed, fb_allowed = BL_GB_ALLOWED, BL_FB_ALLOWED

    # Contact quality allowed by batter handedness: a starter's home-run risk
    # routinely lives on one side of the plate only, and the K split cannot see
    # it. Left raw -- these are read as a ratio to his own overall rate, which
    # is already shrunk, so shrinking both ends would cancel the split out.
    barrel_vs = {"L": float("nan"), "R": float("nan")}
    hard_vs = {"L": float("nan"), "R": float("nan")}
    if n_bbe and "stand" in batted:
        for hand in ("L", "R"):
            side = batted[batted["stand"] == hand]
            if len(side) < MIN_SPLIT_BBE:
                continue
            hard_vs[hand] = float((side["launch_speed"] >= 95).mean())
            if "launch_speed_angle" in side:
                barrel_vs[hand] = float((side["launch_speed_angle"] == 6).mean())

    return PitcherRegression(
        bbe=n_bbe,
        pitches=n_pitches,
        gb_allowed=gb_allowed,
        fb_allowed=fb_allowed,
        barrel_allowed_vs_l=barrel_vs["L"],
        barrel_allowed_vs_r=barrel_vs["R"],
        hard_hit_allowed_vs_l=hard_vs["L"],
        hard_hit_allowed_vs_r=hard_vs["R"],
        babip_allowed=babip,
        woba_allowed=woba,
        xwoba_allowed=xwoba,
        hard_hit_allowed=hard_hit,
        barrel_allowed=barrel,
        raw_contact=raws,
        csw=csw,
        k_pct=k_pct,
        bb_pct=bb_pct,
        two_strike_whiff=two_strike_whiff,
        swstr=swstr,
        whiff=whiff,
        zone_pct=zone_pct,
        chase=chase,
        fstrike=fstrike,
        k_pct_vs_l=k_pct_vs_l,
        k_pct_vs_r=k_pct_vs_r,
        ivb=ivb,
        vfa=vfa,
        vfa_dev=vfa_dev,
        vfa_k=vfa_k,
        extension=ext,
        release_var=rel_var,
        spin=spin,
        stuff_plus=stuff_plus,
        location_plus=location_plus,
        bullpen=bullpen,
    )
