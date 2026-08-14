"""V1-style prop selectors for RBI, XBH, and TB.

These selectors wrap the existing regression/park/weather features and produce a
``Selection`` (factor, signal, score, profile) that the pipeline can attach to
recommendations.  XBH feeds the existing 2B/3B outcome multipliers; TB and RBI are
applied as post-simulation prop multipliers, reusing the existing ``batter_tb``
and simulated RBI markets rather than creating new ones.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from mlb_engine.data.parks import Park
from mlb_engine.features.regression import (
    BL_BARREL,
    BL_BARREL_PA,
    BL_HARD_HIT,
    BL_MAX_EV,
    BL_SPRINT,
    BL_XBA,
    BL_XSLG,
    MIN_BBE,
    BatterRegression,
)
from mlb_engine.models.rbi_rule import RBIFlag, rbi_multiplier

# Contact-quality floor markets: power props keyed on xSLG, contact props on K%.
POWER_FLOOR_MARKETS = ("HR", "2B", "3B", "TB")
CONTACT_FLOOR_MARKETS = ("H", "1B", "HRR")


def power_floor_reason(
    breg: BatterRegression | None,
    stat: str,
    *,
    xslg_floor: float,
    k_ceiling: float,
) -> str | None:
    """Reason a batter prop should be excluded by the contact-quality floor.

    Power markets (HR/2B/3B/TB) exclude bats below ``xslg_floor`` xSLG; contact
    markets (H/1B/HRR) exclude bats above ``k_ceiling`` K%.  Returns ``None`` --
    i.e. no exclusion -- when the sample is too thin (``bbe < MIN_BBE``) or the
    feature is missing (NaN), so we never gate on unknown data.
    """
    if breg is None or breg.bbe < MIN_BBE:
        return None
    if stat in POWER_FLOOR_MARKETS and breg.xslg < xslg_floor:
        return f"power floor: xSLG {breg.xslg:.3f} < {xslg_floor:.3f}"
    if (
        stat in CONTACT_FLOOR_MARKETS
        and breg.k_pct == breg.k_pct  # not NaN
        and breg.k_pct > k_ceiling
    ):
        return f"contact floor: K% {breg.k_pct:.3f} > {k_ceiling:.3f}"
    return None


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# Total-bases selector weights, measured rather than assumed.
#
# 2,609 batter-weeks: each hitter's trailing-42-day profile against his *next*
# seven days of real total bases, fitted unstandardized so a coefficient converts
# straight into this selector's units (score -> factor is ``1 + 0.03 * score``,
# so weight = beta / mean TB per PA / 0.03), then re-fitted on May-June and
# checked on July. Only two terms keep their sign and size out of sample:
#
#   metric      univariate r   score weight (May-Jun -> Jul)
#   max_ev          +0.145      +0.44 -> +0.63   p<.005 in both
#   barrel/PA       +0.125     +31.6  -> +35.8
#   xSLG            +0.114     -50.6  -> +14.1   sign flips
#   xBA             +0.088      +3.3  -> -16.2   sign flips
#   sweet_spot      +0.027      +9.3  -> -12.1   sign flips
#   LD%             +0.006      -7.9  -> +22.0   nothing, either half
#
# So xSLG, xBA and SweetSpot% are univariately positive but carry no information
# max EV and barrel/PA do not already carry: their weights come down to a level
# anchor rather than the largest terms on the line, and LD% is stamped for
# auditing but not scored. Barrel/PA replaces barrel-per-batted-ball, which
# credits a hitter who barrels rarely but puts few balls in play.
TB_MAX_EV_W = _env_float("MLBE_TB_MAX_EV_W", 0.50)
TB_BARREL_PA_W = _env_float("MLBE_TB_BARREL_PA_W", 30.0)
TB_XSLG_W = _env_float("MLBE_TB_XSLG_W", 3.0)
TB_XBA_W = _env_float("MLBE_TB_XBA_W", 3.0)
TB_HARD_HIT_W = _env_float("MLBE_TB_HARD_HIT_W", 1.0)
TB_SPRINT_W = _env_float("MLBE_TB_SPRINT_W", 0.15)

# False-positive brakes: a hitter whose bases outrun his contact. Both are steps
# rather than slopes because that is how they measure -- forward TB/PA relative
# to every other batter-week, on the same 2,609:
#
#   SLG - xSLG > +.050            n=134  -11.0%  p=.008   (-7.5% / -15.9% by half)
#   SLG - xSLG > +.100            n= 47   -4.3%  p=.53    no worse than +.050
#   BABIP > .360 & hard-hit < .40 n= 84  -14.1%  p=.003  (-15.7% / -10.3%)
#   BABIP > .330 & hard-hit < .40 n=216   -9.7%  p=.004
#
# The flagged group's trailing rate is .496 TB/PA against .361 for everyone
# else: they look like the best bats on the board and then produce below average.
# Every one of those rows also trips the existing dxwOBA luck term, so these are
# sized on the *incremental* effect measured within that set (-9.9% and -13.6%),
# not on the raw gap.
TB_SLG_GAP_FLAG = 0.050
TB_SLG_GAP_PENALTY = _env_float("MLBE_TB_SLG_GAP_PENALTY", 3.0)
TB_BABIP_HIGH = 0.360
TB_BABIP_ELEVATED = 0.330
TB_BABIP_PENALTY = _env_float("MLBE_TB_BABIP_PENALTY", 4.0)


@dataclass
class Selection:
    """A V1-style selector recommendation for a single batter/prop."""

    signal: str  # buy | sell | hold | exclude | none
    factor: float  # multiplicative factor applied to probabilities
    score: float  # raw selector score (for output/auditing)
    profile: str  # human-readable rationale
    # Pre-simulation outcome multipliers (e.g. 2B/3B for XBH)
    outcome_multipliers: dict[str, float] = field(default_factory=dict)
    # Post-simulation stat multipliers (e.g. RBI/TB derived arrays)
    post_multipliers: dict[str, float] = field(default_factory=dict)
    # Batter power inputs for the home-run power gate (None when unavailable).
    hr_max_ev: float | None = None
    hr_barrel: float | None = None
    hr_bbe: int | None = None
    # Barrels per plate appearance and mean exit velocity on air contact, for
    # the HR gate's contact-frequency and soft-air tests (None when unavailable).
    hr_barrel_pa: float | None = None
    hr_fb_ld_ev: float | None = None
    # Contact-quality inputs for the H+R+RBI adjuster (None when unavailable).
    bat_sweet_spot: float | None = None
    bat_xslg: float | None = None

    def __bool__(self) -> bool:
        return self.signal not in ("none", "hold") or self.factor != 1.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _finite(x: float) -> float | None:
    """``None`` for a NaN metric, so a gate reads it as "no data" not "zero"."""
    return x if x == x else None


def _tb_false_positives(breg: BatterRegression) -> list[tuple[float, str]]:
    """Score penalties for bases the hitter's contact does not support.

    Total bases is the numerator of slugging, so a hitter slugging well past the
    quality of his contact is collecting bases from bloops, misplays and wind --
    and gives them back. Same for a hitter carrying a high BABIP on below-average
    hard contact. A metric that is unknown never brakes anything.

    The comparison is against expected slugging *on contact*, not against the
    calibrated xSLG: see ``BatterRegression.slg_gap`` for why the calibrated gap
    stops predicting.
    """
    out: list[tuple[float, str]] = []
    gap = breg.slg_gap
    if gap == gap and gap > TB_SLG_GAP_FLAG:
        out.append((TB_SLG_GAP_PENALTY, f"slg-contact={gap:+.3f} over-performing"))
    if breg.hard_hit < BL_HARD_HIT and breg.babip == breg.babip:
        if breg.babip > TB_BABIP_HIGH:
            out.append(
                (TB_BABIP_PENALTY, f"babip={breg.babip:.3f} on hard-hit={breg.hard_hit:.3f}")
            )
        elif breg.babip > TB_BABIP_ELEVATED:
            out.append(
                (
                    TB_BABIP_PENALTY * 0.625,
                    f"babip={breg.babip:.3f} on hard-hit={breg.hard_hit:.3f}",
                )
            )
    return out


def _env_factor(
    park: Park | None,
    weather: dict[str, float] | None,
    power: bool = True,
    hits: bool = True,
) -> float:
    """Combine park/weather into a bounded environment multiplier."""
    weather = weather or {}
    mult = 1.0

    if park is not None:
        # Overall run-scoring environment; damped so park factor doesn't dominate.
        mult *= 1.0 + (park.park_factor - 100.0) / 250.0
        if power:
            # Carry factor is strongest for HR/XBH; damped for TB.
            mult *= park.carry_factor ** (0.35 if hits else 0.25)

    if weather:
        if hits:
            hit_mult = weather.get("1B", 1.0) or weather.get("2B", 1.0)
            mult *= hit_mult
        if power:
            mult *= weather.get("HR", 1.0) ** 0.7

    return _clamp(mult, 0.80, 1.25)


def _score_to_factor(score: float) -> float:
    """Convert a V1-style score to a bounded probability multiplier."""
    return _clamp(1.0 + 0.03 * score, 0.85, 1.30)


def _signal_from_factor(factor: float, has_data: bool) -> str:
    if not has_data:
        return "none"
    if factor < 0.87:
        return "exclude"
    if factor > 1.05:
        return "buy"
    if factor < 0.95:
        return "sell"
    return "hold"


class RBISelector:
    """V1-style RBI selector using the existing rbi_rule.py logic + park/weather."""

    def __init__(self, obp_threshold: float = 0.345) -> None:
        self.obp_threshold = obp_threshold

    def select(
        self,
        flag: RBIFlag | None,
        breg: BatterRegression | None = None,
        park: Park | None = None,
        weather: dict[str, float] | None = None,
        slot: int | None = None,
        bats: str | None = None,
        opp_hand: str | None = None,
    ) -> Selection:
        if flag is None or not flag.flagged:
            return Selection(
                signal="none",
                factor=1.0,
                score=0.0,
                profile="no RBI opportunity trigger",
                post_multipliers={"RBI": 1.0},
            )

        base = rbi_multiplier(flag)
        reasons: list[str] = []
        reasons.append(f"preceding_obp={flag.preceding_obp:.3f}")
        if flag.xslg and not math.isnan(flag.xslg):
            reasons.append(f"xslg={flag.xslg:.3f}")
        if flag.zone_contact and not math.isnan(flag.zone_contact):
            reasons.append(f"zone_contact={flag.zone_contact:.3f}")

        env = _env_factor(park, weather, power=False, hits=True)
        factor = _clamp(base * env, 0.80, 1.30)

        # Slight lineup-slot boost for 3-6 (cleanup) hitters.
        if slot is not None and 2 <= slot <= 5:
            factor *= 1.02
            factor = _clamp(factor, 0.80, 1.30)

        score = (factor - 1.0) / 0.03
        signal = _signal_from_factor(factor, True)
        profile = " | ".join([f"rbi_mult={base:.3f}"] + reasons)

        return Selection(
            signal=signal,
            factor=factor,
            score=round(score, 2),
            profile=profile,
            post_multipliers={"RBI": factor},
        )


class XBHSelector:
    """V1-style extra-base-hit selector feeding the 2B/3B multiplier block."""

    def select(
        self,
        breg: BatterRegression | None,
        park: Park | None = None,
        weather: dict[str, float] | None = None,
        slot: int | None = None,
        bats: str | None = None,
        opp_hand: str | None = None,
    ) -> Selection:
        if breg is None or breg.bbe < MIN_BBE:
            return Selection(
                signal="none",
                factor=1.0,
                score=0.0,
                profile="insufficient batted-ball data",
                outcome_multipliers={"2B": 1.0, "3B": 1.0},
            )

        score = 0.0
        reasons: list[str] = []

        # No contact-quality score. This selector used to rank a hitter on six
        # measures -- xSLG, sweet spot, bat speed, max EV, barrel rate, hard-hit
        # rate -- and move his doubles rate by up to 30% on the result. Fitted
        # out of time against forward 2B+3B per PA (48,120 plate appearances,
        # eight rolling blocks, hitters at 40+ batted balls), not one of them
        # separates the hitters who go on to double:
        #
        #     xslg  -0.017 (p=.44)   sweet   -0.024 (p=.26)   hard  +0.013 (p=.57)
        #     barrel -0.014 (p=.54)  bat_spd -0.014 (p=.51)   xwoba -0.001 (p=.97)
        #
        # Max EV is the near miss and it does not survive either: +0.041 at
        # p=.066, and once the batted-ball count it is a maximum over sits
        # beside it the sign flips in three of eight blocks. It survives on
        # *total bases* (see ``TBSelector``), where the home run it really
        # measures is in the target.
        #
        # #132 removed the same three terms from the rate multiplier; this is
        # the second, independent copy of them that sat on top of it.
        #
        # Park and weather stay: where the ball is hit is not who hits it, and
        # the doubles park factor was fitted on doubles (#113).

        # Lineup context: a middle-order spot is more plate appearances, which
        # is a real effect on a per-game prop rather than a talent claim.
        if slot is not None and 2 <= slot <= 5:
            score += 0.5
            reasons.append("cleanup_spot")

        score = _clamp(score, -8.0, 15.0)
        env = _env_factor(park, weather, power=True, hits=True)
        factor = _clamp(_score_to_factor(score) * env, 0.85, 1.30)

        signal = _signal_from_factor(factor, True)
        return Selection(
            signal=signal,
            factor=factor,
            score=round(score, 2),
            profile=" | ".join(reasons),
            outcome_multipliers={"2B": factor, "3B": factor},
        )


class TBSelector:
    """V1-style total-bases selector feeding the existing batter_tb market."""

    def select(
        self,
        breg: BatterRegression | None,
        park: Park | None = None,
        weather: dict[str, float] | None = None,
        slot: int | None = None,
        bats: str | None = None,
        opp_hand: str | None = None,
    ) -> Selection:
        if breg is None or breg.bbe < MIN_BBE:
            return Selection(
                signal="none",
                factor=1.0,
                score=0.0,
                profile="insufficient batted-ball data",
                post_multipliers={"TB": 1.0},
            )

        score = 0.0
        reasons: list[str] = []

        max_ev_delta = breg.max_ev - BL_MAX_EV
        score += max_ev_delta * TB_MAX_EV_W
        reasons.append(f"max_ev={breg.max_ev:.1f}({max_ev_delta:+.1f})")

        # Barrels per plate appearance, not per batted ball: contact frequency is
        # part of the metric, which is what turns power into bases.
        barrel_pa = breg.barrel_per_pa
        if barrel_pa == barrel_pa:  # not NaN
            barrel_delta = barrel_pa - BL_BARREL_PA
            score += barrel_delta * TB_BARREL_PA_W
            reasons.append(f"barrel/pa={barrel_pa:.3f}({barrel_delta:+.3f})")
        else:
            barrel_delta = breg.barrel_rate - BL_BARREL
            score += barrel_delta * TB_BARREL_PA_W * BL_BARREL_PA / BL_BARREL
            reasons.append(f"barrel={breg.barrel_rate:.3f}({barrel_delta:+.3f})")

        xslg_delta = breg.xslg - BL_XSLG
        score += xslg_delta * TB_XSLG_W
        reasons.append(f"xslg={breg.xslg:.3f}({xslg_delta:+.3f})")

        xba_delta = breg.xba - BL_XBA
        score += xba_delta * TB_XBA_W
        reasons.append(f"xba={breg.xba:.3f}({xba_delta:+.3f})")

        # Sweet-spot rate and bat speed are not scored. Total bases is the market
        # where contact quality has a claim -- the home run it measures is in the
        # target -- and on forward TB per PA over the same 48,120 plate
        # appearances the claim belongs to raw power, not to launch angle:
        #
        #     max_ev  +0.0194 (p<.0001)      sweet     +0.0022 (p=.55)
        #     xwoba   +0.0124 (p=.001)       bat_speed +0.0120 (p=.001)
        #     barrel  +0.0098 (p=.012)       xslg      +0.0074 (p=.049)
        #
        # but they are one signal wearing six coats: fitted together only max EV
        # survives (+0.0215, p=.0002) and xSLG turns negative. Max EV holds up
        # against the batted-ball count it is a maximum over (+0.0170, p<.0001)
        # and keeps its sign in six of eight blocks; sweet spot is the one
        # measure that is dead on this target as well as on doubles, and bat
        # speed only spoke through the power it shares with max EV.

        hard_delta = breg.hard_hit - BL_HARD_HIT
        score += hard_delta * TB_HARD_HIT_W
        reasons.append(f"hard={breg.hard_hit:.3f}({hard_delta:+.3f})")

        # Doubles and triples are 31% of the total bases in play, and speed is
        # what converts contact into them; the bulk of the effect sits on the
        # 2B/3B multiplier itself rather than here.
        sprint_delta = breg.sprint_speed - BL_SPRINT
        score += sprint_delta * TB_SPRINT_W
        reasons.append(f"sprint={breg.sprint_speed:.1f}({sprint_delta:+.1f})")

        for penalty, reason in _tb_false_positives(breg):
            score -= penalty
            reasons.append(reason)

        # Line-drive rate: stamped for auditing, deliberately not scored (r=+0.006
        # against forward TB/PA, and the sign flips between halves of the season).
        if breg.ld_pct == breg.ld_pct:  # not NaN
            reasons.append(f"ld={breg.ld_pct:.3f}")

        # Home/road total-base split, beside tonight's park factor: the reader's
        # check on whether a hitter's bases are a ballpark artefact. Stamped, not
        # scored -- tonight's venue is already priced from the matching half of
        # the hitter's own splits and then multiplied by the park factor below, so
        # scoring it here would charge for the same thing twice.
        if breg.tb_home_bias == breg.tb_home_bias:  # not NaN
            pf = f", tonight's park {park.park_factor:.0f}" if park is not None else ""
            reasons.append(f"home_tb_bias={breg.tb_home_bias:+.1%}{pf}")

        # Stamped for HR/PPV auditing (not scored here).
        if breg.gb_pct == breg.gb_pct:  # not NaN
            reasons.append(f"gb={breg.gb_pct:.3f}")
        if breg.pull_air_pct == breg.pull_air_pct:  # not NaN
            reasons.append(f"pull_air={breg.pull_air_pct:.3f}")

        if slot is not None and 2 <= slot <= 5:
            score += 0.5
            reasons.append("cleanup_spot")

        score = _clamp(score, -8.0, 15.0)

        # Total-bases weighted weather: singles count once, HR four times.
        weather = weather or {}
        hit_mult = weather.get("1B", 1.0) or weather.get("2B", 1.0)
        hr_mult = weather.get("HR", 1.0)
        weather_tb = (3.0 * hit_mult + 4.0 * hr_mult) / 7.0

        env = 1.0
        if park is not None:
            env *= 1.0 + (park.park_factor - 100.0) / 220.0
            env *= park.carry_factor ** 0.3
        env *= _clamp(weather_tb, 0.80, 1.25)

        factor = _clamp(_score_to_factor(score) * env, 0.85, 1.30)

        signal = _signal_from_factor(factor, True)
        return Selection(
            signal=signal,
            factor=factor,
            score=round(score, 2),
            profile=" | ".join(reasons),
            post_multipliers={"TB": factor},
            hr_max_ev=breg.air_max_ev,
            hr_barrel=breg.barrel_rate,
            hr_bbe=breg.bbe,
            hr_barrel_pa=_finite(breg.barrel_per_pa),
            hr_fb_ld_ev=_finite(breg.fb_ld_ev),
            bat_sweet_spot=breg.sweet_spot,
            bat_xslg=breg.xslg,
        )
