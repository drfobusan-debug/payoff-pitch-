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
    BL_BAT_SPEED,
    BL_HARD_HIT,
    BL_MAX_EV,
    BL_SWEET_SPOT,
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


# Total-bases selector weights. Backtesting the graded total-bases props showed
# max exit velocity is the one metric that separates over-winners from losers at
# every line (strongest on the power-driven 2.5/3.5 lines), while xSLG/xBA carry
# little signal -- yet max_ev was historically weighted ~5x smaller than xSLG.
# These up-weight raw power and are env-tunable.
TB_MAX_EV_W = _env_float("MLBE_TB_MAX_EV_W", 0.20)
TB_BARREL_W = _env_float("MLBE_TB_BARREL_W", 5.0)


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
    # Barrel-rate trend windows for the HR barrel gate (None when too thin).
    hr_barrel_3w: float | None = None
    hr_barrel_6w: float | None = None
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

        xslg_delta = breg.xslg - BL_XSLG
        score += xslg_delta * 10.0
        reasons.append(f"xslg={breg.xslg:.3f}({xslg_delta:+.3f})")

        sweet_delta = breg.sweet_spot - BL_SWEET_SPOT
        score += sweet_delta * 8.0
        reasons.append(f"sweet={breg.sweet_spot:.3f}({sweet_delta:+.3f})")

        bat_speed_delta = breg.bat_speed - BL_BAT_SPEED
        score += bat_speed_delta * 0.08
        reasons.append(f"bat_speed={breg.bat_speed:.1f}({bat_speed_delta:+.1f})")

        max_ev_delta = breg.max_ev - BL_MAX_EV
        score += max_ev_delta * 0.04
        reasons.append(f"max_ev={breg.max_ev:.1f}({max_ev_delta:+.1f})")

        barrel_delta = breg.barrel_rate - BL_BARREL
        score += barrel_delta * 3.0
        reasons.append(f"barrel={breg.barrel_rate:.3f}({barrel_delta:+.3f})")

        hard_delta = breg.hard_hit - BL_HARD_HIT
        score += hard_delta * 2.0
        reasons.append(f"hard={breg.hard_hit:.3f}({hard_delta:+.3f})")

        # Platoon / lineup context (small nudges).
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

        xslg_delta = breg.xslg - BL_XSLG
        score += xslg_delta * 10.0
        reasons.append(f"xslg={breg.xslg:.3f}({xslg_delta:+.3f})")

        xba_delta = breg.xba - BL_XBA
        score += xba_delta * 8.0
        reasons.append(f"xba={breg.xba:.3f}({xba_delta:+.3f})")

        sweet_delta = breg.sweet_spot - BL_SWEET_SPOT
        score += sweet_delta * 5.0
        reasons.append(f"sweet={breg.sweet_spot:.3f}({sweet_delta:+.3f})")

        bat_speed_delta = breg.bat_speed - BL_BAT_SPEED
        score += bat_speed_delta * 0.08
        reasons.append(f"bat_speed={breg.bat_speed:.1f}({bat_speed_delta:+.1f})")

        max_ev_delta = breg.max_ev - BL_MAX_EV
        score += max_ev_delta * TB_MAX_EV_W
        reasons.append(f"max_ev={breg.max_ev:.1f}({max_ev_delta:+.1f})")

        barrel_delta = breg.barrel_rate - BL_BARREL
        score += barrel_delta * TB_BARREL_W
        reasons.append(f"barrel={breg.barrel_rate:.3f}({barrel_delta:+.3f})")

        hard_delta = breg.hard_hit - BL_HARD_HIT
        score += hard_delta * 2.0
        reasons.append(f"hard={breg.hard_hit:.3f}({hard_delta:+.3f})")

        sprint_delta = breg.sprint_speed - 27.0
        score += sprint_delta * 0.05
        reasons.append(f"sprint={breg.sprint_speed:.1f}({sprint_delta:+.1f})")

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
