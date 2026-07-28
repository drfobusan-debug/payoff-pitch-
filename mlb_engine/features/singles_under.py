"""Singles "Under" NPV screen -- structural red flags that a batter fails a
1+ singles prop.

The Monte Carlo already reflects opponent defense (middle-infield OAA via the
defense layer) and the starter's contact profile (a groundball arm suppresses
the batter's hit rate), so those Tier-3 game-context factors are *already* in the
simulated singles probability.  What the sim's singles multiplier does **not**
capture is the batter's own structural anti-singles shape:

  Tier 1 -- volume killers: the ball never gets in play.
    * Three-True-Outcome profile: K% > 25% *and* BB% > 12%.
    * Passive in-zone approach: Z-Swing% < 60% (deep counts -> more Ks).

  Tier 2 -- contact "cleaners": contact bypasses the singles bucket.
    * Fly-ball tilt: average launch angle > 20 deg (singles peak ~5-15 deg).
    * Elite power contact: Barrel% > 15% *and* Hard-Hit% > 48% (hits leave as
      doubles/HRs, clearing the singles line).
    * Pull-heavy grounders: Pull% > 45% with a groundball tilt (avg LA < 5 deg)
      -- easily gloved by standard infield positioning.

``build_singles_under`` computes each factor from a batter's Statcast slice and
``singles_under_score`` turns the flags into a bounded NPV score with reasons.
The score is diagnostic -- callers decide how to act on it (e.g. excluding the
singles *over* from betting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Statcast spray-chart origin (home plate) in the hc_x/hc_y coordinate frame.
_HC_X0 = 125.42
_HC_Y0 = 198.27

_SWING_DESC = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
}

# --- factor thresholds (from the framework) ---
K_PCT_HI = 0.25
BB_PCT_HI = 0.12
Z_SWING_LO = 0.60
LA_FLYBALL = 20.0
LA_GROUNDBALL = 5.0
BARREL_HI = 0.15
HARD_HIT_HI = 0.48
PULL_HI = 0.45

MIN_PA = 40  # min plate appearances before the screen is trusted
MIN_BIP = 25  # min balls in play for batted-ball factors

# NPV score at/above which a batter is a strong singles-Under candidate.
SINGLES_UNDER_STRONG = 3.0


@dataclass(frozen=True)
class SinglesUnderProfile:
    """Batter-intrinsic anti-singles factors over a Statcast window."""

    pa: int
    bip: int
    k_pct: float
    bb_pct: float
    z_swing: float
    avg_la: float
    barrel: float
    hard_hit: float
    pull_rate: float

    @property
    def has_data(self) -> bool:
        return self.pa >= MIN_PA


def _spray_pull_rate(bip: pd.DataFrame, stand: str | None) -> float:
    """Share of balls in play hit to the pull field (NaN if uncomputable).

    Uses the Savant spray angle; the pull side flips with batter handedness
    (RHB pull to LF/3B, LHB pull to RF/1B).
    """
    if stand not in ("L", "R") or "hc_x" not in bip or "hc_y" not in bip:
        return float("nan")
    hc = bip[["hc_x", "hc_y"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(hc) < MIN_BIP:
        return float("nan")
    angle = np.degrees(
        np.arctan2(hc["hc_x"] - _HC_X0, _HC_Y0 - hc["hc_y"])
    )
    # Positive angle = toward LF (3B side) = pull for a RHB; mirror for a LHB.
    pull_side = angle if stand == "R" else -angle
    return float((pull_side > 15.0).mean())


def build_singles_under(bdf: pd.DataFrame, stand: str | None) -> SinglesUnderProfile:
    """Compute the singles-Under factors from a batter's pitch-level slice."""
    ev = bdf["events"].dropna()
    pa = int(len(ev))
    k_pct = (
        float(ev.isin(["strikeout", "strikeout_double_play"]).sum() / pa)
        if pa
        else float("nan")
    )
    bb_pct = float(ev.eq("walk").sum() / pa) if pa else float("nan")

    desc = bdf["description"]
    in_zone = bdf["zone"].between(1, 9) if "zone" in bdf else pd.Series(dtype=bool)
    zone_pitches = int(in_zone.sum())
    zone_swings = int((in_zone & desc.isin(_SWING_DESC)).sum())
    z_swing = float(zone_swings / zone_pitches) if zone_pitches else float("nan")

    batted = bdf[bdf["launch_speed"].notna()]
    bip = int(len(batted))
    la = batted["launch_angle"].dropna() if "launch_angle" in batted else pd.Series([])
    avg_la = float(la.mean()) if len(la) else float("nan")
    lsa = (
        batted["launch_speed_angle"].dropna()
        if "launch_speed_angle" in batted
        else pd.Series([])
    )
    barrel = float((lsa == 6).mean()) if len(lsa) else float("nan")
    hard_hit = float((batted["launch_speed"] >= 95).mean()) if bip else float("nan")
    pull_rate = _spray_pull_rate(batted, stand)

    return SinglesUnderProfile(
        pa=pa,
        bip=bip,
        k_pct=k_pct,
        bb_pct=bb_pct,
        z_swing=z_swing,
        avg_la=avg_la,
        barrel=barrel,
        hard_hit=hard_hit,
        pull_rate=pull_rate,
    )


def _ok(x: float) -> bool:
    return not math.isnan(x)


def singles_under_score(p: SinglesUnderProfile) -> tuple[float, list[str]]:
    """Weighted NPV score + human-readable reasons for the singles-Under screen.

    Returns ``(0.0, [])`` when the sample is too thin to trust.  Higher scores
    mean a stronger structural case that the batter fails a 1+ singles prop.
    """
    if not p.has_data:
        return 0.0, []

    score = 0.0
    reasons: list[str] = []

    # Tier 1 -- volume killers.
    if _ok(p.k_pct) and _ok(p.bb_pct) and p.k_pct > K_PCT_HI and p.bb_pct > BB_PCT_HI:
        score += 2.0
        reasons.append(f"TTO: K% {p.k_pct:.1%} & BB% {p.bb_pct:.1%}")
    elif _ok(p.k_pct) and p.k_pct > K_PCT_HI:
        score += 1.0
        reasons.append(f"high K% {p.k_pct:.1%}")
    if _ok(p.z_swing) and p.z_swing < Z_SWING_LO:
        score += 1.0
        reasons.append(f"passive Z-Swing% {p.z_swing:.1%}")

    # Tier 2 -- contact cleaners.
    if _ok(p.avg_la) and p.avg_la > LA_FLYBALL:
        score += 1.5
        reasons.append(f"fly-ball tilt (avg LA {p.avg_la:.1f} deg)")
    if _ok(p.barrel) and _ok(p.hard_hit) and p.barrel > BARREL_HI and p.hard_hit > HARD_HIT_HI:
        score += 1.5
        reasons.append(
            f"elite power contact (barrel {p.barrel:.1%}, hard {p.hard_hit:.1%})"
        )
    if (
        _ok(p.pull_rate)
        and _ok(p.avg_la)
        and p.pull_rate > PULL_HI
        and p.avg_la < LA_GROUNDBALL
    ):
        score += 1.0
        reasons.append(f"pull-heavy grounders (pull {p.pull_rate:.1%})")

    return round(score, 2), reasons


@dataclass
class SinglesUnderResult:
    profile: SinglesUnderProfile
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def is_strong(self) -> bool:
        return self.score >= SINGLES_UNDER_STRONG


def evaluate_singles_under(bdf: pd.DataFrame, stand: str | None) -> SinglesUnderResult:
    """Convenience: build the profile and score it in one call."""
    prof = build_singles_under(bdf, stand)
    score, reasons = singles_under_score(prof)
    return SinglesUnderResult(profile=prof, score=score, reasons=reasons)
