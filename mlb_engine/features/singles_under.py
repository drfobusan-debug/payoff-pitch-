"""Singles "Under" screen -- the batter shape that fails a 1+ singles prop.

The Monte Carlo already reflects opponent defense, the starter's contact profile
and the batter's own singles rate, so this screen is only about what the sim
misses: the batter's structural anti-singles shape.

The five flags this began with came from a framework, with hand-picked weights.
Fitted out of time -- 20,413 batter-games, features from a 42-day window scoring
only the 7 days after it, 13 rolling blocks, target "no single in the game" --
only two survive a joint fit that controls for plate appearances and the
batter's own singles rate:

    k_pct        +0.087  z  5.02   11/13 blocks positive   KEPT
    avg_la       +0.049  z  3.02   11/13 blocks positive   KEPT
    hard_hit     +0.030  z  1.48                           dropped
    bb_pct       +0.022  z  1.38                           dropped
    z_swing      +0.013  z  0.81                           dropped
    barrel       -0.006  z -0.26    6/13 blocks positive   dropped
    pull_rate    -0.057  z -3.78    1/13 blocks positive   dropped, wrong sign

As the flags actually fired, against a 57.6% base rate: high K% 62.8%, fly-ball
tilt 62.2%, **both 65.5%**, elite power contact 58.3% (nothing), passive
Z-Swing% 57.3% (nothing). Pull-heavy grounders fired 36 times in 20,413 and
point the other way -- a pull-heavy batter records *more* singles, not fewer.

So the score is now K% (weight 2.0) plus fly-ball tilt (1.0), in the ratio the
coefficients came out at, and ``SINGLES_UNDER_STRONG`` = 3.0 means both fired --
the 65.5% cell, whose lift over the base rate was positive in 13 of 13 blocks.

The extra profile fields (BB%, Z-Swing%, barrel, hard-hit, pull) are still
computed: they are recorded on the recommendation for the audit, they just no
longer claim to predict a single.

``build_singles_under`` computes the factors from a batter's Statcast slice and
``singles_under_score`` scores them with reasons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import batted_balls

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

# --- factor thresholds ---
K_PCT_HI = 0.25
LA_FLYBALL = 20.0
# Retained for the diagnostic fields, not for scoring.
LA_GROUNDBALL = 5.0
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

    batted = batted_balls(bdf)
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
    """Score + reasons for the singles-Under screen, on the two fitted flags.

    Returns ``(0.0, [])`` when the sample is too thin to trust. 2.0 is the
    strikeout flag, 1.0 the fly-ball flag, so 3.0 -- ``SINGLES_UNDER_STRONG`` --
    is a batter who both misses the ball and lifts it when he doesn't.
    """
    if not p.has_data:
        return 0.0, []

    score = 0.0
    reasons: list[str] = []

    # The ball never gets in play.
    if _ok(p.k_pct) and p.k_pct > K_PCT_HI:
        score += 2.0
        reasons.append(f"high K% {p.k_pct:.1%}")
    # In play, but over the infield rather than through it.
    if _ok(p.avg_la) and p.avg_la > LA_FLYBALL:
        score += 1.0
        reasons.append(f"fly-ball tilt (avg LA {p.avg_la:.1f} deg)")

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
