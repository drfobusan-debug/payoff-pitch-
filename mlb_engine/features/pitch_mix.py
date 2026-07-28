"""Arsenal-matching layer: pitch-mix vs. batter pitch-type performance.

Replaces noisy Batter-vs-Pitcher (BvP) box-score head-to-heads with stable
pitch-type splits. Two sticky, high-PPV signals drive it:

- **Pitcher (whiff stability):** per-pitch-class SwStr% and arsenal usage %. A
  pitcher's swinging-strike rate on a given class is one of the stickiest metrics
  in the sport, so it sets the strikeout floor.
- **Hitter (velocity/contact stability):** per-pitch-class whiff rate and xwOBA-
  on-contact. A hitter's ability (or inability) to handle a class is stable.

For each pitch class the pitcher throws, weight by usage and combine the
pitcher's SwStr% relative to league with the hitter's whiff relative to league
(-> K), and the hitter's xwOBA relative to league (-> hits/power). Bounded, and
neutral when a class lacks a stable sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Statcast pitch_type codes -> broad, stable classes.
_FB = {"FF", "FA", "FT", "SI", "FC"}  # four-seam, sinker, cutter
_BRK = {"SL", "ST", "CU", "KC", "SV", "CS", "SC", "KN"}  # slider/sweeper/curve
_OFF = {"CH", "FS", "FO"}  # changeup/splitter

CLASSES = ("FB", "BRK", "OFF")

# Rough league baselines per class (whiff = per-swing, swstr = per-pitch).
LEAGUE_WHIFF = {"FB": 0.20, "BRK": 0.32, "OFF": 0.30}
LEAGUE_SWSTR = {"FB": 0.09, "BRK": 0.16, "OFF": 0.16}
LEAGUE_XWOBA = {"FB": 0.350, "BRK": 0.280, "OFF": 0.300}

_MIN_PITCHES = 20  # per-class stability floor
_MIN_SWINGS = 12
_K_GAIN = 0.5
_HIT_GAIN = 0.5
_K_CLIP = (0.85, 1.15)
_HIT_CLIP = (0.90, 1.10)

_WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
_SWING_DESCS = _WHIFF_DESCS | {"foul", "hit_into_play"}


def pitch_class(code: object) -> str | None:
    if not isinstance(code, str):
        return None
    if code in _FB:
        return "FB"
    if code in _BRK:
        return "BRK"
    if code in _OFF:
        return "OFF"
    return None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ArsenalProfile:
    usage: dict[str, float] = field(default_factory=dict)
    swstr: dict[str, float] = field(default_factory=dict)


@dataclass
class BatterPitchProfile:
    whiff: dict[str, float] = field(default_factory=dict)
    xwoba: dict[str, float] = field(default_factory=dict)


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "pitch_type" not in df.columns:
        return df.assign(_cls=pd.Series(dtype="object"))
    return df.assign(_cls=df["pitch_type"].map(pitch_class))


def build_arsenal(pitcher_rows: pd.DataFrame) -> ArsenalProfile:
    """Recent arsenal usage % and per-class SwStr% for a pitcher."""
    prof = ArsenalProfile()
    df = _classify(pitcher_rows)
    df = df[df["_cls"].notna()]
    if df.empty:
        return prof
    total = len(df)
    desc = df.get("description")
    for cls, grp in df.groupby("_cls"):
        n = len(grp)
        if n < _MIN_PITCHES:
            continue
        prof.usage[str(cls)] = n / total
        if desc is not None:
            whiffs = grp["description"].isin(_WHIFF_DESCS).sum()
            prof.swstr[str(cls)] = float(whiffs) / n
    return prof


def build_batter_pitch_profile(batter_rows: pd.DataFrame) -> BatterPitchProfile:
    """Per-class whiff rate and xwOBA-on-contact for a batter."""
    prof = BatterPitchProfile()
    df = _classify(batter_rows)
    df = df[df["_cls"].notna()]
    if df.empty:
        return prof
    for cls, grp in df.groupby("_cls"):
        c = str(cls)
        swings = grp[grp["description"].isin(_SWING_DESCS)]
        if len(swings) >= _MIN_SWINGS:
            whiffs = swings["description"].isin(_WHIFF_DESCS).sum()
            prof.whiff[c] = float(whiffs) / len(swings)
        bip = grp["estimated_woba_using_speedangle"].dropna()
        if len(bip) >= _MIN_SWINGS:
            prof.xwoba[c] = float(bip.mean())
    return prof


def arsenal_matchup_multiplier(
    arsenal: ArsenalProfile, batter: BatterPitchProfile
) -> dict[str, float]:
    """Usage-weighted K/hit/power multiplier for a batter vs. a pitcher's mix."""
    total = sum(arsenal.usage.values())
    if total <= 0:
        return {}

    k_factor = 0.0
    hit_factor = 0.0
    for cls, use in arsenal.usage.items():
        w = use / total
        sw = arsenal.swstr.get(cls)
        bw = batter.whiff.get(cls)
        if sw is not None and bw is not None:
            k_rel = (sw / LEAGUE_SWSTR[cls]) * (bw / LEAGUE_WHIFF[cls]) - 1.0
            k_factor += w * k_rel
        bx = batter.xwoba.get(cls)
        if bx is not None:
            hit_factor += w * (bx / LEAGUE_XWOBA[cls] - 1.0)

    k_mult = _clip(1.0 + _K_GAIN * k_factor, *_K_CLIP)
    hit_mult = _clip(1.0 + _HIT_GAIN * hit_factor, *_HIT_CLIP)
    out: dict[str, float] = {}
    if k_mult != 1.0:
        out["K"] = k_mult
    if hit_mult != 1.0:
        out["1B"] = hit_mult
        out["2B"] = hit_mult
        out["HR"] = hit_mult
    return out
