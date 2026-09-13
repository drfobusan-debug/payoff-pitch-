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
# The sinker used to share a class with the four-seamer, which made the single
# most important shape distinction in home-run modelling invisible: a high-ride
# four-seamer at the letters and a heavy two-seamer at the knees are opposite
# pitches, and lumping them together averaged one into the other.
_FB = {"FF", "FA", "FC"}  # four-seam, cutter
_SNK = {"SI", "FT"}  # sinker / two-seam
_BRK = {"SL", "ST", "CU", "KC", "SV", "CS", "SC", "KN"}  # slider/sweeper/curve
_OFF = {"CH", "FS", "FO"}  # changeup/splitter

CLASSES = ("FB", "SNK", "BRK", "OFF")

# Rough league baselines per class (whiff = per-swing, swstr = per-pitch).
LEAGUE_WHIFF = {"FB": 0.20, "SNK": 0.14, "BRK": 0.32, "OFF": 0.30}
LEAGUE_SWSTR = {"FB": 0.09, "SNK": 0.06, "BRK": 0.16, "OFF": 0.16}
# The sinker is hit for a respectable average and almost no power, which is the
# whole point of throwing it; the four-seam baseline is the slugging one.
LEAGUE_XWOBA = {"FB": 0.350, "SNK": 0.330, "BRK": 0.280, "OFF": 0.300}

_MIN_PITCHES = 20  # per-class stability floor
_MIN_SWINGS = 12
# Both gains are held out in scripts/arsenal_matchup_study.py: 17,525
# batter-vs-starter matchups, features from the 42-day windows before each game,
# outcomes from the plate appearances in it.
#
# The strikeout side survives at roughly a third of its old strength. It carried
# real information -- holding the log5 vector still, realised/expected moves with
# the term at slope +1.05 (t +5.71) -- but at gain 0.5 it overshot the tail it
# cares about (top quintile priced .3047 where .2770 happened), and the holdout
# preferred an exponent of 0.25-0.40 in both halves. 0.2 is 0.5 x 0.4.
_K_GAIN = 0.2
# The contact side does not survive at all. Per-class xwOBA-on-contact is the
# same contact the batter's own rates are built from, so the term was multiplying
# a vector that already knew: every dose above zero was worse out of sample on
# singles, hits and home runs, in both halves, and holding the baseline still the
# slope is indistinguishable from zero (hits t -0.78, HR t -0.31). Worse, it
# manufactured false positives exactly where the engine buys -- matchups it
# boosted most were priced at .2627 hits and realised .2232.
_K_CLIP = (0.94, 1.06)

_WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
_SWING_DESCS = _WHIFF_DESCS | {"foul", "hit_into_play"}


def pitch_class(code: object) -> str | None:
    if not isinstance(code, str):
        return None
    if code in _FB:
        return "FB"
    if code in _SNK:
        return "SNK"
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
    """Usage-weighted strikeout multiplier for a batter vs. a pitcher's mix.

    Whiffs only. The per-class xwOBA-on-contact that used to move singles,
    doubles and home runs is still read (the reports use it) but no longer
    priced: it failed its holdout in every window.
    """
    total = sum(arsenal.usage.values())
    if total <= 0:
        return {}

    k_factor = 0.0
    for cls, use in arsenal.usage.items():
        w = use / total
        sw = arsenal.swstr.get(cls)
        bw = batter.whiff.get(cls)
        if sw is not None and bw is not None:
            k_rel = (sw / LEAGUE_SWSTR[cls]) * (bw / LEAGUE_WHIFF[cls]) - 1.0
            k_factor += w * k_rel

    k_mult = _clip(1.0 + _K_GAIN * k_factor, *_K_CLIP)
    return {"K": k_mult} if k_mult != 1.0 else {}
