"""Within-window form trends for a starting pitcher.

The engine reads a starter over a single six-week window, which is the right
sample for *projection* but hides direction: an arm at a 3.60 SIERA on the way
down and one on the way up price identically. The slate article needs the
direction, so this module splits the same window in half and reports the change
in the three signals that are reliable enough to read on three weeks of pitches
(measured on adjacent blocks: velocity r=0.95, CSW r=0.50, K% r=0.52):

* SIERA — skill-interactive ERA, so K/BB/batted-ball type together,
* stuff — CSW% (called strikes + whiffs per pitch),
* velocity — average four-seam velocity (vFA).

Contact-quality rates are deliberately absent: they barely repeat across six
weeks (xwOBA r=0.31, BABIP r=0.10), so a three-week move in them is noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from mlb_engine.features.regression import CALLED_OR_WHIFF
from mlb_engine.features.siera import pitcher_siera

# A half-window needs this much data before its change is worth printing. Three
# weeks is two or three starts, so the floors sit just under a two-start sample.
MIN_TREND_PITCHES = 120
MIN_TREND_PA = 30
MIN_TREND_FASTBALLS = 20

# Four-seam and the pitches Statcast codes as its variants; vFA is read off
# these rather than all pitches so a change in pitch mix cannot masquerade as a
# change in arm strength.
FASTBALL_TYPES = ("FF", "FA")

# Move sizes below which a trend reads as flat. Roughly one half-window standard
# error each: SIERA runs ~0.35 across half-windows, CSW ~1.5 pts, vFA ~0.3 mph.
FLAT_SIERA = 0.25
FLAT_CSW = 0.010
FLAT_VFA = 0.4


@dataclass(frozen=True)
class Trend:
    """One signal's earlier-half value, recent-half value and change."""

    prior: float | None
    recent: float | None

    @property
    def delta(self) -> float | None:
        if self.prior is None or self.recent is None:
            return None
        return self.recent - self.prior


@dataclass(frozen=True)
class PitcherTrends:
    """Direction of a starter's form inside the window the engine already read."""

    days: int
    siera: Trend
    stuff: Trend  # CSW%
    vfa: Trend  # mph


def _csw(pdf: pd.DataFrame) -> float | None:
    if len(pdf) < MIN_TREND_PITCHES or "description" not in pdf:
        return None
    return float(pdf["description"].isin(CALLED_OR_WHIFF).mean())


def _vfa(pdf: pd.DataFrame) -> float | None:
    if "pitch_type" not in pdf or "release_speed" not in pdf:
        return None
    fb = pdf[pdf["pitch_type"].isin(FASTBALL_TYPES)]["release_speed"].dropna()
    if len(fb) < MIN_TREND_FASTBALLS:
        return None
    return float(fb.mean())


def _siera(pdf: pd.DataFrame) -> float | None:
    if len(pdf) < MIN_TREND_PITCHES:
        return None
    s = pitcher_siera(pdf)
    if s.pa < MIN_TREND_PA:
        return None
    return s.siera


def pitcher_trends(pdf: pd.DataFrame, as_of: Date, days: int) -> PitcherTrends:
    """Split ``days`` in half and read SIERA, CSW% and vFA on each half.

    ``pdf`` is one pitcher's pitch-level Statcast slice. Any half without enough
    data reports ``None`` rather than a number the reader would over-read.
    """
    half = max(days // 2, 1)
    dates = pd.to_datetime(pdf["game_date"]).dt.date if len(pdf) else pd.Series(dtype=object)
    recent_start = as_of - timedelta(days=half)
    prior_start = as_of - timedelta(days=days)
    recent = pdf[dates >= recent_start] if len(pdf) else pdf
    prior = pdf[(dates >= prior_start) & (dates < recent_start)] if len(pdf) else pdf
    return PitcherTrends(
        days=days,
        siera=Trend(prior=_siera(prior), recent=_siera(recent)),
        stuff=Trend(prior=_csw(prior), recent=_csw(recent)),
        vfa=Trend(prior=_vfa(prior), recent=_vfa(recent)),
    )
