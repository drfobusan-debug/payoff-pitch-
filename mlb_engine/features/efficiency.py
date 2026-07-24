"""Pitch-efficiency model for starter out (innings-pitched) volume.

Recording outs is mathematically identical to pitching innings (1 IP = 3 outs),
and a starter's out ceiling is governed by *how many pitches he needs to record
those outs* before the manager's pitch-count hook pulls him. Two starters can
face the same number of batters yet throw very different pitch counts; the flat
batters-faced cap in :mod:`mlb_engine.features.workload` cannot see that.

This derives, from a pitcher's recent Statcast starts, the efficiency inputs the
Monte Carlo needs to model a realistic exit point:

* **Pitches per plate appearance (P/PA)** -- the direct proxy for Pit/IP and the
  single strongest driver of out volume. Fed to the sim as a per-PA pitch-cost
  scaler.
* **First-pitch-strike% (F-Strike%)** -- stabilises far faster than raw P/PA, so
  it anchors a small-sample prior for P/PA (high F-Strike% => shorter counts).
* **Walk% (BB%)** -- free passes fail to record outs and bloat the pitch count.
* **Ground-ball% (GB%)** -- ground-ball arms generate quick, low-pitch outs and
  the occasional double play (two outs on one ball in play).
* **Pitch-count cap** -- the effective pitch ceiling: the manager hook tightened
  by the pitcher's own recent pitches-per-start.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import pandas as pd

# League baselines (approximate, recalibratable).
LEAGUE_PPA = 3.9  # pitches per plate appearance
BL_F_STRIKE = 0.60
BL_BB_PCT = 0.080
BL_GB_PCT = 0.43
DEFAULT_PITCH_CAP = 95

MIN_STARTS = 2  # need at least this many recent starts before trusting the data
MIN_PA = 40  # min plate appearances before trusting realised P/PA over the prior
PITCH_BUFFER = 8  # allow a few more pitches than the recent average (upside outings)

# xP/PA prior from F-Strike%: each point of F-Strike% above baseline shortens the
# expected count. Anchored so a league-average F-Strike% maps to league P/PA.
XPPA_FSTRIKE_COEF = 3.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class PitcherEfficiency:
    """Per-start efficiency profile driving the starter's out ceiling."""

    pa: int
    pitches: int
    pitches_per_pa: float
    f_strike_pct: float
    bb_pct: float
    gb_pct: float
    pitch_cap: int

    def expected_pitches_per_pa(self) -> float:
        """F-Strike%-anchored P/PA prior (stabilises faster than raw P/PA)."""
        xppa = LEAGUE_PPA - XPPA_FSTRIKE_COEF * (self.f_strike_pct - BL_F_STRIKE)
        return _clip(xppa, 3.2, 4.6)

    def blended_pitches_per_pa(self) -> float:
        """Realised P/PA regressed toward the F-Strike% prior by sample size."""
        prior = self.expected_pitches_per_pa()
        if self.pa <= 0:
            return prior
        w = min(self.pa, MIN_PA) / MIN_PA
        blended = w * self.pitches_per_pa + (1.0 - w) * prior
        return _clip(blended, 3.2, 4.8)

    def efficiency_scaler(self) -> float:
        """Per-PA pitch-cost multiplier vs a league-average count length."""
        return _clip(self.blended_pitches_per_pa() / LEAGUE_PPA, 0.80, 1.25)

    def gb_dp_rate(self) -> float:
        """Probability a ground-ball out becomes a double play (runner on first).

        Scales with the pitcher's GB% around a ~12% league DP-conversion anchor,
        bounded so no single arm turns two on demand.
        """
        return _clip(0.12 * (self.gb_pct / BL_GB_PCT), 0.05, 0.22)


# First pitch of a plate appearance is a 0-0 count. Statcast ``type``: S=strike,
# B=ball, X=in play. A first-pitch strike counts strikes and balls put in play.
_STRIKE_TYPES = {"S", "X"}


def _pitches_per_start(pit_rows: pd.DataFrame, as_of: Date, form_days: int) -> list[int]:
    """Pitch count in each of the pitcher's starts over the form window."""
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=form_days - 1)
    if "game_date" not in pit_rows:
        return []
    window = pit_rows[(pit_rows["game_date"] >= start) & (pit_rows["game_date"] <= end)]
    if window.empty:
        return []
    return [int(n) for n in window.groupby("game_date").size().tolist()]


def build_pitcher_efficiency(
    pit_rows: pd.DataFrame,
    as_of: Date,
    form_days: int,
    manager_pitch_cap: int = DEFAULT_PITCH_CAP,
) -> PitcherEfficiency:
    """Derive the efficiency profile from a pitcher's recent Statcast slice."""
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=form_days - 1)
    if "game_date" in pit_rows:
        window = pit_rows[(pit_rows["game_date"] >= start) & (pit_rows["game_date"] <= end)]
    else:
        window = pit_rows

    n_pitches = int(len(window))
    ev = window["events"].dropna() if "events" in window else pd.Series([], dtype=object)
    pa = int(len(ev))
    ppa = n_pitches / pa if pa else LEAGUE_PPA

    # First-pitch-strike%: 0-0 counts resolved as a strike or ball-in-play.
    if pa and {"balls", "strikes", "type"}.issubset(window.columns):
        first = window[(window["balls"] == 0) & (window["strikes"] == 0)]
        n_first = int(len(first))
        f_strike = float(first["type"].isin(_STRIKE_TYPES).mean()) if n_first else BL_F_STRIKE
    else:
        f_strike = BL_F_STRIKE

    if pa:
        bb_pct = float(ev.isin(["walk", "hit_by_pitch"]).sum() / pa)
    else:
        bb_pct = BL_BB_PCT

    batted = window[window["bb_type"].notna()] if "bb_type" in window else pd.DataFrame()
    gb_pct = float((batted["bb_type"] == "ground_ball").mean()) if len(batted) else BL_GB_PCT

    starts = _pitches_per_start(pit_rows, as_of, form_days)
    if len(starts) >= MIN_STARTS:
        avg = sum(starts) / len(starts)
        pitch_cap = int(min(manager_pitch_cap, round(avg) + PITCH_BUFFER))
    else:
        pitch_cap = manager_pitch_cap

    return PitcherEfficiency(
        pa=pa,
        pitches=n_pitches,
        pitches_per_pa=ppa,
        f_strike_pct=f_strike,
        bb_pct=bb_pct,
        gb_pct=gb_pct,
        pitch_cap=pitch_cap,
    )
