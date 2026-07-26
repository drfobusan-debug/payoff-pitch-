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
* **WHIP / BB9** -- baserunner traffic. Walks are the #1 out-killer, and a
  starter labouring from the stretch gets a quicker managerial hook regardless
  of his raw pitch economy; both tighten the effective pitch ceiling.
* **Pitch-count cap** -- the effective pitch ceiling: the manager hook tightened
  by the pitcher's own recent pitches-per-start and control (WHIP/BB9).

It also exposes :func:`opponent_discipline_factor`, a lineup-level pitch-economy
multiplier: a patient, high-P/PA-seen offense burns a starter's pitch budget
faster (fewer outs), while a chase-happy lineup lets him work deeper.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from mlb_engine.features.rolling import HIT_EVENTS, WALK_EVENTS

# League baselines (approximate, recalibratable).
LEAGUE_PPA = 3.9  # pitches per plate appearance
BL_F_STRIKE = 0.60
BL_BB_PCT = 0.080
BL_GB_PCT = 0.43
BL_WHIP = 1.30
BL_BB9 = 3.2
BL_HARD_HIT = 0.400  # league hard-hit% (95+ mph) allowed
DEFAULT_PITCH_CAP = 95
MIN_PITCH_CAP = 55  # floor so control/discipline haircuts can't zero the outing

MIN_STARTS = 2  # need at least this many recent starts before trusting the data
MIN_PA = 40  # min plate appearances before trusting realised P/PA over the prior
PITCH_BUFFER = 8  # allow a few more pitches than the recent average (upside outings)

# xP/PA prior from F-Strike%: each point of F-Strike% above baseline shortens the
# expected count. Anchored so a league-average F-Strike% maps to league P/PA.
XPPA_FSTRIKE_COEF = 3.0

# Control-based hook: WHIP/BB9 above baseline tighten the pitch ceiling (traffic
# => quicker exit) on top of raw pitch economy.
CTRL_WHIP_COEF = 0.05
CTRL_BB9_COEF = 0.02


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _control_factor(whip: float, bb9: float) -> float:
    """Pitch-ceiling multiplier (<=1) from baserunner traffic (WHIP/BB9)."""
    f = 1.0 - CTRL_WHIP_COEF * (whip - BL_WHIP) - CTRL_BB9_COEF * (bb9 - BL_BB9)
    return _clip(f, 0.85, 1.0)


@dataclass
class PitcherEfficiency:
    """Per-start efficiency profile driving the starter's out ceiling."""

    pa: int
    pitches: int
    pitches_per_pa: float
    f_strike_pct: float
    bb_pct: float
    gb_pct: float
    whip: float
    bb9: float
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

    def control_cap_factor(self) -> float:
        """How much the pitch ceiling is trimmed by WHIP/BB9 traffic (<=1)."""
        return _control_factor(self.whip, self.bb9)

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


@dataclass(frozen=True)
class RecentStartForm:
    """Blow-up risk over a pitcher's most recent starts.

    WHIP is derived the same way as :class:`PitcherEfficiency` (outs are inferred
    from PA outcomes, so it runs a touch high against true WHIP), and hard-hit%
    is the share of batted balls leaving the bat at 95+ mph. Together they read
    as "traffic plus hard contact", the multi-run-inning script.
    """

    starts: int
    whip: float
    hard_hit_pct: float


def recent_start_form(
    pit_rows: pd.DataFrame, as_of: Date, n_starts: int = 3
) -> RecentStartForm | None:
    """WHIP and hard-hit% allowed over the pitcher's last ``n_starts`` starts.

    ``None`` when fewer than ``n_starts`` starts are on record, so a thin sample
    leaves any gate keyed on this untouched rather than vetoing on noise.
    """
    if "game_date" not in pit_rows or pit_rows.empty:
        return None
    dates = sorted({d for d in pit_rows["game_date"] if d < as_of})
    if len(dates) < n_starts:
        return None
    window = pit_rows[pit_rows["game_date"].isin(dates[-n_starts:])]

    ev = window["events"].dropna() if "events" in window else pd.Series([], dtype=object)
    if ev.empty:
        return None
    walks = int(ev.isin(WALK_EVENTS).sum())
    hits = int(ev.isin(HIT_EVENTS.keys()).sum())
    ip_est = max(len(ev) - hits - walks, 1) / 3.0
    whip = (hits + walks) / ip_est

    batted = window[window["launch_speed"].notna()] if "launch_speed" in window else pd.DataFrame()
    hard_hit = float((batted["launch_speed"] >= 95).mean()) if len(batted) else BL_HARD_HIT

    return RecentStartForm(starts=n_starts, whip=whip, hard_hit_pct=hard_hit)


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
        walks = int(ev.isin(WALK_EVENTS).sum())
        hits = int(ev.isin(HIT_EVENTS.keys()).sum())
        bb_pct = walks / pa
        # Outs proxy: each PA that doesn't reach base is ~one out (ignores DP/sac
        # over-count); IP = outs/3 -> WHIP and BB9 without a box score.
        outs_est = max(pa - hits - walks, 1)
        ip_est = outs_est / 3.0
        whip = (hits + walks) / ip_est
        bb9 = walks / ip_est * 9.0
    else:
        bb_pct = BL_BB_PCT
        whip = BL_WHIP
        bb9 = BL_BB9

    batted = window[window["bb_type"].notna()] if "bb_type" in window else pd.DataFrame()
    gb_pct = float((batted["bb_type"] == "ground_ball").mean()) if len(batted) else BL_GB_PCT

    starts = _pitches_per_start(pit_rows, as_of, form_days)
    if len(starts) >= MIN_STARTS:
        avg = sum(starts) / len(starts)
        pitch_cap = int(min(manager_pitch_cap, round(avg) + PITCH_BUFFER))
    else:
        pitch_cap = manager_pitch_cap
    # Control-based hook: walk-prone / high-traffic starters get pulled sooner.
    pitch_cap = max(int(round(pitch_cap * _control_factor(whip, bb9))), MIN_PITCH_CAP)

    return PitcherEfficiency(
        pa=pa,
        pitches=n_pitches,
        pitches_per_pa=ppa,
        f_strike_pct=f_strike,
        bb_pct=bb_pct,
        gb_pct=gb_pct,
        whip=whip,
        bb9=bb9,
        pitch_cap=pitch_cap,
    )


def opponent_discipline_factor(
    statcast: pd.DataFrame,
    batter_ids: Iterable[int],
    as_of: Date,
    form_days: int,
) -> float:
    """Pitch-economy multiplier from the opposing lineup's plate discipline.

    Returns the lineup's pitches-seen-per-PA relative to league P/PA: >1 for a
    patient offense that runs deep counts (burns the starter's pitch budget
    faster -> fewer outs), <1 for a chase-happy lineup that lets him work deep.
    Fed to the sim as a per-PA pitch-cost multiplier on top of the pitcher's own
    efficiency. K/BB effects are intentionally left to the batter rate model to
    avoid double-counting -- this is purely the pitch-count channel.
    """
    ids = {int(b) for b in batter_ids if b}
    if not ids or "batter" not in statcast.columns:
        return 1.0
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=form_days - 1)
    rows = statcast[statcast["batter"].isin(ids)]
    if "game_date" in rows.columns:
        rows = rows[(rows["game_date"] >= start) & (rows["game_date"] <= end)]
    pa = int(rows["events"].notna().sum()) if "events" in rows.columns else 0
    if pa < MIN_PA:
        return 1.0
    team_ppa = len(rows) / pa
    return _clip(team_ppa / LEAGUE_PPA, 0.90, 1.12)
