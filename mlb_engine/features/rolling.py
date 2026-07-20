"""Rolling-window plate-appearance outcome rates from Statcast pitch data.

The windows follow the user's specification:
  * pitcher form         : last ~4 weeks
  * batter home/away     : last ~3 weeks
  * batter vs RHP        : last ~3 weeks
  * batter vs LHP        : last ~6 weeks

Outcome rates are expressed per plate appearance so they can drive the Monte
Carlo game simulator directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta

import pandas as pd

# Event -> outcome bucket.
HIT_EVENTS = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
WALK_EVENTS = {"walk", "hit_by_pitch"}
K_EVENTS = {"strikeout", "strikeout_double_play"}

# League-average PA outcome rates, used as a Bayesian prior / fallback.
LEAGUE_RATES = {
    "1B": 0.140,
    "2B": 0.045,
    "3B": 0.004,
    "HR": 0.032,
    "BB": 0.085,
    "K": 0.225,
    "OUT": 0.469,
}
PRIOR_STRENGTH = 60.0  # equivalent PA of the league prior


@dataclass
class OutcomeRates:
    """PA-level outcome probabilities (sum to 1)."""

    pa: float
    p_1b: float
    p_2b: float
    p_3b: float
    p_hr: float
    p_bb: float
    p_k: float
    p_out: float

    @property
    def obp(self) -> float:
        return self.p_1b + self.p_2b + self.p_3b + self.p_hr + self.p_bb

    def as_dict(self) -> dict[str, float]:
        return {
            "1B": self.p_1b,
            "2B": self.p_2b,
            "3B": self.p_3b,
            "HR": self.p_hr,
            "BB": self.p_bb,
            "K": self.p_k,
            "OUT": self.p_out,
        }


def _bucket_counts(pa_events: pd.Series) -> dict[str, float]:
    counts = {"1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0, "OUT": 0.0}
    for ev in pa_events.dropna():
        if ev in HIT_EVENTS:
            counts[HIT_EVENTS[ev]] += 1
        elif ev in WALK_EVENTS:
            counts["BB"] += 1
        elif ev in K_EVENTS:
            counts["K"] += 1
        else:
            counts["OUT"] += 1
    return counts


def rates_from_events(pa_events: pd.Series) -> OutcomeRates:
    """Shrink observed counts toward the league prior and normalize to rates."""
    counts = _bucket_counts(pa_events)
    n = sum(counts.values())
    smoothed = {}
    total = n + PRIOR_STRENGTH
    for k in LEAGUE_RATES:
        smoothed[k] = (counts[k] + PRIOR_STRENGTH * LEAGUE_RATES[k]) / total
    return OutcomeRates(
        pa=n,
        p_1b=smoothed["1B"],
        p_2b=smoothed["2B"],
        p_3b=smoothed["3B"],
        p_hr=smoothed["HR"],
        p_bb=smoothed["BB"],
        p_k=smoothed["K"],
        p_out=smoothed["OUT"],
    )


def _slice_dates(df: pd.DataFrame, as_of: Date, days: int) -> pd.DataFrame:
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return df[(df["game_date"] >= start) & (df["game_date"] <= end)]


def _pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that end a plate appearance (non-null events)."""
    return df[df["events"].notna()]


@dataclass
class BatterProfile:
    mlbam_id: int
    home: OutcomeRates
    away: OutcomeRates
    vs_rhp: OutcomeRates
    vs_lhp: OutcomeRates
    overall: OutcomeRates

    def for_context(self, is_home: bool, opp_hand: str | None) -> OutcomeRates:
        """Blend the home/away split with the platoon split."""
        loc = self.home if is_home else self.away
        if opp_hand == "L":
            plat = self.vs_lhp
        elif opp_hand == "R":
            plat = self.vs_rhp
        else:
            return loc
        # Weighted blend by PA of each split (falls back gracefully to priors).
        return _blend(loc, plat)


def _blend(a: OutcomeRates, b: OutcomeRates) -> OutcomeRates:
    wa, wb = a.pa + 1.0, b.pa + 1.0
    tot = wa + wb
    d_a, d_b = a.as_dict(), b.as_dict()
    merged = {k: (d_a[k] * wa + d_b[k] * wb) / tot for k in d_a}
    return OutcomeRates(
        pa=a.pa + b.pa,
        p_1b=merged["1B"],
        p_2b=merged["2B"],
        p_3b=merged["3B"],
        p_hr=merged["HR"],
        p_bb=merged["BB"],
        p_k=merged["K"],
        p_out=merged["OUT"],
    )


def build_batter_profile(
    df: pd.DataFrame,
    batter_id: int,
    as_of: Date,
    home_away_days: int,
    vs_rhp_days: int,
    vs_lhp_days: int,
) -> BatterProfile:
    bdf = _pa_rows(df[df["batter"] == batter_id])

    home_slice = _slice_dates(bdf, as_of, home_away_days)
    # Determine home/away via inning_topbot: batter hits in "Bot" when home.
    home_pa = home_slice[home_slice["inning_topbot"] == "Bot"]["events"]
    away_pa = home_slice[home_slice["inning_topbot"] == "Top"]["events"]

    vs_rhp_pa = _slice_dates(bdf, as_of, vs_rhp_days)
    vs_rhp_pa = vs_rhp_pa[vs_rhp_pa["p_throws"] == "R"]["events"]
    vs_lhp_pa = _slice_dates(bdf, as_of, vs_lhp_days)
    vs_lhp_pa = vs_lhp_pa[vs_lhp_pa["p_throws"] == "L"]["events"]

    overall_pa = _slice_dates(bdf, as_of, max(home_away_days, vs_rhp_days))["events"]

    return BatterProfile(
        mlbam_id=batter_id,
        home=rates_from_events(home_pa),
        away=rates_from_events(away_pa),
        vs_rhp=rates_from_events(vs_rhp_pa),
        vs_lhp=rates_from_events(vs_lhp_pa),
        overall=rates_from_events(overall_pa),
    )


@dataclass
class PitcherProfile:
    mlbam_id: int
    allowed: OutcomeRates  # outcomes allowed per batter faced
    biomech: dict[str, float] = field(default_factory=dict)


def build_pitcher_profile(
    df: pd.DataFrame,
    pitcher_id: int,
    as_of: Date,
    form_days: int,
) -> PitcherProfile:
    pdf = _pa_rows(df[df["pitcher"] == pitcher_id])
    window = _slice_dates(pdf, as_of, form_days)
    return PitcherProfile(
        mlbam_id=pitcher_id,
        allowed=rates_from_events(window["events"]),
    )
