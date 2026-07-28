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

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta

import pandas as pd

# Event -> outcome bucket.
HIT_EVENTS = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
WALK_EVENTS = {"walk", "hit_by_pitch"}
K_EVENTS = {"strikeout", "strikeout_double_play"}

# Total bases per hit event, and the PA-ending events that are not at-bats
# (needed for ISO = SLG - AVG, which is per-AB rather than per-PA).
TB_VALUE = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
NON_AB_EVENTS = {
    "walk",
    "hit_by_pitch",
    "intent_walk",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "catcher_interf",
}

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
MIN_AB_FOR_ISO = 40  # at-bats before a batter's ISO is trusted over no signal
MIN_BBE_FOR_XWOBA = 30  # batted balls before a bullpen's xwOBA allowed is trusted


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


def blend_k_rate(rates: OutcomeRates, k_prior: float, prior_weight: float = 150.0) -> OutcomeRates:
    """Pull the K rate toward a stuff-based prior (xK%), weighted by sample size.

    A thin PA sample leans on the stuff-based expectation; a large sample keeps
    the observed K rate. The remaining (non-K) outcomes are rescaled
    proportionally so the seven outcomes still sum to 1.
    """
    w_obs = max(rates.pa, 0.0)
    total = w_obs + prior_weight
    old_non_k = 1.0 - rates.p_k
    if total <= 0 or old_non_k <= 0:
        return rates
    new_k = (rates.p_k * w_obs + k_prior * prior_weight) / total
    new_k = min(max(new_k, 0.01), 0.60)
    scale = (1.0 - new_k) / old_non_k
    d = rates.as_dict()
    return OutcomeRates(
        pa=rates.pa,
        p_1b=d["1B"] * scale,
        p_2b=d["2B"] * scale,
        p_3b=d["3B"] * scale,
        p_hr=d["HR"] * scale,
        p_bb=d["BB"] * scale,
        p_k=new_k,
        p_out=d["OUT"] * scale,
    )


def blend_bb_rate(rates: OutcomeRates, bb_prior: float, prior_weight: float = 150.0) -> OutcomeRates:
    """Pull the BB rate toward a command-based prior (xBB%), weighted by sample size.

    Mirrors :func:`blend_k_rate`: a thin sample leans on the command-based
    expectation, a large sample keeps the observed rate, and the non-BB outcomes
    are rescaled proportionally so the seven outcomes still sum to 1.
    """
    w_obs = max(rates.pa, 0.0)
    total = w_obs + prior_weight
    old_non_bb = 1.0 - rates.p_bb
    if total <= 0 or old_non_bb <= 0:
        return rates
    new_bb = (rates.p_bb * w_obs + bb_prior * prior_weight) / total
    new_bb = min(max(new_bb, 0.005), 0.30)
    scale = (1.0 - new_bb) / old_non_bb
    d = rates.as_dict()
    return OutcomeRates(
        pa=rates.pa,
        p_1b=d["1B"] * scale,
        p_2b=d["2B"] * scale,
        p_3b=d["3B"] * scale,
        p_hr=d["HR"] * scale,
        p_bb=new_bb,
        p_k=d["K"] * scale,
        p_out=d["OUT"] * scale,
    )


def _slice_dates(df: pd.DataFrame, as_of: Date, days: int) -> pd.DataFrame:
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return df[(df["game_date"] >= start) & (df["game_date"] <= end)]


def _pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that end a plate appearance (non-null events)."""
    return df[df["events"].notna()]


def batter_iso(events: pd.Series, min_ab: int = MIN_AB_FOR_ISO) -> float | None:
    """Isolated power (SLG - AVG) from a batter's PA-ending events.

    ``None`` below ``min_ab`` at-bats: a thin-sample ISO swings wildly and would
    trip a power-based gate on noise alone.
    """
    ab = events.dropna()
    ab = ab[~ab.isin(NON_AB_EVENTS)]
    if len(ab) < min_ab:
        return None
    total_bases = float(ab.map(TB_VALUE).fillna(0).sum())
    hits = float(ab.isin(TB_VALUE).sum())
    return (total_bases - hits) / len(ab)


def lineup_iso(
    df: pd.DataFrame,
    batter_ids: Iterable[int],
    as_of: Date,
    days: int,
    min_ab: int = MIN_AB_FOR_ISO,
) -> float | None:
    """Mean ISO across a lineup, skipping hitters without enough at-bats."""
    ids = {int(b) for b in batter_ids if b}
    if not ids or "batter" not in df.columns:
        return None
    window = _slice_dates(_pa_rows(df[df["batter"].isin(ids)]), as_of, days)
    vals = [
        iso
        for _, rows in window.groupby("batter")
        if (iso := batter_iso(rows["events"], min_ab)) is not None
    ]
    return sum(vals) / len(vals) if vals else None


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


def build_batter_late_rates(
    df: pd.DataFrame,
    batter_id: int,
    as_of: Date,
    days: int,
    min_inning: int = 6,
) -> OutcomeRates:
    """Batter's PA-outcome rates in late innings (>= ``min_inning``) over ``days``.

    Used for the bullpen matchup so hitters are evaluated on the same innings the
    relievers actually work.
    """
    bdf = _pa_rows(df[df["batter"] == batter_id])
    window = _slice_dates(bdf, as_of, days)
    if "inning" in window:
        window = window[window["inning"] >= min_inning]
    return rates_from_events(window["events"])


def bullpen_relief_frame(
    df: pd.DataFrame,
    team_abbrev: str,
    as_of: Date,
    days: int,
    min_inning: int = 6,
) -> pd.DataFrame:
    """Return a team's relief-only pitch rows in late innings over ``days``.

    The fielding team is inferred from ``inning_topbot`` + home/away team codes.
    Each game's starter (any pitcher appearing in the 1st inning) is excluded so
    only true relief appearances (>= ``min_inning``) contribute.
    """
    fielding = df[
        ((df["inning_topbot"] == "Top") & (df["home_team"] == team_abbrev))
        | ((df["inning_topbot"] == "Bot") & (df["away_team"] == team_abbrev))
    ]
    window = _slice_dates(fielding, as_of, days)
    if window.empty or "inning" not in window:
        return window
    starter_pairs = set(
        map(tuple, window.loc[window["inning"] <= 1, ["game_date", "pitcher"]].dropna().to_numpy())
    )
    relief = window[window["inning"] >= min_inning]
    if starter_pairs:
        keys = list(zip(relief["game_date"], relief["pitcher"], strict=False))
        keep = pd.Series([k not in starter_pairs for k in keys], index=relief.index)
        relief = relief[keep]
    return relief


LEVERAGE_INNING = 8  # 8th+ relief = the high-leverage innings the run line hinges on
MIN_LEVERAGE_PA = 20  # need this many 8th+ relief PAs to trust a separate profile


@dataclass
class BullpenProfile:
    """A team's late-inning bullpen: aggregate rates + predictive tripwires."""

    allowed: OutcomeRates
    # Rates allowed by the team's high-leverage arms (8th+ relief). Isolates the
    # closer/setup quality from mop-up men so a shutdown pen and a leaky one stop
    # looking alike late; falls back to ``allowed`` when the 8th+ sample is thin.
    allowed_leverage: OutcomeRates
    relief: pd.DataFrame
    zone_pct: float  # NPV: below ~.40 -> walk trap
    recent_load: float  # NPV: >1 -> heavier 3-day workload than baseline (fatigue)
    xwoba_allowed: float | None = None  # contact quality allowed; None if thin

    @property
    def k_pct(self) -> float:
        return self.allowed.p_k

    def npv_multipliers(self, availability: float | None = None) -> dict[str, float]:
        """Bounded penalty multipliers for the two bullpen NPV tripwires.

        ``availability`` (0..1, higher = more rested) overrides the Statcast
        workload proxy when a usage source such as Rotowire supplies it.
        """
        m: dict[str, float] = {}
        # Walk trap: a low-zone bullpen walks the tying/winning run on in late,
        # high-leverage spots where hitters stop chasing.
        if self.zone_pct and self.zone_pct < 0.40:
            m["BB"] = 1.0 + min(0.20, (0.40 - self.zone_pct) * 2.0)
        # 3-in-4 fatigue: heavy recent workload degrades spin/command -> more damage.
        load = (2.0 - availability) if availability is not None else self.recent_load
        if load > 1.15:
            f = min(0.10, (load - 1.0) * 0.25)
            m["HR"] = 1.0 + f
            m["1B"] = 1.0 + f * 0.5
            m["2B"] = 1.0 + f * 0.5
        return m


def build_bullpen_profile(
    df: pd.DataFrame,
    team_abbrev: str,
    as_of: Date,
    days: int,
    min_inning: int = 6,
) -> BullpenProfile:
    """Aggregate a team's relief corps into rates plus PPV/NPV tripwires."""
    relief = bullpen_relief_frame(df, team_abbrev, as_of, days, min_inning)
    allowed = rates_from_events(_pa_rows(relief)["events"] if len(relief) else pd.Series(dtype=object))

    # High-leverage split: rates from the 8th+ relief PAs only, so the closer/
    # setup corps drives the late-and-close matchup instead of the mop-up-diluted
    # aggregate. Fall back to the aggregate when the 8th+ sample is too thin.
    allowed_leverage = allowed
    if len(relief) and "inning" in relief:
        lev_events = _pa_rows(relief[relief["inning"] >= LEVERAGE_INNING])["events"]
        if len(lev_events) >= MIN_LEVERAGE_PA:
            allowed_leverage = rates_from_events(lev_events)

    zone_pct = (
        float(relief["zone"].between(1, 9).mean())
        if len(relief) and "zone" in relief and relief["zone"].notna().any()
        else 0.0
    )

    xwoba_allowed = None
    if len(relief) and "estimated_woba_using_speedangle" in relief:
        xw = relief["estimated_woba_using_speedangle"].dropna()
        if len(xw) >= MIN_BBE_FOR_XWOBA:
            xwoba_allowed = float(xw.mean())

    recent_load = 0.0
    if len(relief):
        recent_start = (as_of - timedelta(days=1)) - timedelta(days=2)  # last 3 days
        recent = relief[relief["game_date"] >= recent_start]
        expected_3d = len(relief) / days * 3.0
        recent_load = len(recent) / expected_3d if expected_3d > 0 else 0.0

    return BullpenProfile(
        allowed=allowed,
        allowed_leverage=allowed_leverage,
        relief=relief,
        zone_pct=zone_pct,
        recent_load=recent_load,
        xwoba_allowed=xwoba_allowed,
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
