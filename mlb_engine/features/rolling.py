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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Event -> outcome bucket.
HIT_EVENTS = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
# Walks charged to the pitcher. The intentional walk is one of them, and leaving it
# out sent it to the bucketer's ``else`` and counted it as an out. It lands on the
# hitters the engine bets most -- Alvarez, Ohtani, Soto and Schwarber are walked on
# purpose precisely because they are the ones worth walking -- so the miss was
# concentrated on the best bats rather than spread thin across the league.
WALK_EVENTS = {"walk", "hit_by_pitch", "intent_walk"}
# Reaching first without a hit, an out, or anything the pitcher did. Kept apart from
# WALK_EVENTS because that set also stands in for a pitcher's walks allowed, and
# interference is the catcher's doing.
FREE_PASS_EVENTS = WALK_EVENTS | {"catcher_interf"}
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

# Equivalent PA of the *hierarchical* prior: how hard a batter's home/away and
# platoon splits are pulled toward his own overall rate before anything is
# pulled toward the league.
#
# A 21-day home split is roughly 40 PA, so under a flat 60-PA league prior a
# hitter arrived at the simulator already ~60% league-average -- and that
# compression is measurable in the graded ledger. Splitting batter_h o0.5 by
# hitter quality, the realised gap between the worst and best quartile was 9.3
# points while the model priced 3.7, and essentially all of the error sat on
# weak hitters (bottom quartile priced .577, won .493). Weak bats priced as
# near-average look underpriced against a market that has them right, which is
# how they become false-positive buys.
#
# The fix is hierarchical rather than heavier: a split regresses toward *this
# batter's* overall rate, and only the overall regresses toward the league. A
# thin platoon split for a poor hitter then lands on that poor hitter's own
# baseline instead of on an average major leaguer.
SPLIT_PRIOR_STRENGTH = 60.0

# Equivalent PA of prior each outcome deserves *individually*. A flat 60 assumes
# every bucket is equally noisy, and they are not: for a rate observed over n PA
# the spread across hitters is talent plus binomial noise p(1-p)/n, so the prior
# strength that minimises squared error is k = p(1-p)/var(talent), which differs
# by an order of magnitude between a strikeout and a double.
#
# Fitted on 280 hitters with 60+ PA in the 42 days to 2026-08-11
# (``scripts.ros_prior_study shrink``), with talent variance estimated two
# independent ways -- the spread of THE BAT X rest-of-season projections, and the
# window's own spread with binomial noise subtracted -- taking whichever implies
# the *lighter* shrinkage:
#
#     outcome   talent sd (ROS)   talent sd (window)   k
#     1B             0.0192            0.0261         181
#     2B             0.0046            0.0055        1247
#     3B             0.0023            0.0031         423
#     HR             0.0122            0.0130         181
#     BB             0.0212            0.0293         101
#     K              0.0543            0.0591          48
#     OUT            0.0498            0.0604          68
#
# So the flat 60 was about right for strikeouts and outs -- which is why the
# engine's best-ranked markets are the K-driven ones -- and far too light
# everywhere else. A hitter's 42-day doubles rate is almost entirely noise: the
# spread we were feeding the simulator was 2.8x the spread of real talent and
# correlated 0.20 with it. That is the mechanism behind a model that knows a
# hitter will get a hit but not which kind.
#
# Heavier shrinkage is only safe with somewhere true to shrink *to*. Toward the
# league mean it would flatten the lineup and make the ranking worse; toward the
# hitter's own projection it removes noise and keeps the hitter. The two changes
# are one change.
OUTCOME_PRIOR_STRENGTH = {
    "1B": 181.0,
    "2B": 1247.0,
    "3B": 423.0,
    "HR": 181.0,
    "BB": 101.0,
    "K": 48.0,
    "OUT": 68.0,
}

# The same estimator on the 30 bullpens, whose aggregate is the thinnest sample
# the engine trusts (a mean 265 relief PA in 21 days). Talent variance across
# pens is measured from the spread of usage-weighted rest-of-season projections
# for each team's relief arms and from the pens' own spread net of binomial
# noise, again taking the lighter shrinkage (``scripts.ros_prior_study pen``):
#
#     bucket   talent sd   noise in the window at w=0.82   k
#     1B         0.0081              0.0174              1834
#     2B         0.0000              0.0097               cap
#     3B         0.0000              0.0032               cap
#     HR         0.0059              0.0079               708
#     BB         0.0163              0.0151               344
#     K          0.0268              0.0209               241
#     OUT        0.0246              0.0250               410
#
# Pens are far more alike than hitters -- a pen's strikeout rate varies half as
# much as a hitter's -- so at a flat 60 PA the vector the simulator receives is
# mostly sampling error: noise exceeds real spread in every bucket but K and BB.
#
# The two extra-base buckets are the sharp result: **across 30 pens, the entire
# observed spread in doubles and triples allowed is explained by binomial noise**,
# leaving no measurable talent at all. A pen's own doubles rate over three weeks
# carries nothing, so it is shrunk effectively to the league pen, which is what
# the cap expresses.
#
# Unlike the hitter case the projection matters here as the *estimator* rather
# than as a target: pens differ so little that shrinking toward the league pen
# instead of toward each pen's own projection costs under a point.
PEN_PRIOR_CAP = 5000.0  # "use the league pen", written as an equivalent-PA weight
PEN_PRIOR_STRENGTH = {
    "1B": 1834.0,
    "2B": PEN_PRIOR_CAP,
    "3B": PEN_PRIOR_CAP,
    "HR": 708.0,
    "BB": 344.0,
    "K": 241.0,
    "OUT": 410.0,
}
MIN_AB_FOR_ISO = 40  # at-bats before a batter's ISO is trusted over no signal
MIN_BBE_FOR_XWOBA = 30  # batted balls before a bullpen's xwOBA allowed is trusted

# League-average relief xwOBA allowed, the target a thin three-week bullpen read
# is shrunk toward. Measured over 2026-06-16..07-27, 30 pens, ~17k batters faced.
LEAGUE_PEN_XWOBA = 0.306


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


OUTCOMES_ORDER = ("1B", "2B", "3B", "HR", "BB", "K", "OUT")

# Columns a FanGraphs rest-of-season projection export must carry for the rate
# vector to be built from it. ``1B`` is published directly; if a future export
# drops it, it is H - 2B - 3B - HR.
ROS_COLUMNS = ("PA", "H", "2B", "3B", "HR", "BB", "SO", "MLBAMID")


def ros_rates_from_projection(df: pd.DataFrame) -> pd.DataFrame:
    """Per-PA outcome rates from a rest-of-season projection export.

    A projection is a talent estimate, so it is the right shrinkage target for a
    thin window: the league mean pulls every hitter toward the same place, which
    is exactly what a model that has to *rank* hitters cannot afford.

    Returns one row per MLBAM id with the seven simulator outcomes, which sum to
    1 by construction. HBP is folded into BB, as the simulator has no separate
    bucket for it and both put a man on first.
    """
    missing = [c for c in ROS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"projection export is missing {missing}")
    d = df[df["PA"] > 0].copy()
    singles = d["1B"] if "1B" in d.columns else d["H"] - d["2B"] - d["3B"] - d["HR"]
    pa = d["PA"].astype(float)
    walks = d["BB"].astype(float) + d.get("HBP", 0.0)
    rates = pd.DataFrame(
        {
            "mlbam_id": d["MLBAMID"].astype("Int64"),
            "pa": pa,
            "1B": singles.astype(float) / pa,
            "2B": d["2B"].astype(float) / pa,
            "3B": d["3B"].astype(float) / pa,
            "HR": d["HR"].astype(float) / pa,
            "BB": walks / pa,
            "K": d["SO"].astype(float) / pa,
        }
    ).dropna(subset=["mlbam_id"])
    on_base_or_k = rates[["1B", "2B", "3B", "HR", "BB", "K"]].sum(axis=1)
    rates["OUT"] = (1.0 - on_base_or_k).clip(lower=0.0)
    total = rates[list(OUTCOMES_ORDER)].sum(axis=1)
    for oc in OUTCOMES_ORDER:
        rates[oc] = rates[oc] / total
    return rates.drop_duplicates("mlbam_id").reset_index(drop=True)


def load_ros_priors(path: str | Path) -> dict[int, dict[str, float]]:
    """MLBAM id -> prior rate vector, from a file written by ``scripts.ros_prior_study``.

    A missing file is not an error: the engine falls back to the league prior,
    which is what it has always used.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if "mlbam_id" not in df.columns:
        return {}
    if any(oc not in df.columns for oc in OUTCOMES_ORDER):
        return {}
    out: dict[int, dict[str, float]] = {}
    for row in df.to_dict("records"):
        vec = {oc: float(row[oc]) for oc in OUTCOMES_ORDER}
        if any(v != v for v in vec.values()):  # NaN
            continue
        out[int(row["mlbam_id"])] = vec
    return out


def raw_window_counts(
    df: pd.DataFrame, batter_id: int, as_of: Date, days: int
) -> tuple[dict[str, float], float]:
    """Unshrunk outcome counts for one batter's window, and its PA total.

    Exposed for the shrinkage study: fitting how hard a bucket should be
    regressed needs the raw counts, because shrinkage is the thing being fitted.
    """
    rows = _slice_dates(_pa_rows(df[df["batter"] == batter_id]), as_of, days)
    counts = _bucket_counts(rows["events"])
    return counts, sum(counts.values())


def pa_outcome_counts(frame: pd.DataFrame) -> dict[str, float]:
    """Outcome counts for any frame of pitch rows, keeping only PA-ending ones."""
    if frame.empty or "events" not in frame:
        return dict.fromkeys(OUTCOMES_ORDER, 0.0)
    return _bucket_counts(_pa_rows(frame)["events"])


def _bucket_counts(pa_events: pd.Series) -> dict[str, float]:
    counts = {"1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0, "OUT": 0.0}
    for ev in pa_events.dropna():
        if ev in HIT_EVENTS:
            counts[HIT_EVENTS[ev]] += 1
        elif ev in FREE_PASS_EVENTS:
            counts["BB"] += 1
        elif ev in K_EVENTS:
            counts["K"] += 1
        else:
            counts["OUT"] += 1
    return counts


def rates_from_events(
    pa_events: pd.Series,
    prior: Mapping[str, float] | None = None,
    prior_strength: float | Mapping[str, float] = PRIOR_STRENGTH,
) -> OutcomeRates:
    """Shrink observed counts toward a prior and normalize to rates.

    ``prior`` defaults to the league rates. Passing a batter's own overall rates
    instead makes the shrinkage hierarchical: the split regresses toward the
    hitter rather than toward an average major leaguer.

    ``prior_strength`` may be a single equivalent-PA figure, or one per outcome
    (see ``OUTCOME_PRIOR_STRENGTH``) when the buckets differ in how noisy they
    are. Per-outcome shrinkage gives each bucket its own denominator, so the
    seven rates are renormalized to sum to 1 afterwards.
    """
    target = LEAGUE_RATES if prior is None else prior
    counts = _bucket_counts(pa_events)
    n = sum(counts.values())
    smoothed = {}
    for k in LEAGUE_RATES:
        strength = (
            prior_strength[k] if isinstance(prior_strength, Mapping) else prior_strength
        )
        smoothed[k] = (counts[k] + strength * target[k]) / (n + strength)
    scale = sum(smoothed.values())
    if scale > 0:
        smoothed = {k: v / scale for k, v in smoothed.items()}
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


# Home runs are the rarest outcome the simulator draws, so an observed HR/PA is
# the noisiest of the rates -- and unlike K and BB it is heavily contaminated by
# the parks and weather a hitter happened to face. Expected HR is both steadier
# and the better predictor of future home runs, so it gets a heavier prior than
# the K/BB blends: the observed rate only takes over past a full season's PAs.
HR_PRIOR_WEIGHT = 200.0


def blend_hr_rate(
    rates: OutcomeRates, xhr_prior: float, prior_weight: float = HR_PRIOR_WEIGHT
) -> OutcomeRates:
    """Pull the HR rate toward an expected-HR prior (xHR/PA) by sample size.

    Mirrors :func:`blend_k_rate`: a thin or lucky sample leans on what the
    batted balls were physically worth against the walls they were hit toward,
    a large sample keeps more of the observed rate, and the non-HR outcomes are
    rescaled proportionally so the seven outcomes still sum to 1. A NaN prior
    (no distance data) leaves the rates untouched.
    """
    if xhr_prior != xhr_prior or prior_weight <= 0:  # NaN prior
        return rates
    w_obs = max(rates.pa, 0.0)
    total = w_obs + prior_weight
    old_non_hr = 1.0 - rates.p_hr
    if total <= 0 or old_non_hr <= 0:
        return rates
    new_hr = (rates.p_hr * w_obs + xhr_prior * prior_weight) / total
    new_hr = min(max(new_hr, 0.001), 0.15)
    scale = (1.0 - new_hr) / old_non_hr
    d = rates.as_dict()
    return OutcomeRates(
        pa=rates.pa,
        p_1b=d["1B"] * scale,
        p_2b=d["2B"] * scale,
        p_3b=d["3B"] * scale,
        p_hr=new_hr,
        p_bb=d["BB"] * scale,
        p_k=d["K"] * scale,
        p_out=d["OUT"] * scale,
    )


def scale_hr_rate(rates: OutcomeRates, mult: float) -> OutcomeRates:
    """Scale the HR rate by a multiplier, rescaling the rest to still sum to 1.

    Used for effects that act on the park rather than the hitter, where the
    right operation is "this contact is worth more here" and not a change to
    what the contact was. A NaN multiplier leaves the rates untouched.
    """
    if mult != mult or mult <= 0.0:  # NaN or nonsense
        return rates
    new_hr = min(max(rates.p_hr * mult, 0.001), 0.15)
    old_non_hr = 1.0 - rates.p_hr
    if old_non_hr <= 0:
        return rates
    scale = (1.0 - new_hr) / old_non_hr
    d = rates.as_dict()
    return OutcomeRates(
        pa=rates.pa,
        p_1b=d["1B"] * scale,
        p_2b=d["2B"] * scale,
        p_3b=d["3B"] * scale,
        p_hr=new_hr,
        p_bb=d["BB"] * scale,
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
    split_prior: bool = True,
    ros_prior: Mapping[str, float] | None = None,
) -> BatterProfile:
    """Build a batter's context splits.

    ``split_prior`` regresses each split toward the batter's own overall rate;
    set it False to restore the flat league-average prior on every split.

    ``ros_prior`` is this hitter's rest-of-season projection, used as the target
    the overall window regresses toward, and it switches the overall window onto
    the per-outcome prior strengths. Without it the target is the league mean at
    a flat 60 PA, which is what the engine has always done: heavy shrinkage
    toward an average major leaguer would flatten the lineup rather than clean
    it, so the projection and the fitted strengths arrive together or not at all.
    """
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

    # Hierarchical: the overall window regresses toward the league, then each
    # split regresses toward *this hitter's* overall rather than toward an
    # average major leaguer. A 40-PA home split for a weak bat then lands on his
    # own baseline instead of being handed most of a league-average profile.
    if ros_prior is None:
        overall = rates_from_events(overall_pa)
    else:
        overall = rates_from_events(overall_pa, ros_prior, OUTCOME_PRIOR_STRENGTH)
    prior = overall.as_dict() if split_prior else None
    return BatterProfile(
        mlbam_id=batter_id,
        home=rates_from_events(home_pa, prior, SPLIT_PRIOR_STRENGTH),
        away=rates_from_events(away_pa, prior, SPLIT_PRIOR_STRENGTH),
        vs_rhp=rates_from_events(vs_rhp_pa, prior, SPLIT_PRIOR_STRENGTH),
        vs_lhp=rates_from_events(vs_lhp_pa, prior, SPLIT_PRIOR_STRENGTH),
        overall=overall,
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
# Relief before the 8th is the *bridge*: the middle men who cover the innings
# between the starter's hook and the setup/closer pair. Read separately because
# a pen's 8th+ arms are its two best and its bridge arms are not, so charging a
# 6th-inning hand-off the closer's rates overrates every pen -- most of all the
# good ones, whose leverage-to-bridge gap is widest.
MIN_BRIDGE_PA = 20


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
    # Rates allowed in relief before ``LEVERAGE_INNING`` (the bridge innings).
    # None when that sample is too thin to read, in which case ``bridge`` serves
    # the aggregate.
    allowed_bridge: OutcomeRates | None = None
    # Relief rows over the longer skill window (K%, whiff, velocity persist far
    # better than results do); the short window when no skill window is set.
    skill: pd.DataFrame | None = None
    xwoba_raw: float | None = None  # pre-shrinkage mean, for reporting

    @property
    def k_pct(self) -> float:
        return self.allowed.p_k

    @property
    def bridge(self) -> OutcomeRates:
        """Rates for the innings between the starter's hook and the 8th."""
        return self.allowed if self.allowed_bridge is None else self.allowed_bridge

    @property
    def skill_frame(self) -> pd.DataFrame:
        """Rows to read persistent stuff/command signals from."""
        return self.relief if self.skill is None else self.skill

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


# Linear wOBA weights (FanGraphs scale). HBP is folded into BB, which the
# outcome model does not separate.
WOBA_WEIGHTS = {
    "BB": 0.690,
    "1B": 0.880,
    "2B": 1.247,
    "3B": 1.578,
    "HR": 2.031,
    "K": 0.0,
    "OUT": 0.0,
}
MIN_ARM_PA = 25  # batters faced before one reliever gets his own wOBA line


def woba_from_rates(rates: dict[str, float]) -> float:
    """wOBA implied by a set of PA-outcome probabilities.

    Puts a log5 matchup on the same scale as a season wOBA line, so "this order
    vs this arm" and "this order vs an average arm" are comparable numbers.
    """
    return sum(WOBA_WEIGHTS[k] * rates.get(k, 0.0) for k in WOBA_WEIGHTS)


def pen_arm_spread(relief: pd.DataFrame) -> tuple[float | None, int]:
    """Spread of wOBA allowed across a bullpen's individual arms.

    How much the pen's aggregate line depends on *which* reliever appears: a
    uniform corps and one carried by a closer in front of four leaky arms have
    the same mean and very different variance late in a one-run game. Returns
    the population standard deviation over arms with at least ``MIN_ARM_PA``
    batters faced, and how many arms that is.
    """
    if not len(relief) or "pitcher" not in relief or "events" not in relief:
        return None, 0
    wobas = []
    for _, arm in relief.groupby("pitcher"):
        events = arm["events"].dropna()
        if len(events) < MIN_ARM_PA:
            continue
        wobas.append(woba_from_rates(rates_from_events(events).as_dict()))
    if len(wobas) < 2:
        return None, len(wobas)
    mean = sum(wobas) / len(wobas)
    var = sum((w - mean) ** 2 for w in wobas) / len(wobas)
    return round(var**0.5, 3), len(wobas)


def shrink_pen_xwoba(raw: float, weight: float, league: float = LEAGUE_PEN_XWOBA) -> float:
    """Pull a bullpen's observed xwOBA allowed toward the league mean.

    ``weight`` is the share of the team's deviation to keep, i.e. the measured
    split-half reliability of the window. Three weeks of relief work is ~270
    batters faced across a dozen arms and repeats at r=0.37, so roughly two
    thirds of the distance from league average is noise.
    """
    w = min(max(weight, 0.0), 1.0)
    return league + w * (raw - league)


def build_bullpen_profile(
    df: pd.DataFrame,
    team_abbrev: str,
    as_of: Date,
    days: int,
    min_inning: int = 6,
    skill_days: int = 0,
    xwoba_shrink: float = 1.0,
    prior_strength: float | Mapping[str, float] = PRIOR_STRENGTH,
) -> BullpenProfile:
    """Aggregate a team's relief corps into rates plus PPV/NPV tripwires.

    ``prior_strength`` accepts one figure per outcome (``PEN_PRIOR_STRENGTH``);
    the leverage and bridge subsets are thinner still, so they take the same
    strengths as the aggregate.

    ``days`` covers the results-based rates, which are best read recently.
    ``skill_days`` (0 to disable) covers the stuff and command signals, which
    are best read over a longer window: measured out of sample against the
    following three weeks, relief K% correlates 0.73 on 42 days against 0.66 on
    21, and jointly the longer read carries the weight (+0.68 vs +0.14).
    """
    relief = bullpen_relief_frame(df, team_abbrev, as_of, days, min_inning)
    allowed = rates_from_events(
        _pa_rows(relief)["events"] if len(relief) else pd.Series(dtype=object),
        prior_strength=prior_strength,
    )

    # High-leverage split: rates from the 8th+ relief PAs only, so the closer/
    # setup corps drives the late-and-close matchup instead of the mop-up-diluted
    # aggregate. Fall back to the aggregate when the 8th+ sample is too thin.
    allowed_leverage = allowed
    allowed_bridge = None
    if len(relief) and "inning" in relief:
        lev_events = _pa_rows(relief[relief["inning"] >= LEVERAGE_INNING])["events"]
        if len(lev_events) >= MIN_LEVERAGE_PA:
            allowed_leverage = rates_from_events(lev_events, prior_strength=prior_strength)
        bridge_events = _pa_rows(relief[relief["inning"] < LEVERAGE_INNING])["events"]
        if len(bridge_events) >= MIN_BRIDGE_PA:
            allowed_bridge = rates_from_events(bridge_events, prior_strength=prior_strength)

    zone_pct = (
        float(relief["zone"].between(1, 9).mean())
        if len(relief) and "zone" in relief and relief["zone"].notna().any()
        else 0.0
    )

    xwoba_raw = None
    xwoba_allowed = None
    if len(relief) and "estimated_woba_using_speedangle" in relief:
        xw = relief["estimated_woba_using_speedangle"].dropna()
        if len(xw) >= MIN_BBE_FOR_XWOBA:
            xwoba_raw = float(xw.mean())
            xwoba_allowed = shrink_pen_xwoba(xwoba_raw, xwoba_shrink)

    recent_load = 0.0
    if len(relief):
        recent_start = (as_of - timedelta(days=1)) - timedelta(days=2)  # last 3 days
        recent = relief[relief["game_date"] >= recent_start]
        expected_3d = len(relief) / days * 3.0
        recent_load = len(recent) / expected_3d if expected_3d > 0 else 0.0

    skill = (
        bullpen_relief_frame(df, team_abbrev, as_of, skill_days, min_inning)
        if skill_days > days
        else None
    )

    return BullpenProfile(
        allowed=allowed,
        allowed_leverage=allowed_leverage,
        relief=relief,
        zone_pct=zone_pct,
        recent_load=recent_load,
        xwoba_allowed=xwoba_allowed,
        allowed_bridge=allowed_bridge,
        skill=skill,
        xwoba_raw=xwoba_raw,
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
