"""Team-strength rates (§5.1): per-game numerators and exposures, EB posteriors.

Every metric is ``numerator / exposure * scale`` where both terms are summed over
the games in scope, so a rate as-of a slate is exact (no averaging of averages)
and the exposure ``n`` that feeds the EB posterior is the metric's own
denominator -- 5v5 seconds for a 5v5 rate, PK seconds for a PK rate, all-strength
seconds for a penalty rate. That is why the same October has a stable Corsi
rate and a still-prior-dominated PK rate: the PK has one-tenth the exposure and
a larger fitted ``k``.

Inputs are MoneyPuck score-and-venue-adjusted xG where it exists (5v5), raw xG
for special teams (MoneyPuck does not publish an adjusted PP/PK series), and
score-adjusted shot attempts. Hits/giveaways/takeaways are not metrics here.

``k`` values live in ``nhl_engine.config.ShrinkParams`` with the reliability
study that produced them quoted alongside (``scripts/nhl/reliability_study.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import pandas as pd

from engine_common.shrink import eb_posterior, reliability_at
from nhl_engine.data.moneypuck import as_of

PER_60 = 3600.0


@dataclass(frozen=True)
class Metric:
    """``sum(sign*col for terms) / sum(exposure) * scale`` over games in scope."""

    key: str
    situation: str
    terms: tuple[tuple[str, float], ...]
    exposure_situation: str = ""
    exposure_col: str = "iceTime"
    scale: float = PER_60
    label: str = ""

    @property
    def exp_situation(self) -> str:
        return self.exposure_situation or self.situation


def _m(key: str, situation: str, *terms: tuple[str, float], **kw: object) -> Metric:
    return Metric(key, situation, terms, **kw)  # type: ignore[arg-type]


METRICS: tuple[Metric, ...] = (
    _m("xgf60_5v5", "5on5", ("scoreVenueAdjustedxGoalsFor", 1.0), label="5v5 xGF/60 (adj)"),
    _m("xga60_5v5", "5on5", ("scoreVenueAdjustedxGoalsAgainst", 1.0), label="5v5 xGA/60 (adj)"),
    _m("cf60_5v5", "5on5", ("scoreAdjustedShotsAttemptsFor", 1.0), label="5v5 CF/60 (score-adj)"),
    _m(
        "ca60_5v5",
        "5on5",
        ("scoreAdjustedShotsAttemptsAgainst", 1.0),
        label="5v5 CA/60 (score-adj)",
    ),
    _m("hdxgf60_5v5", "5on5", ("highDangerxGoalsFor", 1.0), label="5v5 high-danger xGF/60"),
    _m("hdxga60_5v5", "5on5", ("highDangerxGoalsAgainst", 1.0), label="5v5 high-danger xGA/60"),
    _m("gf60_5v5", "5on5", ("goalsFor", 1.0), label="5v5 GF/60"),
    _m("ga60_5v5", "5on5", ("goalsAgainst", 1.0), label="5v5 GA/60"),
    _m(
        "fin60_5v5",
        "5on5",
        ("goalsFor", 1.0),
        ("xGoalsFor", -1.0),
        label="5v5 finishing (G - xG)/60",
    ),
    _m(
        "tsv60_5v5",
        "5on5",
        ("xGoalsAgainst", 1.0),
        ("goalsAgainst", -1.0),
        label="5v5 team GSAx/60 (display)",
    ),
    _m("pp_xgf60", "5on4", ("xGoalsFor", 1.0), label="PP xGF/60"),
    _m("pp_xga60", "5on4", ("xGoalsAgainst", 1.0), label="PP xGA/60 (shorthanded against)"),
    _m("pk_xga60", "4on5", ("xGoalsAgainst", 1.0), label="PK xGA/60"),
    _m("sh_xgf60", "4on5", ("xGoalsFor", 1.0), label="SH xGF/60"),
    _m(
        "pp_share",
        "5on4",
        ("iceTime", 1.0),
        exposure_situation="all",
        scale=1.0,
        label="share of game time on the PP",
    ),
    _m(
        "pk_share",
        "4on5",
        ("iceTime", 1.0),
        exposure_situation="all",
        scale=1.0,
        label="share of game time on the PK",
    ),
    _m("pen_drawn60", "all", ("penaltiesAgainst", 1.0), label="penalties drawn/60"),
    _m("pen_taken60", "all", ("penaltiesFor", 1.0), label="penalties taken/60"),
)
METRIC_BY_KEY: dict[str, Metric] = {m.key: m for m in METRICS}


def game_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per ``gameId`` with ``<key>__num`` and ``<key>__exp`` per metric.

    ``games`` is a MoneyPuck team game log (one row per game per situation).
    Games missing a situation row contribute zero numerator and zero exposure
    for metrics in that situation, which is the correct treatment (a game with
    no PK time is no evidence about the PK).
    """
    if games.empty:
        cols = ["gameId", "gameDate", "season", "opposingTeam", "home_or_away"]
        for m in METRICS:
            cols += [f"{m.key}__num", f"{m.key}__exp"]
        return pd.DataFrame(columns=cols)
    by_sit = {s: g.set_index("gameId") for s, g in games.groupby("situation")}
    base = (
        games.drop_duplicates("gameId")
        .set_index("gameId")[["gameDate", "season", "opposingTeam", "home_or_away"]]
        .sort_values(["gameDate"], kind="stable")
    )
    out = base.copy()
    for m in METRICS:
        num = pd.Series(0.0, index=base.index)
        sit = by_sit.get(m.situation)
        if sit is not None:
            for col, sign in m.terms:
                num = num.add(sign * sit[col].reindex(base.index).fillna(0.0), fill_value=0.0)
        exp_sit = by_sit.get(m.exp_situation)
        exp = (
            exp_sit[m.exposure_col].reindex(base.index).fillna(0.0)
            if exp_sit is not None
            else pd.Series(0.0, index=base.index)
        )
        out[f"{m.key}__num"] = num.astype(float)
        out[f"{m.key}__exp"] = exp.astype(float)
    return out.reset_index()


def rate(table: pd.DataFrame, key: str) -> tuple[float, float]:
    """``(rate, exposure)`` for a metric over every row of a game table.

    ``exposure`` is in the metric's denominator units (seconds for per-60 rates).
    A zero exposure returns ``(nan, 0.0)``; the caller shrinks fully to prior.
    """
    m = METRIC_BY_KEY[key]
    num = float(table[f"{key}__num"].sum()) if not table.empty else 0.0
    exp = float(table[f"{key}__exp"].sum()) if not table.empty else 0.0
    if exp <= 0:
        return float("nan"), 0.0
    return num / exp * m.scale, exp


@dataclass(frozen=True)
class Estimate:
    key: str
    observed: float
    exposure: float
    prior: float
    k: float
    posterior: float
    reliability: float
    games: int

    @property
    def prior_weight(self) -> float:
        return 1.0 - self.reliability


def estimate(table: pd.DataFrame, key: str, *, prior: float, k: float) -> Estimate:
    """EB posterior for one metric over ``table`` (already sliced as-of)."""
    observed, exposure = rate(table, key)
    if exposure <= 0:
        return Estimate(key, float("nan"), 0.0, prior, k, prior, 0.0, int(len(table)))
    post = eb_posterior(prior, observed, exposure, k)
    return Estimate(
        key, observed, exposure, prior, k, post, reliability_at(exposure, k), int(len(table))
    )


def team_strength(
    games: pd.DataFrame,
    slate: Date,
    *,
    season: int,
    priors: dict[str, float],
    ks: dict[str, float],
) -> dict[str, Estimate]:
    """Every metric's as-of posterior for one team.

    ``priors`` come from ``data/preseason.py`` (or the league rate when the prior
    asset is missing -- the caller decides and the card says which); ``ks`` from
    ``config.ShrinkParams``.
    """
    table = game_table(as_of(games, slate, season=season))
    return {
        m.key: estimate(table, m.key, prior=priors[m.key], k=ks[m.key])
        for m in METRICS
        if m.key in priors and m.key in ks
    }


def league_rates(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Exposure-weighted league rate per metric across many teams' game tables."""
    out: dict[str, float] = {}
    for m in METRICS:
        num = sum(float(t[f"{m.key}__num"].sum()) for t in tables.values() if not t.empty)
        exp = sum(float(t[f"{m.key}__exp"].sum()) for t in tables.values() if not t.empty)
        out[m.key] = num / exp * m.scale if exp > 0 else float("nan")
    return out


__all__ = [
    "METRICS",
    "METRIC_BY_KEY",
    "PER_60",
    "Estimate",
    "Metric",
    "estimate",
    "game_table",
    "league_rates",
    "rate",
    "team_strength",
]
