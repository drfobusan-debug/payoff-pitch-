"""Goalie layer (§5.2): GSAx/60 as the skill input, Sv% display-only.

``goalie_game_table`` reduces a MoneyPuck goalie game log (all situations) to
one row per appearance with the two rates the engine cares about:

* ``gsax60`` -- (xGoals against - goals against) per 60, the skill input. Using
  xG per shot faced is what removes the "new-team mirage": stopping low-danger
  shots behind a great defence earns little credit.
* ``svpct`` -- saves / shots on goal. Carried for the card; never enters a prior.

``goalie_skill`` builds the as-of estimate in two EB steps:

1. a **career prior**: previous seasons' GSAx/60, each discounted by
   ``r_yy ** lag`` (the fitted year-to-year correlation, the AR(1) shortcut for
   "last season tells you more than the one before"), shrunk toward the league
   rate with ``k_career``;
2. the **current season** shrunk toward that prior with ``k_season``.

A goalie under ``min_minutes`` of NHL history gets the ``callup_prior`` instead
of the league rate as his anchor (§5.2: the empirical distribution of call-up
starts, fitted by ``scripts/nhl/goalie_study.py``; ``None`` until that study has
run, in which case the league rate is used and ``anchor`` says so).

Also exposed per appearance: ``days_rest`` (time since last start, the TSLS
input) and cumulative ``career_toi`` before the game, so the studies and the
card read them from the same table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import pandas as pd

from engine_common.shrink import eb_posterior, reliability_at
from nhl_engine.data.moneypuck import as_of
from nhl_engine.features.strength import PER_60, Metric

GOALIE_METRICS: tuple[Metric, ...] = (
    Metric(
        "gsax60", "all", (("xGoals", 1.0), ("goals", -1.0)), exposure_col="toi", label="GSAx/60"
    ),
    Metric(
        "svpct",
        "all",
        (("saves", 1.0),),
        exposure_col="ongoal",
        scale=1.0,
        label="Sv% (display only)",
    ),
)


def goalie_game_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per appearance from a MoneyPuck goalie log; all-situations only."""
    cols = [
        "playerId", "gameId", "gameDate", "season", "playerTeam", "opposingTeam", "home_or_away",
        "toi", "xga", "ga", "ongoal", "saves", "gsax60__num", "gsax60__exp", "svpct__num",
        "svpct__exp", "days_rest", "career_toi",
    ]  # fmt: skip
    if games.empty:
        return pd.DataFrame(columns=cols)
    g = games[games["situation"] == "all"].sort_values(["gameDate", "gameId"], kind="stable").copy()
    g = g.rename(columns={"icetime": "toi", "xGoals": "xga", "goals": "ga"})
    g["ongoal"] = g["ongoal"].fillna(0.0).astype(float)
    g["saves"] = g["ongoal"] - g["ga"].astype(float)
    g["gsax60__num"] = g["xga"].astype(float) - g["ga"].astype(float)
    g["gsax60__exp"] = g["toi"].astype(float)
    g["svpct__num"] = g["saves"]
    g["svpct__exp"] = g["ongoal"]
    prev = pd.to_datetime(g["gameDate"]).shift(1)
    g["days_rest"] = (pd.to_datetime(g["gameDate"]) - prev).dt.days.astype("float")
    g["career_toi"] = g["toi"].astype(float).cumsum().shift(1).fillna(0.0)
    return g[cols].reset_index(drop=True)


def gsax60(table: pd.DataFrame) -> tuple[float, float]:
    exp = float(table["gsax60__exp"].sum()) if not table.empty else 0.0
    if exp <= 0:
        return float("nan"), 0.0
    return float(table["gsax60__num"].sum()) / exp * PER_60, exp


def svpct(table: pd.DataFrame) -> float:
    shots = float(table["svpct__exp"].sum()) if not table.empty else 0.0
    return float(table["svpct__num"].sum()) / shots if shots > 0 else float("nan")


@dataclass(frozen=True)
class GoalieSkill:
    player_id: int
    gsax60: float
    career_prior: float
    season_observed: float
    season_toi: float
    career_toi: float
    reliability: float
    anchor: str
    svpct_season: float
    days_rest: float | None
    games_season: int

    @property
    def is_callup(self) -> bool:
        return self.anchor == "callup"


def goalie_skill(
    games: pd.DataFrame,
    slate: Date,
    *,
    season: int,
    league_gsax60: float,
    k_season: float,
    k_career: float,
    r_yy: float,
    min_minutes: float,
    callup_prior: float | None = None,
) -> GoalieSkill:
    """As-of GSAx/60 estimate for one goalie; see module docstring for the two EB steps."""
    table = goalie_game_table(games)
    hist = as_of(table, slate)
    cur = hist[hist["season"] == season]
    prev = hist[hist["season"] < season]
    pid = int(table["playerId"].iloc[0]) if not table.empty else 0
    career_toi = float(hist["toi"].sum()) if not hist.empty else 0.0

    anchor = "league"
    base = league_gsax60
    if career_toi < min_minutes * 60:
        if callup_prior is not None:
            anchor, base = "callup", callup_prior
        else:
            anchor = "league (callup prior not fitted)"

    if prev.empty:
        career_prior = base
    else:
        num = exp = 0.0
        for s, t in prev.groupby("season"):
            w = r_yy ** (season - int(s))
            num += w * float(t["gsax60__num"].sum())
            exp += w * float(t["gsax60__exp"].sum())
        career_prior = eb_posterior(base, num / exp * PER_60, exp, k_career) if exp > 0 else base

    obs, n = gsax60(cur)
    post = eb_posterior(career_prior, obs, n, k_season) if n > 0 else career_prior
    last_rest = None
    if not hist.empty:
        gap = (slate - hist["gameDate"].iloc[-1]).days
        last_rest = float(gap)
    return GoalieSkill(
        player_id=pid,
        gsax60=post,
        career_prior=career_prior,
        season_observed=obs,
        season_toi=n,
        career_toi=career_toi,
        reliability=reliability_at(n, k_season),
        anchor=anchor,
        svpct_season=svpct(cur),
        days_rest=last_rest,
        games_season=int(len(cur)),
    )


__all__ = ["GOALIE_METRICS", "GoalieSkill", "goalie_game_table", "goalie_skill", "gsax60", "svpct"]
