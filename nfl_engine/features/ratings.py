"""Opponent-adjusted team ratings, shrunk toward the mean.

For each metric the panel carries, every team-game is one observation of

    metric = league mean + offence(i) - defence(j) + home * home_edge

solved as a single ridge least squares over all games at once, so a team's
rating is net of who it played rather than a raw average. Defence is signed so a
*positive* number allows more, which makes a good defence a negative number in
every metric at once.

Two choices that are the whole rating layer:

**The ridge penalty is the regression, not a tuning knob.** A one-game-old NFL
rating is almost entirely noise, and a rating that is not shrunk hard prices a
14-point win over a bad team as a 14-point team. The penalty is what pulls every
team toward the league mean by how little evidence it has -- an unbeaten team
with two games played comes out barely above average, which is correct.

**History is discounted exponentially, not truncated at a season boundary.** A
week-1 rating that ignores last season is a rating of nothing, and one that
weights last September equally is a rating of a roster that no longer exists.
Both constants are fitted together by ``scripts/nfl/ratings_study.py --grid`` on
out-of-sample margin error over 2013-2025 (n=3,450 games). At the shipped ridge,
a half-life of 8 weeks gives MAE 10.299 against 10.337 at 6 weeks, 10.315 at 12
and 10.474 at 4 -- the fit wants about half a season of memory, and it is a
shallow optimum in both directions.

The weights are normalised to average 1 before the penalty is applied, which is
what makes the two parameters separately meaningful: unnormalised, a shorter
memory shrinks the total weight and the same nominal ridge shrinks every rating
several times harder, so the grid reads as an interaction that is really an
artefact of the scaling.

September, and the discount that is not there
---------------------------------------------
The decay clock counts *weeks present in the panel*, so the offseason is one
tick: a week-1 rating weights the previous January at ~0.92 and is, honestly
described, "how you finished last season" applied to a roster that has since
turned over. The obvious fix is to charge the boundary extra age, and
``scripts/nfl/september_study.py`` measures it as **wrong**, monotonically, in
the exact window it was meant to help -- weeks 1-4 margin MAE against extra
half-lives of offseason:

    extra weeks   0       2       4       6       8      12      20
    weeks 1-4   10.306  10.322  10.339  10.359  10.392  10.475  10.698

Forgetting last season faster is not the answer, because prior-season evidence is
carrying real signal in September; the same script's memory grid says early weeks
want *more* memory, not less (weeks 1-4 MAE 10.260 at a 12-week half-life against
10.306 at 8), which the shipped 8 weeks keeps because it is better over the full
season (10.282 against 10.314) and the September gain is 0.05 points.

What does help in September is *adding* offseason information rather than
discarding evidence, and the one piece of it the panel already carries is who is
taking the snaps -- see ``quarterback.py``, which also finds that a genuine
offseason quarterback change carries no rating bias at all.

**What this rating is worth is measured in scripts/nfl/ratings_study.py, and it
is not worth a bet on its own**: 10.282 margin MAE against the closing line's
9.905, and its disagreement with the line explains none of that line's error
(t = +0.25 over 3,450 games). It is here because the score distribution needs a
mean to shape, because it is the mean when no market has posted, and because the
props layer needs a game script -- not because it beats the market.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Fitted by scripts/nfl/ratings_study.py --fit over 2013-2025 walk-forward.
HALF_LIFE_WEEKS = 8.0
RIDGE = 600.0
# A rating needs some history behind it before it means anything; below this the
# book reports league means and the caller falls back to the market.
MIN_HISTORY_GAMES = 400
# Extra weeks of age charged per season boundary crossed. Ships at zero because
# it measured worse at every value tried -- see the module docstring. It is a
# parameter rather than a deleted experiment so ``scripts/nfl/september_study.py``
# can keep re-running the test that says not to use it.
OFFSEASON_WEEKS = 0.0

METRICS = ("epa", "success", "proe", "sec_per_play", "drives")


@dataclass(frozen=True)
class TeamRating:
    """One team's opponent-adjusted rating, as deltas from the league mean.

    ``off_*`` is what the team does with the ball, ``def_*`` what it allows, so
    a good defence is negative on ``def_epa`` and ``def_success`` alike.
    """

    team: str
    off_epa: float = 0.0
    def_epa: float = 0.0
    off_success: float = 0.0
    def_success: float = 0.0
    off_proe: float = 0.0
    off_sec_per_play: float = 0.0
    off_drives: float = 0.0

    def net_success(self) -> float:
        """The single number the walk-forward found most predictive of margin."""
        return self.off_success - self.def_success


@dataclass(frozen=True)
class RatingBook:
    """Every team's rating as of a point in time, plus the league's own means."""

    teams: dict[str, TeamRating] = field(default_factory=dict)
    league: dict[str, float] = field(default_factory=dict)
    home_edge: dict[str, float] = field(default_factory=dict)
    games_used: int = 0

    def rating(self, team: str) -> TeamRating:
        """A team we have never seen rates as exactly average, not as an error."""
        return self.teams.get(team, TeamRating(team=team))

    def is_usable(self) -> bool:
        return self.games_used >= MIN_HISTORY_GAMES


def _solve(
    frame: pd.DataFrame,
    metric: str,
    teams: list[str],
    weights: np.ndarray,
    *,
    ridge: float,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Weighted ridge fit of ``metric``; returns offence, defence, home, mean.

    The intercept and the home term are left unpenalised: shrinking them toward
    zero would shrink the league average itself, which is not an estimate we are
    short of data on.
    """
    have = frame[metric].notna().to_numpy()
    sub = frame[have]
    if sub.empty:
        return {}, {}, 0.0, 0.0
    w = weights[have]
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    rows = len(sub)
    design = np.zeros((rows, 2 * n_teams + 2))
    design[:, 0] = 1.0
    off = sub.posteam.map(index).to_numpy()
    dfn = sub.defteam.map(index).to_numpy()
    design[np.arange(rows), 1 + off] = 1.0
    design[np.arange(rows), 1 + n_teams + dfn] = 1.0
    design[:, -1] = sub.is_home.to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    root = np.sqrt(w)
    xw = design * root[:, None]
    yw = y * root
    penalty = np.full(design.shape[1], ridge)
    penalty[0] = 0.0
    penalty[-1] = 0.0
    normal = xw.T @ xw + np.diag(penalty)
    try:
        beta = np.linalg.solve(normal, xw.T @ yw)
    except np.linalg.LinAlgError:
        # Degenerate early-season data (one team, or every game at home) leaves
        # the home term collinear with the intercept; neither is penalised, so
        # the normal equations can be singular. Least squares picks one.
        beta = np.linalg.lstsq(normal, xw.T @ yw, rcond=None)[0]
    offence = {team: float(beta[1 + i]) for team, i in index.items()}
    defence = {team: float(beta[1 + n_teams + i]) for team, i in index.items()}
    return offence, defence, float(beta[-1]), float(beta[0])


def _weights(
    frame: pd.DataFrame,
    asof: float,
    half_life: float,
    *,
    asof_season: float | None = None,
    offseason: float = OFFSEASON_WEEKS,
) -> np.ndarray:
    """Exponential decay in weeks of age, so a rating forgets at a fitted rate.

    Normalised to average 1, which makes the ridge penalty mean the same thing at
    every half-life: without it, a short memory shrinks the total weight and the
    same nominal penalty shrinks the ratings several times harder, so the two
    parameters cannot be fitted independently.

    ``offseason`` charges each season boundary that extra many weeks of age, which
    is the discount September looks like it should want and measured worse at
    every value; it is zero unless a study asks for it.
    """
    age = np.maximum(asof - frame.week_index.to_numpy(dtype=float), 0.0)
    if offseason and asof_season is not None:
        gap = np.maximum(float(asof_season) - frame.season.to_numpy(dtype=float), 0.0)
        age = age + offseason * gap
    weights = np.asarray(0.5 ** (age / half_life), dtype=float)
    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def week_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Number every (season, week) consecutively: the clock the decay runs on.

    Weeks, not games, because a team's rating should age over a bye the same way
    it ages over a week it played.
    """
    order = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    order = order.reset_index(drop=True)
    order["week_index"] = np.arange(len(order), dtype=float)
    return frame.merge(order, on=["season", "week"], how="left")


def fit(
    history: pd.DataFrame,
    *,
    asof: float | None = None,
    asof_season: int | None = None,
    half_life: float = HALF_LIFE_WEEKS,
    ridge: float = RIDGE,
    offseason: float = OFFSEASON_WEEKS,
) -> RatingBook:
    """Rate every team in ``history``, which must contain only prior games.

    ``asof`` is the week index the ratings are for; ratings age relative to it.
    ``asof_season`` and ``offseason`` together charge extra age across a season
    boundary, and are only used by the September study: the shipped discount is
    zero.
    """
    if history.empty:
        return RatingBook()
    frame = history if "week_index" in history.columns else week_index(history)
    frame = frame.reset_index(drop=True)
    point = float(frame.week_index.max()) + 1.0 if asof is None else float(asof)
    weights = _weights(frame, point, half_life, asof_season=asof_season, offseason=offseason)
    teams = sorted(set(frame.posteam.dropna()) | set(frame.defteam.dropna()))
    offence: dict[str, dict[str, float]] = {}
    defence: dict[str, dict[str, float]] = {}
    league: dict[str, float] = {}
    home: dict[str, float] = {}
    for metric in METRICS:
        if metric not in frame.columns:
            continue
        off, dfn, home_edge, mean = _solve(frame, metric, teams, weights, ridge=ridge)
        offence[metric] = off
        defence[metric] = dfn
        league[metric] = mean
        home[metric] = home_edge
    ratings = {
        team: TeamRating(
            team=team,
            off_epa=offence.get("epa", {}).get(team, 0.0),
            def_epa=defence.get("epa", {}).get(team, 0.0),
            off_success=offence.get("success", {}).get(team, 0.0),
            def_success=defence.get("success", {}).get(team, 0.0),
            off_proe=offence.get("proe", {}).get(team, 0.0),
            off_sec_per_play=offence.get("sec_per_play", {}).get(team, 0.0),
            off_drives=offence.get("drives", {}).get(team, 0.0),
        )
        for team in teams
    }
    return RatingBook(
        teams=ratings,
        league=league,
        home_edge=home,
        games_used=int(frame.game_id.nunique()),
    )
