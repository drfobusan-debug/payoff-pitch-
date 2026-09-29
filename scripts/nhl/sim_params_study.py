"""Fit the game-simulation parameters (master plan §5.3) from the shot log + shift charts.

    python scripts/nhl/sim_params_study.py --from 2022 --to 2024 --out ~/.nhl_engine/studies/sim_params.json

Everything ``nhl_engine/models/periods.py`` needs beyond the two teams' §5.1
rates and the goalies' GSAx is measured here, never typed in:

* ``score_mult``  -- all-situation xG per second by score state from the
  shooting team's view (trail 2+, trail 1, tied, lead 1, lead 2+), relative to
  tied, regulation only. Exposure is exact: the score only changes at goals,
  so the goal timeline gives the seconds spent in each state. Shots taken with
  either net empty are excluded from the numerator (EN is its own block).
* ``period_mult`` -- tied-state xG per second in P1/P2/P3 relative to the mean.
* ``home_xg_share`` -- home share of tied 5v5 xG (home-ice edge).
* ``goals_per_xg`` -- league finishing on the shot log (should be ~1.0).
* ``pen``         -- minor penalties per team per 60 and PP seconds per penalty
  from the strength-state ice time (team logs).
* ``en``          -- goalie-pull behaviour from the shift charts: for each
  trailing team in the third period, the time remaining when the goalie's
  last regulation shift ended (by deficit), seconds spent pulled, xG/60 for
  the pulling team while 6v5, and goals against per pulled second.
* ``ot``          -- share of tied-after-60 games decided in OT, the OT goal
  rate per second, and a logistic link from the season 5v5 xG% gap to the OT
  winner (expected weak); ``so_home`` -- home share of shootout wins.

Output is a JSON the config quotes; ``config.SimParams`` carries the numbers
and this docstring's provenance. The dispersion / market-relationship fit
(§5.3, Dixon-Coles ρ) needs 60+ live boards and is deliberately not here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nhl_engine.config import cache_dir
from nhl_engine.data.moneypuck import MoneyPuckClient, full_game_id
from nhl_engine.data.nhlapi import NHLAPIClient

REG = 3600
PERIOD = 1200
EXTRA = (
    "homeTeamGoals",
    "awayTeamGoals",
    "homeEmptyNet",
    "awayEmptyNet",
    "shotOnEmptyNet",
    "homeTeamWon",
)
STATES = ("trail2", "trail1", "tied", "lead1", "lead2")


def _state(diff: int) -> str:
    if diff <= -2:
        return "trail2"
    if diff == -1:
        return "trail1"
    if diff == 0:
        return "tied"
    if diff == 1:
        return "lead1"
    return "lead2"


def _goal_timeline(g: pd.DataFrame) -> list[tuple[int, int]]:
    """``[(second, +1 home / -1 away), ...]`` regulation goals, sorted."""
    goals = g[(g["goal"] == 1) & (g["period"] <= 3)]
    return sorted(
        (int(t), 1 if h else -1)
        for t, h in zip(goals["time"], goals["isHomeTeam"].astype(float) > 0, strict=True)
    )


def _diff_at(timeline: list[tuple[int, int]], t: int) -> int:
    return sum(s for tt, s in timeline if tt < t)


def score_state_fit(shots: pd.DataFrame) -> dict[str, object]:
    """xG/sec by shooter's score state (all situations, regulation, nets in)."""
    num: dict[str, float] = defaultdict(float)
    goals: dict[str, float] = defaultdict(float)
    exp: dict[str, float] = defaultdict(float)
    p_num: dict[int, float] = defaultdict(float)
    p_exp: dict[int, float] = defaultdict(float)
    home_tied = away_tied = 0.0
    for _, g in shots.groupby("game_id"):
        tl = _goal_timeline(g)
        # exposure: walk the timeline; both teams accrue seconds in mirrored states
        cuts = [0] + [t for t, _ in tl if 0 < t < REG] + [REG]
        diff = 0
        for i, (a, b) in enumerate(zip(cuts, cuts[1:], strict=False)):
            if i > 0:
                diff += tl[i - 1][1]
            secs = float(b - a)
            exp[_state(diff)] += secs
            exp[_state(-diff)] += secs
            if diff == 0:
                for p in (1, 2, 3):
                    lo, hi = (p - 1) * PERIOD, p * PERIOD
                    p_exp[p] += max(0.0, float(min(b, hi) - max(a, lo)))
        reg = g[(g["period"] <= 3) & (g["homeEmptyNet"] == 0) & (g["awayEmptyNet"] == 0)]
        for t, is_home, xg, goal in zip(
            reg["time"].astype(int),
            reg["isHomeTeam"].astype(float) > 0,
            reg["xGoal"].astype(float),
            reg["goal"].astype(int),
            strict=True,
        ):
            d = _diff_at(tl, t)
            st = _state(d if is_home else -d)
            num[st] += xg
            goals[st] += goal
            if d == 0:
                p_num[(t // PERIOD) + 1 if t < REG else 3] += xg
                if is_home:
                    home_tied += xg
                else:
                    away_tied += xg
    tied_rate = num["tied"] / exp["tied"]
    mult = {s: (num[s] / exp[s]) / tied_rate for s in STATES}
    se = {s: mult[s] / math.sqrt(max(goals[s], 1.0)) for s in STATES}
    p_mean = sum(p_num.values()) / sum(p_exp.values())
    return {
        "score_mult": mult,
        "score_mult_se": se,
        "score_exposure_hours": {s: exp[s] / 3600 for s in STATES},
        "period_mult": {str(p): (p_num[p] / p_exp[p]) / p_mean for p in (1, 2, 3)},
        "home_xg_share": home_tied / (home_tied + away_tied),
        "goals_per_xg": float(sum(goals.values()) / sum(num.values())),
        "xg_per_team_60_tied": tied_rate * 3600,
    }


def ot_fit(shots: pd.DataFrame, xg_share: dict[tuple[int, str], float]) -> dict[str, object]:
    tied = 0
    ot_goal = 0
    ot_secs = 0.0
    so_home = so_n = 0
    xs: list[float] = []
    ys: list[int] = []
    for gid, g in shots.groupby("game_id"):
        tl = _goal_timeline(g)
        if sum(s for _, s in tl) != 0:
            continue
        tied += 1
        ot = g[(g["period"] == 4) & (g["goal"] == 1)]
        season = int(gid) // 1_000_000 if int(gid) > 1_000_000 else 0
        home, away = str(g["homeTeamCode"].iloc[0]), str(g["awayTeamCode"].iloc[0])
        won_home = int(g["homeTeamWon"].iloc[0]) == 1
        if not ot.empty:
            ot_goal += 1
            t = int(ot["time"].iloc[0]) - REG
            ot_secs += t
            gap = xg_share.get((season, home), 0.5) - xg_share.get((season, away), 0.5)
            xs.append(gap)
            ys.append(1 if won_home else 0)
        else:
            ot_secs += 300
            so_n += 1
            so_home += int(won_home)
    a, b, se_a, se_b = (
        _logit_fit(np.array(xs), np.array(ys)) if len(xs) > 50 else (0.0, 0.0, 0.0, 0.0)
    )
    return {
        "tied_after_60": tied,
        "p_ot_goal": ot_goal / max(tied, 1),
        "ot_goal_per_sec": ot_goal / max(ot_secs, 1.0),
        "ot_home_win_share": float(np.mean(ys)) if ys else float("nan"),
        "ot_home_intercept": a,
        "ot_home_intercept_se": se_a,
        "ot_strength_slope": b,
        "ot_strength_slope_se": se_b,
        "so_n": so_n,
        "so_home": so_home / max(so_n, 1),
    }


def _logit_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """``P(home wins OT) = sigmoid(a + b * x)`` by Newton steps; returns a, b and SEs."""
    X = np.column_stack([np.ones_like(x), x])
    w: np.ndarray = np.zeros(2)
    cov: np.ndarray = np.eye(2)
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None])
        cov = np.linalg.inv(hess)
        step = cov @ grad
        w = w + step
        if float(np.abs(step).max()) < 1e-8:
            break
    return float(w[0]), float(w[1]), float(math.sqrt(cov[0, 0])), float(math.sqrt(cov[1, 1]))


def penalty_fit(mp: MoneyPuckClient, seasons: list[int]) -> dict[str, float]:
    pens = secs_all = secs_pp = 0.0
    for season in seasons:
        teams = mp.season_teams(season)
        if teams.empty:
            continue
        allsit = teams[teams["situation"] == "all"]
        pp = teams[teams["situation"] == "5on4"]
        pens += float(allsit["penaltiesFor"].sum())
        secs_all += float(allsit["iceTime"].sum())
        secs_pp += float(pp["iceTime"].sum())
    return {
        "penalties_per_team_60": pens / secs_all * 3600 if secs_all else float("nan"),
        "pp_seconds_per_penalty": secs_pp / pens if pens else float("nan"),
    }


def empty_net_fit(
    shots: pd.DataFrame, api: NHLAPIClient, season: int, max_games: int
) -> dict[str, object]:
    pull_left: dict[int, list[float]] = defaultdict(list)
    trailing_at_5: dict[int, int] = defaultdict(int)
    pulled_secs = 0.0
    xg_for = goals_for = goals_against = 0.0
    n_games = 0
    for gid, g in shots.groupby("game_id"):
        if n_games >= max_games:
            break
        full = full_game_id(season, int(gid))
        roster = api.roster(full)
        shifts = api.shifts(full)
        if not roster or not shifts:
            continue
        n_games += 1
        home, away = str(g["homeTeamCode"].iloc[0]), str(g["awayTeamCode"].iloc[0])
        goalies = {r.player_id for r in roster if r.position == "G"}
        tl = _goal_timeline(g)
        for team in (home, away):
            d5 = -_diff_at(tl, REG - 300) * (1 if team == home else -1)
            if d5 > 0:
                trailing_at_5[min(d5, 3)] += 1
            p3 = [s for s in shifts if s.period == 3 and s.team == team and s.player_id in goalies]
            if not p3:
                continue
            covered = np.zeros(PERIOD, dtype=bool)
            for s in p3:
                covered[max(0, s.start) : min(PERIOD, s.end)] = True
            off = np.flatnonzero(~covered)
            if len(off) < 15:
                continue
            # a pull is a >=15 s stretch with the net empty while trailing (a
            # delayed-penalty gap or a swap while tied/leading is not one)
            runs = np.split(off, np.flatnonzero(np.diff(off) > 1) + 1)
            pulls = [
                r
                for r in runs
                if len(r) >= 15
                and (_diff_at(tl, 2 * PERIOD + int(r[0])) * (1 if team == home else -1)) < 0
            ]
            if not pulls:
                continue
            first = int(pulls[0][0])
            deficit = -_diff_at(tl, 2 * PERIOD + first) * (1 if team == home else -1)
            pull_left[min(deficit, 3)].append(float(PERIOD - first))
            pulled_secs += float(sum(len(r) for r in pulls))
            # numerators from MoneyPuck's own empty-net flags on the same games
            flag = "homeEmptyNet" if team == home else "awayEmptyNet"
            en = g[(g["period"] == 3) & (g[flag] == 1)]
            mine = en[(en["isHomeTeam"].astype(float) > 0) == (team == home)]
            xg_for += float(mine["xGoal"].sum())
            goals_for += float(mine["goal"].sum())
            theirs = en[(en["isHomeTeam"].astype(float) > 0) != (team == home)]
            goals_against += float(theirs["goal"].sum())
    return {
        "games": n_games,
        "pulls": {str(d): len(v) for d, v in pull_left.items()},
        "trailing_at_5min": {str(d): n for d, n in trailing_at_5.items()},
        "pull_prob": {str(d): len(pull_left[d]) / n for d, n in trailing_at_5.items() if n},
        "pull_time_left_quantiles": {
            str(d): [float(q) for q in np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])]
            for d, v in pull_left.items()
            if len(v) >= 20
        },
        "pulled_seconds_per_pull": pulled_secs / max(sum(len(v) for v in pull_left.values()), 1),
        "xg60_for_while_pulled": xg_for / max(pulled_secs, 1.0) * 3600,
        "goals60_for_while_pulled": goals_for / max(pulled_secs, 1.0) * 3600,
        "goals60_against_empty_net": goals_against / max(pulled_secs, 1.0) * 3600,
    }


def season_xg_share(mp: MoneyPuckClient, season: int) -> dict[tuple[int, str], float]:
    teams = mp.season_teams(season)
    out: dict[tuple[int, str], float] = {}
    if teams.empty:
        return out
    t5 = teams[teams["situation"] == "5on5"]
    for _, r in t5.iterrows():
        xf, xa = float(r["xGoalsFor"]), float(r["xGoalsAgainst"])
        out[(season, str(r["team"]))] = xf / (xf + xa) if xf + xa > 0 else 0.5
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2022)
    ap.add_argument("--to", dest="end", type=int, default=2024)
    ap.add_argument("--en-games", type=int, default=400, help="games per season for the EN fit")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    seasons = list(range(args.start, args.end + 1))
    mp = MoneyPuckClient(cache_dir=cache_dir())
    api = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")

    frames = []
    xg_share: dict[tuple[int, str], float] = {}
    en_by_season = {}
    for season in seasons:
        df = mp.shots(season, extra=EXTRA)
        if df.empty:
            print(f"{season}: no shot log", file=sys.stderr)
            continue
        df = df.copy()
        df["game_id"] = df["game_id"].astype(int) + season * 1_000_000
        frames.append(df)
        xg_share.update(season_xg_share(mp, season))
        en_by_season[str(season)] = empty_net_fit(
            df.assign(game_id=df["game_id"] - season * 1_000_000), api, season, args.en_games
        )
        print(f"{season}: {df['game_id'].nunique()} games", file=sys.stderr)
    shots = pd.concat(frames, ignore_index=True)
    report: dict[str, object] = {
        "seasons": seasons,
        "games": int(shots["game_id"].nunique()),
        **score_state_fit(shots),
        "pen": penalty_fit(mp, seasons),
        "en": en_by_season,
        "ot": ot_fit(shots, xg_share),
    }
    text = json.dumps(report, indent=2, default=float)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
