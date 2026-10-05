"""Fit the rating book's update speeds on a no-look-ahead replay, then grade it against the close.

The fit minimises the squared error of both teams' scores (which covers the
margin and the total at once) over the training seasons, by coordinate
descent over a small grid per parameter. The first season in the archive is
burn-in and is never graded. A held-out season, which the fit never sees, is then
graded against the closing spread and total:

* MAE of the model and of the close on the same games;
* the slope of ``actual - close`` on ``model - close``, with a bootstrap
  interval clustered by slate date. A slope whose interval excludes zero
  means that the model's disagreement with the close carries information the close
  lacks. One whose interval includes zero means that the rating adds nothing to the close,
  and it is then only a prior for the simulation, not a bet signal.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date as Date
from pathlib import Path

from nba_engine.data import history
from nba_engine.models.ratings import Prediction, RatingParams, replay
from nba_engine.models.schedule import TeamSchedule, season_of
from nba_engine.schemas import GameResult

GRID: dict[str, tuple[float, ...]] = {
    "drift_sd": (0.15, 0.25, 0.35, 0.5, 0.7, 1.0),
    "prior_sd": (2.0, 3.0, 4.0, 5.0),
    "carry": (0.4, 0.55, 0.7, 0.85, 1.0),
    "season_sd": (0.5, 1.0, 2.0, 3.0, 4.0),
    "tired_weight": (0.4, 0.6, 0.8, 1.0),
    "pace_drift_sd": (0.1, 0.2, 0.3, 0.5, 0.8),
    "pace_carry": (0.4, 0.55, 0.7, 0.85, 1.0),
    "pace_season_sd": (0.5, 1.0, 2.0, 3.0),
    "league_gain": (0.003, 0.006, 0.01, 0.02),
    "hca_gain": (0.002, 0.005, 0.01, 0.02),
}
ROUNDS = 2

Lines = dict[tuple[str, str], tuple[float | None, float | None]]


def score_sse(preds: Sequence[Prediction], finals: dict[str, GameResult], seasons: set[int]) -> float:
    """Mean squared error per team score over the graded seasons."""
    err, n = 0.0, 0
    for p in preds:
        g = finals[p.espn_id]
        if season_of(g.game_date) not in seasons:
            continue
        hp = p.home_ppp * p.poss / 100.0
        ap = p.away_ppp * p.poss / 100.0
        err += (g.final_home - hp) ** 2 + (g.final_away - ap) ** 2
        n += 2
    return err / n if n else float("inf")


def fit(
    games: Sequence[GameResult],
    train: set[int],
    sched: dict[tuple[str, str], TeamSchedule],
    start: RatingParams | None = None,
    grid: dict[str, tuple[float, ...]] | None = None,
    rounds: int = ROUNDS,
) -> tuple[RatingParams, float]:
    """Coordinate descent over ``grid`` on the training seasons' score error."""
    finals = {g.espn_id: g for g in games}
    best = start or RatingParams()
    best_err = score_sse(replay(games, best, sched)[0], finals, train)
    for _ in range(rounds):
        for name, values in (grid or GRID).items():
            for v in values:
                trial = replace(best, **{name: v})
                err = score_sse(replay(games, trial, sched)[0], finals, train)
                if err < best_err - 1e-9:
                    best, best_err = trial, err
    return best, best_err


def closing_lines(data_dir: Path, days: Iterable[Date]) -> Lines:
    """``(slate date, matchup)`` -> (median closing home spread, median closing total)."""
    out: Lines = {}
    for day in days:
        spreads: dict[str, list[float]] = defaultdict(list)
        totals: dict[str, list[float]] = defaultdict(list)
        for r in history.history_rows(data_dir, day):
            if r.line is None:
                continue
            if r.market == "game_ats" and r.side == r.matchup.split(" @ ")[-1]:
                spreads[r.matchup].append(r.line)
            elif r.market == "game_total" and r.side == "over":
                totals[r.matchup].append(r.line)
        for m in set(spreads) | set(totals):
            out[(day.isoformat(), m)] = (
                statistics.median(spreads[m]) if spreads[m] else None,
                statistics.median(totals[m]) if totals[m] else None,
            )
    return out


@dataclass(frozen=True)
class CloseGrade:
    market: str  # margin | total
    n: int
    model_mae: float
    close_mae: float
    slope: float
    lo: float
    hi: float

    @property
    def adds_information(self) -> bool:
        return self.lo > 0.0


def _slope(pairs: Sequence[tuple[float, float]]) -> float:
    sxx = sum(x * x for x, _ in pairs)
    return sum(x * y for x, y in pairs) / sxx if sxx > 0 else 0.0


def vs_close(
    preds: Sequence[Prediction],
    finals: dict[str, GameResult],
    lines: Lines,
    seasons: set[int],
    draws: int = 2000,
    seed: int = 7,
) -> list[CloseGrade]:
    """Model against the close on the games both priced, in the graded seasons."""
    rows: dict[str, list[tuple[str, float, float, float]]] = {"margin": [], "total": []}
    for p in preds:
        g = finals[p.espn_id]
        if season_of(g.game_date) not in seasons:
            continue
        spread, total = lines.get((p.game_date, f"{p.away} @ {p.home}"), (None, None))
        if spread is not None:
            rows["margin"].append((p.game_date, p.margin, -spread, g.final_home - g.final_away))
        if total is not None:
            rows["total"].append((p.game_date, p.total, total, g.final_home + g.final_away))
    out: list[CloseGrade] = []
    rng = random.Random(seed)
    for market, rs in rows.items():
        if not rs:
            continue
        by_day: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for d, model, close, actual in rs:
            by_day[d].append((model - close, actual - close))
        days = list(by_day)
        boot = []
        for _ in range(draws):
            pick = [pair for _ in days for pair in by_day[days[rng.randrange(len(days))]]]
            boot.append(_slope(pick))
        boot.sort()
        out.append(
            CloseGrade(
                market=market,
                n=len(rs),
                model_mae=statistics.mean(abs(a - m) for _, m, _, a in rs),
                close_mae=statistics.mean(abs(a - c) for _, _, c, a in rs),
                slope=_slope([pair for v in by_day.values() for pair in v]),
                lo=boot[int(0.025 * draws)],
                hi=boot[int(0.975 * draws) - 1],
            )
        )
    return out


__all__ = ["GRID", "CloseGrade", "closing_lines", "fit", "score_sse", "vs_close"]
