"""Fit the minutes book on a no-look-ahead replay, then grade it against the recent average.

The fit minimises mean absolute error of projected vs box minutes (regulation-
scaled) over the training seasons, by coordinate descent over a small grid per
parameter. A held-out season the fit never sees is graded against the naive
projection a bettor reads off a box score, his last ``RECENT`` games' average,
on three groups:

* ``all``: every available player;
* ``rotation``: baseline >= 15 minutes, the players who carry props;
* ``fresh_out``: rotation players on a team with a regular freshly out, the
  nights next-man-up decides.

The gain carries a bootstrap interval clustered by slate date. ``spread`` is
the miss's SD and mean by projected-minutes band, which the simulation draws
minutes from.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date as Date

from nba_engine.models.minutes import RECENT, MinutesParams, Projection, replay
from nba_engine.models.schedule import season_of
from nba_engine.schemas import GameResult

GRID: dict[str, tuple[float, ...]] = {
    "half_life": (2.0, 3.0, 4.0, 5.0, 7.0, 10.0),
    "carry": (0.0, 0.1, 0.2, 0.5, 1.0),
    "prior_minutes": (3.0, 6.0, 10.0),
    "prior_weight": (0.5, 1.0, 2.0),
    "deficit_power": (0.0, 0.25, 0.5, 1.0),
    "surplus_power": (0.0, 0.25, 0.5, 1.0),
    "cap": (34.0, 36.0, 37.0, 38.0, 40.0),
    "pair_k": (5.0, 10.0, 20.0, 40.0, 1e9),
    "fresh_games": (1.0, 2.0, 4.0, 6.0),
    "star_minutes": (10.0, 15.0, 20.0, 25.0),
}
ROUNDS = 2
ROTATION = 15.0
BANDS: tuple[tuple[float, float], ...] = ((0, 10), (10, 20), (20, 28), (28, 34), (34, 48))

Group = Callable[[Projection], bool]
GROUPS: dict[str, Group] = {
    "all": lambda r: True,
    "rotation": lambda r: r.baseline >= ROTATION,
    "fresh_out": lambda r: r.baseline >= ROTATION and r.fresh_out,
}


def graded(rows: Sequence[Projection], seasons: set[int]) -> list[Projection]:
    return [r for r in rows if r.actual is not None and season_of(r.game_date) in seasons]


def mae(rows: Sequence[Projection], seasons: set[int]) -> float:
    sel = graded(rows, seasons)
    return sum(abs(r.minutes - (r.actual or 0.0)) for r in sel) / max(len(sel), 1)


def fit(games: Sequence[GameResult], train: set[int]) -> tuple[MinutesParams, float]:
    best = MinutesParams()
    best_err = mae(replay(games, best)[0], train)
    for _ in range(ROUNDS):
        for name, values in GRID.items():
            for v in values:
                trial = replace(best, **{name: float(v)})
                if trial == best:
                    continue
                err = mae(replay(games, trial)[0], train)
                if err < best_err - 1e-9:
                    best, best_err = trial, err
    return best, best_err


@dataclass(frozen=True)
class Grade:
    group: str
    n: int
    model_mae: float
    recent_mae: float
    gain: float  # recent - model; positive means the model misses less
    lo: float
    hi: float


def _naive(r: Projection) -> float:
    return r.recent if r.recent is not None else r.baseline


def grade(
    rows: Sequence[Projection], seasons: set[int], draws: int = 1000, seed: int = 7
) -> list[Grade]:
    """Model vs the last-``RECENT``-games average, with a slate-date bootstrap on the gain."""
    rng = random.Random(seed)
    out: list[Grade] = []
    for name, keep in GROUPS.items():
        sel = [r for r in graded(rows, seasons) if keep(r)]
        if not sel:
            continue
        by_day: dict[Date, list[tuple[float, float]]] = defaultdict(list)
        for r in sel:
            a = r.actual or 0.0
            by_day[r.game_date].append((abs(r.minutes - a), abs(_naive(r) - a)))
        days = list(by_day.values())
        model = sum(m for d in days for m, _ in d) / len(sel)
        naive = sum(n for d in days for _, n in d) / len(sel)
        gains: list[float] = []
        for _ in range(draws):
            pick = [days[rng.randrange(len(days))] for _ in days]
            n = sum(len(d) for d in pick)
            gains.append(sum(nv - m for d in pick for m, nv in d) / n)
        gains.sort()
        lo = gains[int(0.025 * draws)] if draws else naive - model
        hi = gains[min(int(0.975 * draws), draws - 1)] if draws else naive - model
        out.append(Grade(name, len(sel), model, naive, naive - model, lo, hi))
    return out


def spread(rows: Sequence[Projection], seasons: set[int]) -> list[dict[str, float]]:
    """Miss (actual - projected) by projected-minutes band: n, mean, SD."""
    out: list[dict[str, float]] = []
    sel = graded(rows, seasons)
    for lo, hi in BANDS:
        miss = [(r.actual or 0.0) - r.minutes for r in sel if lo <= r.minutes < hi]
        if len(miss) < 2:
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": float(len(miss)),
                "mean": statistics.fmean(miss),
                "sd": statistics.stdev(miss),
            }
        )
    return out


__all__ = ["GRID", "GROUPS", "RECENT", "Grade", "fit", "grade", "mae", "spread"]
