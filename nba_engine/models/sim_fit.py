"""Fit the game simulation on a no-look-ahead replay, then fit its blend with the close.

1. **Fresh-out production.** The rating's margin miss is regressed on
   ``lost(away) - lost(home)`` and its total miss on ``lost(away) + lost(home)``
   over the training seasons, with a slate-date bootstrap; a slope whose 95%
   interval includes zero is applied as 0.
2. **Shape.** With those slopes applied, the full-game SDs are the training
   residuals' SDs; the first half's margin share is the through-origin slope of
   the half margin on the modelled margin, its total share the ratio of sums,
   and the half SDs their residuals'.
3. **Halves.** ``pace_shift`` and ``eff_shift`` come from play-by-play
   possessions and points per half (``data.pbp``), each with an interval.
4. **Blend.** For each market, one point per game: the main line's home (or
   Over) side at the close, the model's probability, the close's de-vigged
   fair, and the settled result. ``w`` minimises the log loss of
   ``w * p_model + (1 - w) * p_fair`` on the training seasons; it is applied
   only when its bootstrap interval excludes zero. The held-out season is
   graded with the applied weight.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date as Date
from pathlib import Path

from nba_engine.audit.settle import LOSS, WIN, settle_row
from nba_engine.data import history
from nba_engine.data.pbp import HalfLine
from nba_engine.market.board import Selection, selections
from nba_engine.models.game_sim import SIM_MARKETS, GameDist, GameSim, SimParams, dist
from nba_engine.models.schedule import season_of
from nba_engine.schemas import GameResult

Pair = tuple[float, float]


def _interval(
    by_day: Mapping[str, Sequence[Pair]],
    stat: Callable[[Sequence[Pair]], float],
    draws: int,
    seed: int,
) -> tuple[float, float]:
    days = sorted(by_day)
    if not days or draws <= 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    boot = sorted(
        stat([pair for _ in days for pair in by_day[days[rng.randrange(len(days))]]])
        for _ in range(draws)
    )
    return boot[int(0.025 * draws)], boot[max(int(0.975 * draws) - 1, 0)]


def _ols(pairs: Sequence[Pair]) -> float:
    n = len(pairs)
    if n < 2:
        return 0.0
    mx = sum(x for x, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / sxx if sxx > 0 else 0.0


@dataclass(frozen=True)
class SlopeFit:
    name: str
    n: int
    value: float
    lo: float
    hi: float

    @property
    def applied(self) -> float:
        return self.value if self.lo > 0.0 or self.hi < 0.0 else 0.0


def _graded(
    sims: Iterable[GameSim], finals: Mapping[str, GameResult], seasons: set[int]
) -> list[tuple[GameSim, GameResult]]:
    out: list[tuple[GameSim, GameResult]] = []
    for s in sims:
        g = finals.get(s.espn_id)
        if g is not None and season_of(g.game_date) in seasons and len(g.home_q) >= 4:
            out.append((s, g))
    return out


def fit_lost(
    sims: Iterable[GameSim],
    finals: Mapping[str, GameResult],
    seasons: set[int],
    draws: int = 1000,
    seed: int = 13,
) -> tuple[SlopeFit, SlopeFit]:
    margin: dict[str, list[Pair]] = defaultdict(list)
    total: dict[str, list[Pair]] = defaultdict(list)
    for s, g in _graded(sims, finals, seasons):
        margin[s.game_date].append(
            (s.lost_away - s.lost_home, g.final_home - g.final_away - s.rating_margin)
        )
        total[s.game_date].append(
            (s.lost_away + s.lost_home, g.final_home + g.final_away - s.rating_total)
        )
    fits: list[SlopeFit] = []
    for name, by_day in (("lost_margin", margin), ("lost_total", total)):
        pairs = [p for v in by_day.values() for p in v]
        lo, hi = _interval(by_day, _ols, draws, seed)
        fits.append(SlopeFit(name, len(pairs), _ols(pairs), lo, hi))
    return fits[0], fits[1]


def fit_shape(
    sims: Iterable[GameSim],
    finals: Mapping[str, GameResult],
    seasons: set[int],
    params: SimParams,
) -> SimParams:
    """SDs and first-half shares from the training residuals under ``params``' means."""
    rows = [(dist(s, params), g) for s, g in _graded(sims, finals, seasons)]
    if len(rows) < 2:
        return params
    m = [(d.margin, float(g.final_home - g.final_away)) for d, g in rows]
    t = [(d.total, float(g.final_home + g.final_away)) for d, g in rows]
    h1m = [(d.margin, float(g.h1_home - g.h1_away)) for d, g in rows]
    h1t = [(d.total, float(g.h1_home + g.h1_away)) for d, g in rows]
    share_m = sum(x * y for x, y in h1m) / sum(x * x for x, _ in h1m)
    share_t = sum(y for _, y in h1t) / sum(x for x, _ in h1t)
    return replace(
        params,
        sd_margin=statistics.pstdev(y - x for x, y in m),
        sd_total=statistics.pstdev(y - x for x, y in t),
        h1_margin_share=share_m,
        h1_total_share=share_t,
        sd_h1_margin=statistics.pstdev(y - share_m * x for x, y in h1m),
        sd_h1_total=statistics.pstdev(y - share_t * x for x, y in h1t),
    )


@dataclass(frozen=True)
class HalfFit:
    n: int  # team-games
    h1_poss: float  # per team
    h2_poss: float
    h1_ppp: float  # points per 100 possessions
    h2_ppp: float
    pace_shift: float
    pace_lo: float
    pace_hi: float
    eff_shift: float
    eff_lo: float
    eff_hi: float


def fit_halves(
    lines: Mapping[str, Sequence[HalfLine]],
    finals: Mapping[str, GameResult],
    seasons: set[int],
    draws: int = 1000,
    seed: int = 17,
) -> HalfFit:
    """Second half against first: possessions (pace) and points per possession (efficiency)."""
    by_day: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for eid, halves in lines.items():
        g = finals.get(eid)
        if g is None or season_of(g.game_date) not in seasons:
            continue
        for side in ("home", "away"):
            h = {x.half: x for x in halves if x.side == side}
            if 1 in h and 2 in h and h[1].possessions > 0 and h[2].possessions > 0:
                by_day[g.game_date.isoformat()].append(
                    (h[1].possessions, float(h[1].points), h[2].possessions, float(h[2].points))
                )
    rows = [r for v in by_day.values() for r in v]
    if not rows:
        return HalfFit(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def sums(rs: Sequence[tuple[float, float, float, float]]) -> tuple[float, ...]:
        return tuple(sum(r[i] for r in rs) for i in range(4))

    def pace(rs: Sequence[tuple[float, float, float, float]]) -> float:
        p1, _, p2, _ = sums(rs)
        return p2 / p1 - 1.0

    def eff(rs: Sequence[tuple[float, float, float, float]]) -> float:
        p1, x1, p2, x2 = sums(rs)
        return (x2 / p2) / (x1 / p1) - 1.0

    days = sorted(by_day)
    rng = random.Random(seed)
    boots = [
        [r for _ in days for r in by_day[days[rng.randrange(len(days))]]] for _ in range(draws)
    ]
    pace_b = sorted(pace(b) for b in boots)
    eff_b = sorted(eff(b) for b in boots)
    lo_i, hi_i = int(0.025 * draws), max(int(0.975 * draws) - 1, 0)
    p1, x1, p2, x2 = sums(rows)
    n = len(rows)
    return HalfFit(
        n=n,
        h1_poss=p1 / n,
        h2_poss=p2 / n,
        h1_ppp=100.0 * x1 / p1,
        h2_ppp=100.0 * x2 / p2,
        pace_shift=pace(rows),
        pace_lo=pace_b[lo_i] if draws else 0.0,
        pace_hi=pace_b[hi_i] if draws else 0.0,
        eff_shift=eff(rows),
        eff_lo=eff_b[lo_i] if draws else 0.0,
        eff_hi=eff_b[hi_i] if draws else 0.0,
    )


@dataclass(frozen=True)
class BlendPoint:
    market: str
    season: int
    game_date: str
    p_model: float
    fair: float
    won: bool


def _main_lines(sels: Iterable[Selection]) -> list[Selection]:
    """Per game and market, the home (or Over) side whose fair is nearest even."""
    best: dict[tuple[str, str], Selection] = {}
    for s in sels:
        if s.market not in SIM_MARKETS or s.fair is None:
            continue
        home = s.matchup.split(" @ ")[-1]
        if s.side != ("over" if s.market.endswith("_total") else home):
            continue
        k = (s.matchup, s.market)
        held = best.get(k)
        if held is None or abs(s.fair - 0.5) < abs((held.fair or 0.5) - 0.5):
            best[k] = s
    return list(best.values())


def points_for_day(
    sels: Iterable[Selection],
    dists: Mapping[tuple[str, str], GameDist],
    finals: Mapping[tuple[str, str], GameResult],
) -> list[BlendPoint]:
    out: list[BlendPoint] = []
    for s in _main_lines(sels):
        d, g = dists.get((s.game_date, s.matchup)), finals.get((s.game_date, s.matchup))
        if d is None or g is None or s.fair is None:
            continue
        p = d.prob(s.market, s.side, s.line)
        outcome, _ = settle_row(s.market, s.side, s.entity, s.line, g)
        if p is None or outcome not in (WIN, LOSS):
            continue
        out.append(
            BlendPoint(s.market, season_of(g.game_date), s.game_date, p, s.fair, outcome == WIN)
        )
    return out


def close_points(
    data_dir: Path,
    days: Iterable[Date],
    dists: Mapping[tuple[str, str], GameDist],
    finals: Mapping[tuple[str, str], GameResult],
) -> list[BlendPoint]:
    """One point per game and market from the archived closes."""
    out: list[BlendPoint] = []
    for day in days:
        rows = [r for r in history.history_rows(data_dir, day) if r.market in SIM_MARKETS]
        if rows:
            out.extend(points_for_day(selections(rows), dists, finals))
    return out


def log_loss(points: Sequence[BlendPoint], weight: float) -> float:
    if not points:
        return 0.0
    total = 0.0
    for x in points:
        q = min(max(weight * x.p_model + (1.0 - weight) * x.fair, 1e-9), 1.0 - 1e-9)
        total -= math.log(q if x.won else 1.0 - q)
    return total / len(points)


def best_weight(points: Sequence[BlendPoint], steps: int = 25) -> float:
    """Log loss is convex in the weight, so a ternary search on [0, 1] finds it.

    A minimum no better than the close alone reads as exactly 0.
    """
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        a, b = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if log_loss(points, a) <= log_loss(points, b):
            hi = b
        else:
            lo = a
    w = (lo + hi) / 2.0
    return 0.0 if log_loss(points, 0.0) <= log_loss(points, w) else w


@dataclass(frozen=True)
class BlendFit:
    market: str
    n: int
    weight: float
    lo: float
    hi: float
    holdout_n: int
    ll_fair: float  # holdout log loss of the close alone
    ll_blend: float  # with the applied weight
    ll_model: float  # the model alone

    @property
    def applied(self) -> float:
        return self.weight if self.lo > 0.0 else 0.0


def fit_blend(
    points: Sequence[BlendPoint],
    train: set[int],
    holdout: set[int],
    draws: int = 200,
    seed: int = 19,
) -> list[BlendFit]:
    out: list[BlendFit] = []
    for market in SIM_MARKETS:
        tr = [x for x in points if x.market == market and x.season in train]
        ho = [x for x in points if x.market == market and x.season in holdout]
        if not tr:
            continue
        by_day: dict[str, list[BlendPoint]] = defaultdict(list)
        for x in tr:
            by_day[x.game_date].append(x)
        days = sorted(by_day)
        rng = random.Random(seed)
        boot = sorted(
            best_weight([x for _ in days for x in by_day[days[rng.randrange(len(days))]]])
            for _ in range(draws)
        )
        w = best_weight(tr)
        lo = boot[int(0.025 * draws)] if draws else 0.0
        hi = boot[max(int(0.975 * draws) - 1, 0)] if draws else 1.0
        applied = w if lo > 0.0 else 0.0
        out.append(
            BlendFit(
                market=market,
                n=len(tr),
                weight=w,
                lo=lo,
                hi=hi,
                holdout_n=len(ho),
                ll_fair=log_loss(ho, 0.0),
                ll_blend=log_loss(ho, applied),
                ll_model=log_loss(ho, 1.0),
            )
        )
    return out
