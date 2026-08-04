"""A/B the score engines on out-of-sample prediction quality.

A realized PPV backtest needs historical *market lines*, which the free Odds API
tier does not expose. What CFBD *does* give us for past seasons is every final
score, so this backtest isolates the question the two engines actually differ
on -- the shape and location of the score distribution -- with line-free,
leak-free metrics:

* **Moneyline Brier score** -- ``mean((home_win_prob - actual_home_win)^2)``.
  Only final scores are needed, so this is a clean calibration comparison.
* **Margin RMSE** and **Total RMSE** -- projection accuracy of each engine's
  mean margin / total against the realized game.

Games are priced from the *same* ratings-implied means for both engines, so any
delta is attributable to the distribution model (normal vs Markov), not a
different forecast. Ratings come from the season being tested, matching how the
engine would have priced that slate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cfb_engine.config import Config
from cfb_engine.data.cfbd import CFBDClient, GameResult, RatingBook
from cfb_engine.models.markov import DriveShape, MarkovSim
from cfb_engine.models.montecarlo import ExpectedGame, GameSimResult, MonteCarlo


@dataclass
class EngineScore:
    engine: str
    n: int
    brier: float
    logloss: float
    margin_rmse: float
    total_rmse: float


def _rating_margin_total(
    ratings: RatingBook, home: str, away: str, hfa: float
) -> tuple[float, float] | None:
    h = ratings.get(home)
    a = ratings.get(away)
    if h is None or a is None:
        return None
    la = ratings.league_avg or 27.5
    home_pts = h.offense * a.defense / la
    away_pts = a.offense * h.defense / la
    return (home_pts - away_pts) + hfa, home_pts + away_pts


def _sim(
    engine: str,
    exp: ExpectedGame,
    mc: MonteCarlo,
    markov: MarkovSim,
    shape: DriveShape | None,
) -> GameSimResult:
    if engine == "markov" and shape is not None:
        return markov.simulate(exp, shape)
    return mc.simulate(exp)


def run_backtest(cfg: Config, season: int, *, engines: tuple[str, ...] = ("normal", "markov")) -> list[EngineScore]:
    """Score each engine on the given season's completed games."""
    cfbd = CFBDClient(cfg.creds.cfbd_api_key)
    ratings = cfbd.fetch_ratings(season)
    if ratings is None:
        return []
    advanced = cfbd.fetch_advanced(season)
    results: list[GameResult] = cfbd.fetch_all_results(season)

    mc = MonteCarlo(cfg.model)
    markov = MarkovSim(cfg.model)
    hfa = cfg.model.home_field_pts

    acc = {
        e: {"n": 0, "brier": 0.0, "logloss": 0.0, "margin_se": 0.0, "total_se": 0.0}
        for e in engines
    }
    for res in results:
        rt = _rating_margin_total(ratings, res.home, res.away, hfa)
        if rt is None:
            continue
        exp_margin, exp_total = rt
        exp = ExpectedGame(exp_margin, exp_total, cfg.model.margin_sd, cfg.model.total_sd)
        ha = advanced.get(res.home)
        aa = advanced.get(res.away)
        shape = (
            DriveShape(ha.drives_per_game, aa.drives_per_game)
            if ha is not None and aa is not None
            else None
        )
        actual_margin = res.home_points - res.away_points
        actual_total = res.home_points + res.away_points
        home_won = 1.0 if actual_margin > 0 else 0.0
        for engine in engines:
            sim = _sim(engine, exp, mc, markov, shape)
            p = min(max(sim.home_win_prob(), 1e-6), 1 - 1e-6)
            a = acc[engine]
            a["n"] += 1
            a["brier"] += (p - home_won) ** 2
            a["logloss"] += -(home_won * math.log(p) + (1 - home_won) * math.log(1 - p))
            a["margin_se"] += (sim.exp_margin - actual_margin) ** 2
            a["total_se"] += (sim.exp_total - actual_total) ** 2

    out: list[EngineScore] = []
    for engine in engines:
        a = acc[engine]
        n = int(a["n"])
        if n == 0:
            continue
        out.append(
            EngineScore(
                engine=engine,
                n=n,
                brier=round(a["brier"] / n, 4),
                logloss=round(a["logloss"] / n, 4),
                margin_rmse=round(math.sqrt(a["margin_se"] / n), 2),
                total_rmse=round(math.sqrt(a["total_se"] / n), 2),
            )
        )
    return out
