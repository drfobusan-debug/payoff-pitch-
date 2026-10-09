"""Measure the league rates the sim prices against, and the sim's goal level.

    python scripts/nhl/league_level_study.py --from 2016 --to 2025

Two questions, both answered on MoneyPuck regular seasons:

1. **League rate shrink.** ``pipeline.league_for`` divides every team's rate by
   the league's. Early in a season that league rate comes from a handful of
   games, so 5v5 finishing (``fin60_5v5``) or the penalty rate can sit far from
   where the season will settle and drag every game the same way. For each
   season and each count of team-games played (ordered by date) this compares
   ``w x live + (1 - w) x prior``, ``w = n / (n + k)``, against the rate over the
   rest of the season, where the prior is the previous season's league rate,
   and picks ``k`` (in team-games) per metric by squared error.
2. **Goal level.** With both teams at the season's league rates and league
   goalies, the sim's regulation + OT goals per game against what the league
   actually scored (MoneyPuck all-situation goals, which exclude the shootout).
   The ratio is the multiplier on every goal rate that puts a league-average
   game at the league's real scoring.
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from nhl_engine.config import SimParams, cache_dir
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.teamnames import CODES
from nhl_engine.features.strength import METRICS, game_table, league_rates
from nhl_engine.models.goals import game_rates
from nhl_engine.models.periods import simulate

CHECKPOINTS = (16, 32, 64, 128, 256, 512)
K_GRID = (0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 1e12)


def season_tables(mp: MoneyPuckClient, seasons: range) -> dict[int, pd.DataFrame]:
    """Every team-game of each season, one row each, date-ordered."""
    by_season: dict[int, list[pd.DataFrame]] = {s: [] for s in seasons}
    for code in sorted(CODES):
        games = mp.team_games(code)
        for s in seasons:
            g = games[games["season"] == s]
            if not g.empty:
                by_season[s].append(game_table(g))
    return {
        s: pd.concat(f, ignore_index=True).sort_values("gameDate", kind="stable")
        for s, f in by_season.items()
        if f
    }


def _rate(df: pd.DataFrame, key: str, scale: float) -> float:
    exp = float(df[f"{key}__exp"].sum())
    return float(df[f"{key}__num"].sum()) / exp * scale if exp > 0 else float("nan")


def shrink_fit(tables: dict[int, pd.DataFrame]) -> dict[str, float]:
    seasons = sorted(tables)
    out: dict[str, float] = {}
    print(
        f"{'metric':<14}"
        + "".join(f"{f'k={k:g}':>10}" for k in K_GRID[:-1])
        + f"{'k=inf':>10}  best"
    )
    for m in METRICS:
        sse = np.zeros(len(K_GRID))
        for s in seasons[1:]:
            prior = _rate(tables[s - 1], m.key, m.scale)
            t = tables[s]
            for n in CHECKPOINTS:
                if n >= len(t) - 100:
                    continue
                live = _rate(t.iloc[:n], m.key, m.scale)
                rest = _rate(t.iloc[n:], m.key, m.scale)
                if not all(np.isfinite([prior, live, rest])):
                    continue
                for i, k in enumerate(K_GRID):
                    w = n / (n + k)
                    sse[i] += (w * live + (1 - w) * prior - rest) ** 2
        best = K_GRID[int(np.argmin(sse))]
        out[m.key] = best
        rel = sse / sse.min() if sse.min() > 0 else sse
        print(f"{m.key:<14}" + "".join(f"{r:>10.2f}" for r in rel) + f"  {best:g}")
    return out


def level_fit(tables: dict[int, pd.DataFrame], mp: MoneyPuckClient, seasons: list[int]) -> float:
    params = SimParams()
    act_tot, sim_tot = 0.0, 0.0
    for s in seasons:
        lg = league_rates({"all": tables[s]})
        games = pd.concat(
            [g[(g["season"] == s) & (g["situation"] == "all")] for g in map(mp.team_games, CODES)]
        )
        actual = 2.0 * float(games["goalsFor"].sum()) / len(games)
        ties = float((games["goalsFor"] == games["goalsAgainst"]).mean())
        sim = simulate(
            game_rates(lg, lg, lg, home_goalie_gsax60=0.0, away_goalie_gsax60=0.0, params=params),
            params,
            draws=40000,
            seed=s,
        )
        model = float((sim.ot_goals_h + sim.ot_goals_a).mean())
        act_tot += actual
        sim_tot += model
        print(
            f"{s}: actual reg+OT goals/game {actual:.3f} (tied finals {ties:.3f}) "
            f"sim {model:.3f} ratio {actual / model:.3f}"
        )
    ratio = act_tot / sim_tot
    print(f"pooled {seasons[0]}-{seasons[-1]}: goal level multiplier {ratio:.3f}")
    return ratio


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", type=int, default=2016)
    ap.add_argument("--to", dest="end", type=int, default=2025)
    ap.add_argument("--level-from", type=int, default=2022)
    args = ap.parse_args(argv)
    mp = MoneyPuckClient(cache_dir=cache_dir(), cache_ttl=30 * 86400)
    tables = season_tables(mp, range(args.start, args.end + 1))
    print(
        f"league rate shrink: SSE relative to best, seasons {args.start + 1}-{args.end}, "
        f"checkpoints {CHECKPOINTS} team-games"
    )
    shrink_fit(tables)
    print()
    level_fit(tables, mp, list(range(args.level_from, args.end + 1)))
    print(f"\nrun {date.today()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
