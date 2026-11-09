"""Under-bias decomposition: which sim input carries the goal-level shortfall.

Analysis tooling, not pricing. For each night the sim is rebuilt as the card
builds it (as-of MoneyPuck logs, preseason prior, official starters) and the
expected total is recomputed with one input at a time pinned to its league
value. Pinning an input that moves the sim mean toward actual goals by at
least half the base gap names that input as the carrier; pinning everything
(``all``) and still seeing the gap says the shortfall is the league level
itself (``goals_per_xg`` / ``goal_level`` / league shrink), not team inputs.
Nothing here changes pricing: a carrier is then tested on held-out Octobers
before any knob moves.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from nhl_engine import closetest, pipeline
from nhl_engine.config import Config
from nhl_engine.data import capture
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.preseason import PreseasonPrior
from nhl_engine.data.teamnames import CODES
from nhl_engine.models import markets
from nhl_engine.models.goals import game_rates
from nhl_engine.models.periods import simulate
from nhl_engine.schemas import Game, GameResult

log = logging.getLogger(__name__)

XG5 = ("xgf60_5v5", "xga60_5v5")
SPECIAL = ("pp_xgf60", "pk_xga60")
FIN = ("fin60_5v5",)
VARIANTS: tuple[str, ...] = ("base", "xg5", "special", "fin", "goalie", "all", "league_prior")
FIELDS = ("game_date", "matchup", "goals", "line", *VARIANTS)


@dataclass(frozen=True)
class Row:
    game_date: str
    matchup: str
    goals: int
    line: float | None
    sim: dict[str, float]


def _pin(
    rates: Mapping[str, float], league: Mapping[str, float], keys: Iterable[str]
) -> dict[str, float]:
    out = dict(rates)
    for k in keys:
        if k in league:
            out[k] = league[k]
    return out


def exp_total(
    home: Mapping[str, float],
    away: Mapping[str, float],
    league: Mapping[str, float],
    *,
    home_gsax: float,
    away_gsax: float,
    cfg: Config,
    seed: int,
) -> float:
    rates = game_rates(
        home,
        away,
        league,
        home_goalie_gsax60=home_gsax,
        away_goalie_gsax60=away_gsax,
        params=cfg.sim,
    )
    return markets.summary(simulate(rates, cfg.sim, seed=seed))["exp_total"]


def variants(
    home_in: pipeline.TeamInputs,
    away_in: pipeline.TeamInputs,
    league: dict[str, float],
    prior_league: Mapping[str, float],
    *,
    cfg: Config,
    seed: int,
) -> dict[str, float]:
    h, a = home_in.rates, away_in.rates
    hg, ag = home_in.starter.gsax60, away_in.starter.gsax60
    every = XG5 + SPECIAL + FIN

    def run(
        hr: Mapping[str, float],
        ar: Mapping[str, float],
        lg: Mapping[str, float],
        hgs: float,
        ags: float,
    ) -> float:
        return exp_total(hr, ar, lg, home_gsax=hgs, away_gsax=ags, cfg=cfg, seed=seed)

    return {
        "base": run(h, a, league, hg, ag),
        "xg5": run(_pin(h, league, XG5), _pin(a, league, XG5), league, hg, ag),
        "special": run(_pin(h, league, SPECIAL), _pin(a, league, SPECIAL), league, hg, ag),
        "fin": run(_pin(h, league, FIN), _pin(a, league, FIN), league, hg, ag),
        "goalie": run(h, a, league, 0.0, 0.0),
        "all": run(_pin(h, league, every), _pin(a, league, every), league, 0.0, 0.0),
        "league_prior": run(h, a, {**league, **prior_league}, hg, ag),
    }


def main_total_line(root: Path, day: Date, matchup: str) -> float | None:
    path = capture.latest_snapshot(root, day, label="board")
    if path is None:
        return None
    sels = [
        s
        for s in pipeline.selections(capture.read_snapshot(path))
        if s.matchup == matchup and s.market == "game_total" and s.side == "over" and s.line
    ]
    return min(sels, key=lambda s: abs(s.consensus - 0.5)).line if sels else None


def rebuild_day(
    *,
    root: Path,
    day: Date,
    season: int,
    cfg: Config,
    mp: MoneyPuckClient,
    prior: PreseasonPrior,
    games: Sequence[Game],
    results: dict[str, GameResult],
) -> list[Row]:
    league = pipeline.league_for(mp, day, season, prior, cfg.shrink.league_k_games)
    overrides = closetest.goalie_overrides(games, results)
    cache: dict[str, pipeline.TeamInputs] = {}

    def inputs(code: str) -> pipeline.TeamInputs:
        if code not in cache:
            cache[code] = pipeline.team_inputs(
                mp,
                code,
                day,
                season=season,
                cfg=cfg,
                prior=prior,
                league=league,
                lineups={},
                overrides=overrides,
            )
        return cache[code]

    out: list[Row] = []
    for i, g in enumerate(sorted(games, key=lambda g: g.matchup)):
        res = results.get(g.matchup)
        if res is None or not res.decided or g.home not in CODES or g.away not in CODES:
            continue
        try:
            sims = variants(
                inputs(g.home),
                inputs(g.away),
                league,
                prior.league,
                cfg=cfg,
                seed=int(day.strftime("%Y%m%d")) * 100 + i,
            )
        except Exception:  # one bad game must not sink the night
            log.exception("%s %s: decompose failed", day, g.matchup)
            continue
        out.append(
            Row(
                day.isoformat(),
                g.matchup,
                res.final_home + res.final_away,
                main_total_line(root, day, g.matchup),
                sims,
            )
        )
    return out


def write_csv(rows: Iterable[Row], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow(
                [
                    r.game_date,
                    r.matchup,
                    r.goals,
                    "" if r.line is None else r.line,
                    *(f"{r.sim[v]:.4f}" for v in VARIANTS),
                ]
            )
            n += 1
    return n


def read_csv(path: Path) -> list[Row]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [
            Row(
                d["game_date"],
                d["matchup"],
                int(d["goals"]),
                float(d["line"]) if d["line"] else None,
                {v: float(d[v]) for v in VARIANTS},
            )
            for d in csv.DictReader(fh)
        ]


def report(rows: Sequence[Row]) -> str:
    n = len(rows)
    if n == 0:
        return "decompose: no games"
    goals = sum(r.goals for r in rows) / n
    lines = [r.line for r in rows if r.line is not None]
    base = sum(r.sim["base"] for r in rows) / n
    gap = goals - base
    sd = (
        math.sqrt(sum((r.goals - r.sim["base"] - gap) ** 2 for r in rows) / (n - 1))
        if n > 1
        else math.nan
    )
    line_txt = (
        f"{sum(lines) / len(lines):.2f} (n={len(lines)})" if lines else "n/a (no board archive)"
    )
    out = [
        f"decompose: n={n} goals {goals:.2f}, line {line_txt}, "
        f"sim base {base:.2f}; goals-sim {gap:+.2f} (se {sd / math.sqrt(n):.2f})",
        f"{'pinned to league':<14} {'sim':>6} {'vs base':>8} {'share of gap':>13}  verdict",
    ]
    for v in VARIANTS[1:]:
        m = sum(r.sim[v] for r in rows) / n
        d = m - base
        share = d / gap if gap else math.nan
        verdict = (
            "carrier"
            if gap and share >= 0.5
            else ("partial" if gap and share >= 0.25 else "not it")
        )
        if v == "all" and gap and share < 0.5:
            verdict = "gap is the league level, not team inputs"
        out.append(f"{v:<14} {m:6.2f} {d:+8.2f} {share:+13.2f}  {verdict}")
    return "\n".join(out)
