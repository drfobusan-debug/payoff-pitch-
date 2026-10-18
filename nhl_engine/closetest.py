"""Preregistered close test: rebuild the sim on historical closes and regress misses.

For each night with ``hist_*`` snapshots the sim is run exactly as the card runs it, as of
that date (MoneyPuck logs before the slate, previous-season preseason prior, no isotonic
calibration, official starting goalies as confirmed overrides — information the market
also had at the close). Per game it keeps the unanchored ``model_prob`` and the no-vig
close for the home moneyline and the main total, joined to the official final.

Decision rule, fixed before the pull (ML and totals fitted separately):
    (result − p_close) = b · (p_sim − p_close)      OLS, no intercept
    |b| ≤ 1 SE  → graded-only;  b > 1 SE → anchor ≈ 1 − b, probation-gated.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nhl_engine import history, pipeline
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.calibration import Calibrator
from nhl_engine.config import Config
from nhl_engine.data.book_rules import BookRules
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.preseason import PreseasonPrior
from nhl_engine.schemas import Game, GameResult

log = logging.getLogger(__name__)

FIELDS = (
    "game_date",
    "matchup",
    "market",
    "line",
    "p_sim",
    "p_close",
    "result",
    "sim_total",
    "goals",
    "goalie_status",
)


@dataclass(frozen=True)
class Obs:
    game_date: str
    matchup: str
    market: str  # game_ml (home) | game_total (over at the main line)
    line: float | None
    p_sim: float
    p_close: float
    result: float  # 1 / 0 / 0.5 push
    sim_total: float | None
    goals: int | None
    goalie_status: str


def goalie_overrides(
    games: Sequence[Game], results: dict[str, GameResult]
) -> dict[str, dict[str, object]]:
    """Official starting goalies (boxscore ``starter`` flag, NHL player id) as confirmed overrides."""
    out: dict[str, dict[str, object]] = {}
    for g in games:
        res = results.get(g.matchup)
        if res is None:
            continue
        for p in res.players:
            if p.position == "G" and p.starter and p.player_id:
                out[p.team] = {
                    "player_id": p.player_id,
                    "name": p.name,
                    "status": "confirmed",
                    "source": "official boxscore",
                }
    return out


def _main_total(rows: list[LedgerRow]) -> LedgerRow | None:
    overs = [r for r in rows if r.market == "game_total" and r.side == "over" and r.line]
    return min(overs, key=lambda r: abs(r.consensus - 0.5)) if overs else None


def observe(card: pipeline.SlateCard, results: dict[str, GameResult]) -> list[Obs]:
    out: list[Obs] = []
    for g in card.games:
        res = results.get(g.matchup)
        if res is None or not res.decided:
            continue
        goals = res.final_home + res.final_away
        sim_total = g.summary.get("exp_total", math.nan)
        ml = next((r for r in g.rows if r.market == "game_ml" and r.side == g.home), None)
        if ml is not None:
            out.append(
                Obs(
                    card.slate_date.isoformat(),
                    g.matchup,
                    "game_ml",
                    None,
                    ml.model_prob,
                    ml.consensus,
                    1.0 if res.final_home > res.final_away else 0.0,
                    None if math.isnan(sim_total) else sim_total,
                    goals,
                    g.goalie_status,
                )
            )
        tot = _main_total(g.rows)
        if tot is not None and tot.line is not None:
            result = 0.5 if goals == tot.line else float(goals > tot.line)
            out.append(
                Obs(
                    card.slate_date.isoformat(),
                    g.matchup,
                    "game_total",
                    tot.line,
                    tot.model_prob,
                    tot.consensus,
                    result,
                    None if math.isnan(sim_total) else sim_total,
                    goals,
                    g.goalie_status,
                )
            )
    return out


def rebuild_day(
    *,
    root: Path,
    day: Date,
    season: int,
    cfg: Config,
    mp: MoneyPuckClient,
    prior: PreseasonPrior,
    rules: BookRules,
    games: Sequence[Game],
    results: dict[str, GameResult],
) -> pipeline.SlateCard:
    quotes = history.hist_rows(root, day)
    starts = {g.matchup: g.start_utc for g in games if g.start_utc}
    parsed = [d for d in (history._parse(s) for s in starts.values()) if d is not None]
    first = min(parsed, default=None)
    now = (first or datetime.combine(day, datetime.min.time(), timezone.utc)) - timedelta(minutes=1)
    return pipeline.run_slate(
        quotes,
        slate=day,
        season=season,
        cfg=cfg,
        mp=mp,
        prior=prior,
        rules=rules,
        calib=Calibrator(),
        data_dir=root,
        tag="hist",
        now=now,
        starts=starts,
        overrides=goalie_overrides(games, results),
    )


@dataclass(frozen=True)
class Fit:
    market: str
    n: int
    slope: float
    se: float
    brier_sim: float
    brier_close: float
    mean_sim: float
    mean_close: float
    mean_result: float

    @property
    def verdict(self) -> str:
        if self.n < 2 or not math.isfinite(self.se):
            return "underpowered"
        if abs(self.slope) <= self.se:
            return "graded-only (slope within 1 SE of zero)"
        if self.slope > 0:
            return f"anchor ~ {max(0.0, min(1.0, 1 - self.slope)):.2f} (probation-gated)"
        return "graded-only (slope negative)"


def fit(obs: Iterable[Obs], market: str) -> Fit:
    rows = [o for o in obs if o.market == market and o.result != 0.5]
    n = len(rows)
    x = [o.p_sim - o.p_close for o in rows]
    y = [o.result - o.p_close for o in rows]
    sxx = sum(v * v for v in x)
    slope = sum(a * b for a, b in zip(x, y, strict=True)) / sxx if sxx > 0 else math.nan
    resid = [b - slope * a for a, b in zip(x, y, strict=True)]
    se = math.sqrt(sum(r * r for r in resid) / (n - 1) / sxx) if n > 1 and sxx > 0 else math.nan
    bs = sum((o.p_sim - o.result) ** 2 for o in rows) / n if n else math.nan
    bc = sum((o.p_close - o.result) ** 2 for o in rows) / n if n else math.nan
    mean = lambda vals: sum(vals) / n if n else math.nan  # noqa: E731
    return Fit(
        market,
        n,
        slope,
        se,
        bs,
        bc,
        mean([o.p_sim for o in rows]),
        mean([o.p_close for o in rows]),
        mean([o.result for o in rows]),
    )


def goal_level_check(obs: Iterable[Obs]) -> dict[str, float]:
    """Scored goals vs the sim's expected total and the main line (one row per game)."""
    rows = [
        o
        for o in obs
        if o.market == "game_total" and o.sim_total is not None and o.goals is not None
    ]
    n = len(rows)
    if n == 0:
        return {"n": 0}
    d_sim = [o.goals - o.sim_total for o in rows if o.goals is not None and o.sim_total is not None]
    d_line = [o.goals - o.line for o in rows if o.goals is not None and o.line is not None]
    sd = pd.Series(d_sim).std() if n > 1 else math.nan
    return {
        "n": n,
        "goals": sum(o.goals for o in rows if o.goals is not None) / n,
        "sim_total": sum(o.sim_total for o in rows if o.sim_total is not None) / n,
        "line": sum(o.line for o in rows if o.line is not None) / n,
        "goals_minus_sim": sum(d_sim) / n,
        "goals_minus_sim_se": sd / math.sqrt(n) if n > 1 else math.nan,
        "goals_minus_line": sum(d_line) / len(d_line) if d_line else math.nan,
    }


def write_csv(obs: Iterable[Obs], path: Path) -> int:
    rows = [asdict(o) for o in obs]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_csv(path: Path) -> list[Obs]:
    out: list[Obs] = []
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out.append(
                Obs(
                    r["game_date"],
                    r["matchup"],
                    r["market"],
                    float(r["line"]) if r["line"] else None,
                    float(r["p_sim"]),
                    float(r["p_close"]),
                    float(r["result"]),
                    float(r["sim_total"]) if r["sim_total"] else None,
                    int(r["goals"]) if r["goals"] else None,
                    r["goalie_status"],
                )
            )
    return out


def report(obs: Sequence[Obs]) -> str:
    lines = []
    for m in ("game_ml", "game_total"):
        f = fit(obs, m)
        lines.append(
            f"{m}: n={f.n} slope={f.slope:.3f} se={f.se:.3f} -> {f.verdict}; "
            f"Brier sim {f.brier_sim:.4f} vs close {f.brier_close:.4f}; "
            f"mean p sim {f.mean_sim:.3f} / close {f.mean_close:.3f} / hit {f.mean_result:.3f}"
        )
    g = goal_level_check(obs)
    if g.get("n"):
        lines.append(
            f"goal level: n={g['n']} goals {g['goals']:.2f}, sim {g['sim_total']:.2f}, "
            f"line {g['line']:.2f}; goals-sim {g['goals_minus_sim']:+.2f} "
            f"(se {g['goals_minus_sim_se']:.2f}), goals-line {g['goals_minus_line']:+.2f}"
        )
    return "\n".join(lines)
