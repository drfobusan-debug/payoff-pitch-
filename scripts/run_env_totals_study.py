"""Measure and grade the league-level run-environment correction to the totals.

Three questions, in the order they have to be answered:

1. What does the simulator score with two league-average lineups, and what is a
   unit of non-out scale worth in runs? Those are ``models.run_env``'s
   ``BASELINE_TOTAL`` and ``RUNS_PER_SCALE``, and they are properties of today's
   simulator rather than constants -- re-run this whenever the run models change.
2. What is that scale worth to each total market's over, in log odds? Measured as
   a central difference across the clamp, the game totals through the Monte Carlo
   and the first-five totals through the Markov chain that actually prices them.
   This regenerates ``models.run_env.TOTALS``.
3. Does correcting by it beat not correcting, out of time? Graded on the ledger's
   own graded total rows, each slate's target being the league's trailing 30 days
   ending the day *before* it, so no outcome the correction is scored on is inside
   the number that set it.

Offline apart from the free StatsAPI schedule read; no odds credits.

    python -m scripts.run_env_totals_study --ledger ~/.mlb_engine/audit/ledger.csv
"""

from __future__ import annotations

import argparse
import math
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.features.removal import RemovalHazard
from mlb_engine.features.rolling import LEAGUE_RATES
from mlb_engine.models.markov_f5 import f5_from_rates
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.run_env import (
    BASELINE_TOTAL,
    RUNS_PER_SCALE,
    SCALE_CLAMP,
    TOTALS,
    scale_for_total,
    scale_rates,
)

EPS = 1e-6
GAME_LINES = (7.5, 8.5, 9.5, 10.5)
F5_LINES = (4.5, 5.5)
# Alternating hands, so the platoon layer is neutral rather than absent.
HANDS = ("L", "R", "L", "R", "L", "R", "L", "R", "R")


def _logit(p: float) -> float:
    return math.log(min(max(p, EPS), 1 - EPS) / (1 - min(max(p, EPS), 1 - EPS)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _league_team(scale: float) -> TeamSimConfig:
    """Nine league-average hitters against a league-average staff."""
    rates = scale_rates(LEAGUE_RATES, scale)
    return TeamSimConfig(
        bat_vs_starter=[dict(rates) for _ in range(9)],
        bat_vs_pen=[dict(rates) for _ in range(9)],
        bat_vs_pen_close=[dict(rates) for _ in range(9)],
        bat_vs_pen_bridge=[dict(rates) for _ in range(9)],
        bat_hands=HANDS,
        starter_hand="R",
        removal_hazard=RemovalHazard(),
    )


def _game_totals(scale: float, sims: int, seed: int) -> np.ndarray:
    res = MonteCarlo(sims, seed=seed).simulate(_league_team(scale), _league_team(scale))
    return (res.home_runs_full + res.away_runs_full).astype(float)


def measure_simulator(sims: int, seed: int) -> None:
    """The simulator's own run level and elasticity, against what is shipped."""
    lo, hi = SCALE_CLAMP
    means = {}
    for scale in (lo, 1.0, hi):
        total = _game_totals(scale, sims, seed)
        means[scale] = float(total.mean())
        print(f"  scale {scale:.2f}: {means[scale]:.3f} runs  (sd {total.std():.3f})")
    slope = (means[hi] - means[lo]) / (hi - lo)
    print(f"  measured baseline {means[1.0]:.3f} vs shipped {BASELINE_TOTAL}")
    print(f"  measured elasticity {slope:.2f} vs shipped {RUNS_PER_SCALE} runs/scale")


def measure_coefficients(sims: int, seed: int) -> dict[str, dict[float, float]]:
    """Log odds of each total market's over per unit of non-out scale."""
    lo, hi = SCALE_CLAMP
    # Common random numbers: the same seed on both legs, so the difference is the
    # scale rather than the draw.
    g_lo, g_hi = _game_totals(lo, sims, seed), _game_totals(hi, sims, seed)
    f_lo = f5_from_rates(scale_rates(LEAGUE_RATES, lo), scale_rates(LEAGUE_RATES, lo))
    f_hi = f5_from_rates(scale_rates(LEAGUE_RATES, hi), scale_rates(LEAGUE_RATES, hi))
    coef: dict[str, dict[float, float]] = {"game_total": {}, "f5_total": {}}
    for line in GAME_LINES:
        p_lo, p_hi = float((g_lo > line).mean()), float((g_hi > line).mean())
        coef["game_total"][line] = (_logit(p_hi) - _logit(p_lo)) / (hi - lo)
        print(f"  game_total {line}: {p_lo:.3f} -> {p_hi:.3f}   "
              f"{coef['game_total'][line]:.2f}/scale  (shipped "
              f"{TOTALS['game_total'].get(line, float('nan')):.2f})")
    for line in F5_LINES:
        p_lo, p_hi = f_lo.p_total_over(line), f_hi.p_total_over(line)
        coef["f5_total"][line] = (_logit(p_hi) - _logit(p_lo)) / (hi - lo)
        print(f"  f5_total   {line}: {p_lo:.3f} -> {p_hi:.3f}   "
              f"{coef['f5_total'][line]:.2f}/scale  (shipped "
              f"{TOTALS['f5_total'].get(line, float('nan')):.2f})")
    return coef


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.clip(p, EPS, 1 - EPS) - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grade(
    ledger: Path,
    client: MLBStatsClient,
    coef: dict[str, dict[float, float]],
    days: int,
) -> pd.DataFrame:
    """Score the correction against the uncorrected number, slate by slate."""
    d = pd.read_csv(ledger, low_memory=False)
    d = d[d["market"].isin(TOTALS) & d["result"].isin(["win", "loss"])].copy()
    d = d.dropna(subset=["model_prob", "line"])
    d["won"] = (d["result"] == "win").astype(float)
    d["over"] = d["selection"].str.contains("Over", case=False)

    # The engine's own read, called exactly as a slate calls it, so the study
    # cannot be right about a league the pipeline would measure differently.
    targets = {
        slate: client.league_runs_per_game(Date.fromisoformat(str(slate)), days=days)
        for slate in sorted(d["date"].unique())
    }
    d["scale"] = [
        1.0 if targets[s] is None else scale_for_total(targets[s]) for s in d["date"]
    ]
    print("\ntrailing league total by slate (first and last three)")
    read = [(s, t) for s, t in targets.items() if t is not None]
    for slate, target in read[:3] + read[-3:]:
        print(f"  {slate}  league {target:.2f} runs -> scale {scale_for_total(target):.4f}")
    d["coef"] = [
        coef.get(m, {}).get(float(ln), 0.0)
        for m, ln in zip(d["market"], d["line"], strict=True)
    ]
    # The coefficient is the over's; the under is the same opinion inverted.
    shift = d["coef"] * (d["scale"] - 1.0)
    d["corrected"] = _sigmoid(
        np.log(d["model_prob"].clip(EPS, 1 - EPS) / (1 - d["model_prob"].clip(EPS, 1 - EPS)))
        + np.where(d["over"], shift, -shift)
    )

    print(f"\ngraded total rows {len(d)} over {d['date'].nunique()} slates")
    for (market, line), sub in d.groupby(["market", "line"]):
        overs = sub[sub["over"]]
        print(f"  {market} {line}  n={len(sub):5}  over said "
              f"{overs['model_prob'].mean():.3f} -> {overs['corrected'].mean():.3f}, "
              f"hit {overs['won'].mean():.3f}   brier "
              f"{_brier(sub['model_prob'].to_numpy(), sub['won'].to_numpy()):.4f} -> "
              f"{_brier(sub['corrected'].to_numpy(), sub['won'].to_numpy()):.4f}")
    for label, sub in list(d.groupby("market")) + [("pooled", d)]:
        y = sub["won"].to_numpy()
        print(f"  {label:11} n={len(sub):5}  "
              f"brier {_brier(sub['model_prob'].to_numpy(), y):.4f} -> "
              f"{_brier(sub['corrected'].to_numpy(), y):.4f}   "
              f"log loss {_log_loss(sub['model_prob'].to_numpy(), y):.4f} -> "
              f"{_log_loss(sub['corrected'].to_numpy(), y):.4f}")

    print("\nweekly blocks, both total markets pooled")
    d["week"] = pd.to_datetime(d["date"]).dt.to_period("W").dt.start_time.dt.date
    for week, sub in d.groupby("week"):
        y = sub["won"].to_numpy()
        b0 = _brier(sub["model_prob"].to_numpy(), y)
        b1 = _brier(sub["corrected"].to_numpy(), y)
        print(f"  {week}  n={len(sub):5}  brier {b0:.4f} -> {b1:.4f} ({b1 - b0:+.4f})")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".mlb_engine/audit/ledger.csv")
    ap.add_argument("--sims", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--days", type=int, default=30, help="trailing league-total window")
    ap.add_argument("--out", type=Path, help="write the graded rows here")
    args = ap.parse_args()

    print("simulator run level, two league-average lineups")
    measure_simulator(args.sims, args.seed)
    print("\nlog odds per unit of non-out scale")
    coef = measure_coefficients(args.sims, args.seed)

    client = MLBStatsClient()
    graded = grade(args.ledger, client, coef, args.days)
    if args.out:
        graded.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
