"""Set the simulator's run environment to the league's, and grade the move first.

The shape study established what is *not* wrong: over the games in the ledger the
model's implied mean total matches the final score to +0.04 runs. That is a
statement about the games the model priced, not about the simulator -- the two only
agree because the calibration map and the edge floors sit between them. Replayed
with league-average lineups and staffs the simulator itself scores
``run_env.BASELINE_TOTAL`` runs against a league playing 8.95 (season to date), so
the raw run environment *is* hot, and every over in the book inherits it.

This script does three things, in the order they have to be done in:

* re-measures the simulator's own baseline and its elasticity to the non-out scale,
  which are the two constants ``models.run_env`` solves the correction from;
* measures what a scale is worth to each market's probability, by repricing a
  league-average game at both scales -- these are the deltas the grader applies,
  so no market is patched with a number fitted to its own outcomes;
* grades the correction **walk-forward** on the graded ledger: the scale is solved
  from the league total on the training dates only, then applied to the later
  dates' rows and scored against what happened (Brier and log loss per market,
  against the model's own logged probability).

Nothing here writes a probability. The correction it grades is off by default
(``MLBE_RUN_ENV``); this is the evidence for turning it on.

    python scripts/run_env_study.py [--sims N] [--split DATE] [--target RUNS]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.features.removal import RemovalHazard
from mlb_engine.features.rolling import LEAGUE_RATES
from mlb_engine.models import run_env
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.models.props import p_over
from scripts.total_shape_study import LINES, fit_implied, join_finals, load_totals

log = logging.getLogger("run_env")

# A league-average lineup has no handedness pattern worth inventing; alternating is
# the neutral one and keeps the platoon machinery in the path being measured.
HANDS = ("L", "R", "L", "R", "L", "R", "L", "R", "R")

# Markets graded from the ledger, mapped to the (stat, line) the simulator reports.
# H+R+RBI is a sum, so it is built the way props.py builds it.
GRADED = {
    "batter_tb": ("TB", (1.5, 2.5)),
    "batter_rbi": ("RBI", (0.5,)),
    "batter_hrr": ("HRR", (1.5, 2.5)),
}

# The grid ``--coeffs`` measures: every market and line the correction is allowed to
# move, which is every counting market a book posts a number for. Wider than
# ``GRADED`` on purpose -- the coefficient is a simulator measurement, so it is
# taken for lines the ledger is too thin to grade rather than left to a guess.
COEFF_GRID: dict[str, tuple[str, tuple[float, ...]]] = {
    "game_total": ("total", (6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.5)),
    "batter_h": ("H", (0.5, 1.5, 2.5)),
    "batter_1b": ("1B", (0.5, 1.5)),
    "batter_2b": ("2B", (0.5,)),
    "batter_hr": ("HR", (0.5,)),
    "batter_r": ("R", (0.5, 1.5)),
    "batter_rbi": ("RBI", (0.5, 1.5)),
    "batter_tb": ("TB", (0.5, 1.5, 2.5, 3.5)),
    "batter_hrr": ("HRR", (0.5, 1.5, 2.5, 3.5)),
}


def league_team(scale: float) -> TeamSimConfig:
    """Nine league-average slots against a league-average staff, at one scale."""
    rates = run_env.scale_rates(LEAGUE_RATES, scale)
    nine = [dict(rates) for _ in range(9)]
    return TeamSimConfig(
        bat_vs_starter=nine,
        bat_vs_pen=[dict(rates) for _ in range(9)],
        bat_vs_pen_close=[dict(rates) for _ in range(9)],
        bat_vs_pen_bridge=[dict(rates) for _ in range(9)],
        bat_hands=HANDS,
        starter_hand="R",
        removal_hazard=RemovalHazard(),
    )


def sim_at(scale: float, sims: int, seed: int = 7) -> dict[str, float]:
    """A league-average game at one scale: the total, and each market's price.

    The same seed at every scale, so a difference between two calls is the scale
    and not the draw.
    """
    res = MonteCarlo(sims, seed=seed).simulate(league_team(scale), league_team(scale))
    total = (res.home_runs_full + res.away_runs_full).astype(float)
    out: dict[str, float] = {"total": float(total.mean())}
    for line in LINES:
        out[f"game_total|{line}"] = float((total > line).mean())
    bat = res.bat["home"]
    tb = (bat["1B"][:, :] + 2 * bat["2B"][:, :] + 3 * bat["3B"][:, :] + 4 * bat["HR"][:, :]).astype(
        float
    )
    hrr = (bat["H"] + bat["R"] + bat["RBI"]).astype(float)
    arrays = {"TB": tb, "RBI": bat["RBI"].astype(float), "HRR": hrr}
    for market, (stat, lines) in GRADED.items():
        arr = arrays[stat]
        for line in lines:
            # Averaged over the nine slots: the delta is a league-level number, and
            # a per-slot one would be fitting the lineup this study invented.
            out[f"{market}|{line}"] = float(
                np.mean([p_over(arr[:, slot], line) for slot in range(9)])
            )
    return out


def grid_prices(scale: float, sims: int, seed: int = 7) -> dict[str, float]:
    """Every ``COEFF_GRID`` over-probability in a league-average game at one scale."""
    res = MonteCarlo(sims, seed=seed).simulate(league_team(scale), league_team(scale))
    total = (res.home_runs_full + res.away_runs_full).astype(float)
    bat = res.bat["home"]
    tb = (bat["1B"] + 2 * bat["2B"] + 3 * bat["3B"] + 4 * bat["HR"]).astype(float)
    arrays: dict[str, np.ndarray] = {
        "H": bat["H"].astype(float),
        "1B": bat["1B"].astype(float),
        "2B": bat["2B"].astype(float),
        "HR": bat["HR"].astype(float),
        "R": bat["R"].astype(float),
        "RBI": bat["RBI"].astype(float),
        "TB": tb,
        "HRR": (bat["H"] + bat["R"] + bat["RBI"]).astype(float),
    }
    out: dict[str, float] = {}
    for market, (stat, lines) in COEFF_GRID.items():
        for line in lines:
            if stat == "total":
                out[f"{market}|{line}"] = float((total > line).mean())
                continue
            arr = arrays[stat]
            # Averaged over the nine slots: the coefficient is a league-level
            # number, and a per-slot one would be fitting the invented lineup.
            out[f"{market}|{line}"] = float(
                np.mean([p_over(arr[:, slot], line) for slot in range(9)])
            )
    return out


def coefficients(sims: int, lo: float, hi: float) -> None:
    """Print ``run_env.LOGIT_PER_SCALE``: log odds of the over per unit of scale.

    Measured as a central difference across the clamp on common random numbers (the
    same seed at both scales), which is what makes a 1-2 point move readable at
    this many games. Paste-ready, because these are the constants the correction is
    applied from and they should be re-measured, not hand-adjusted.
    """
    low, high = grid_prices(lo, sims), grid_prices(hi, sims)
    print(f"\nLOGIT_PER_SCALE, central difference {lo:.2f}..{hi:.2f}, {sims} games per point")
    for market, (_, lines) in COEFF_GRID.items():
        cells = []
        for line in lines:
            key = f"{market}|{line}"
            p0, p1 = low[key], high[key]
            if not (0.005 < p0 < 0.995 and 0.005 < p1 < 0.995):
                continue
            slope = (np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))) / (hi - lo)
            cells.append(f"{line}: {slope:.2f}")
        print(f'    "{market}": {{{", ".join(cells)}}},')


def elasticity(sims: int, lo: float, hi: float) -> tuple[float, dict[str, float]]:
    """Runs per unit of scale, and the baseline total at scale 1.0."""
    base = sim_at(1.0, sims)
    low = sim_at(lo, sims)
    high = sim_at(hi, sims)
    slope = (high["total"] - low["total"]) / (hi - lo)
    print(
        f"\nsimulator, {sims} games per point"
        f"\n  scale {lo:.2f}: {low['total']:.3f} runs"
        f"\n  scale 1.00: {base['total']:.3f} runs   (run_env.BASELINE_TOTAL"
        f" {run_env.BASELINE_TOTAL})"
        f"\n  scale {hi:.2f}: {high['total']:.3f} runs"
        f"\n  elasticity {slope:.1f} runs per unit scale   (run_env.RUNS_PER_SCALE"
        f" {run_env.RUNS_PER_SCALE})"
    )
    return slope, base


def logit_deltas(scale: float, sims: int, base: dict[str, float]) -> dict[str, float]:
    """What the scale is worth to each market, in log odds of the over."""
    moved = sim_at(scale, sims)
    out: dict[str, float] = {}
    print(f"\nwhat scale {scale:.4f} is worth per market (league-average game)")
    for key, p0 in base.items():
        if key == "total" or key not in moved:
            continue
        p1 = moved[key]
        if not (0.01 < p0 < 0.99 and 0.01 < p1 < 0.99):
            continue
        out[key] = float(np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0)))
        print(f"  {key:<18} {p0:.3f} -> {p1:.3f}  ({100 * (p1 - p0):+.1f} pts)")
    return out


def shift(prob: pd.Series, delta: float, over: pd.Series) -> pd.Series:
    """Move a logged probability by the market's log-odds delta, sign by side."""
    p = prob.clip(1e-4, 1 - 1e-4)
    signed = np.where(over, delta, -delta)
    z = np.log(p / (1 - p)) + signed
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=prob.index)


def score(name: str, prob: pd.Series, won: pd.Series) -> tuple[float, float]:
    brier = float(np.mean((prob - won) ** 2))
    p = prob.clip(1e-6, 1 - 1e-6)
    ll = float(-np.mean(won * np.log(p) + (1 - won) * np.log(1 - p)))
    print(f"  {name:<12} brier {brier:.4f}  log loss {ll:.4f}")
    return brier, ll


def graded_rows(ledger: Path) -> pd.DataFrame:
    d = pd.read_csv(ledger, low_memory=False)
    if "source" in d.columns:
        d = d[(d["source"].isna()) | (d["source"] == "engine")]
    d = d[d["market"].isin(GRADED) & d["result"].isin(["win", "loss"])].copy()
    d["won"] = (d["result"] == "win").astype(float)
    d["over"] = d["selection"].str.contains("Over| o", case=False, regex=True)
    return d.dropna(subset=["model_prob", "line", "date"])


def grade_props(rows: pd.DataFrame, deltas: dict[str, float], split: str) -> None:
    test = rows[rows["date"] > split]
    print(f"\nbatter markets after {split}: {len(test)} graded rows")
    for market, (_, lines) in GRADED.items():
        for line in lines:
            key = f"{market}|{line}"
            sub = test[(test["market"] == market) & (np.isclose(test["line"], line))]
            if key not in deltas or len(sub) < 100:
                continue
            print(f"\n  {market} {line} ({len(sub)} rows, realized {sub['won'].mean():.3f})")
            base = score("logged", sub["model_prob"], sub["won"])
            corr = score(
                "corrected", shift(sub["model_prob"], deltas[key], sub["over"]), sub["won"]
            )
            print(f"    brier {corr[0] - base[0]:+.4f}   log loss {corr[1] - base[1]:+.4f}")


def grade_totals(games: pd.DataFrame, split: str, runs: float) -> None:
    """Grade the same correction on game totals, where it is a mean shift in runs."""
    test = games[games["date"] > split]
    print(f"\ngame totals after {split}: {len(test)} games, shifting the mean {runs:+.3f} runs")
    for line in LINES:
        p0 = test[line]
        p1 = pd.Series(norm.sf((line - (test["mu"] + runs)) / test["sd"]), index=test.index)
        won = (test["total"] > line).astype(float)
        print(f"\n  o{line} (realized {won.mean():.3f})")
        base = score("logged", p0, won)
        corr = score("corrected", p1, won)
        print(f"    brier {corr[0] - base[0]:+.4f}   log loss {corr[1] - base[1]:+.4f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".mlb_engine/audit/ledger.csv")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".mlb_engine/cache/boxscores")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--split", default="2026-08-05", help="fit on or before, grade after")
    ap.add_argument("--target", type=float, help="league runs per game (default: measured)")
    ap.add_argument(
        "--coeffs",
        action="store_true",
        help="re-measure run_env.LOGIT_PER_SCALE and print it, then stop",
    )
    args = ap.parse_args()

    if args.coeffs:
        coefficients(args.sims, 0.96, 1.04)
        return

    games = fit_implied(join_finals(load_totals(args.ledger), args.cache))
    train = games[games["date"] <= args.split]

    # Walk-forward: the target is the league the *training* dates played, so no
    # graded outcome the correction is scored on is inside the number that set it.
    target = args.target
    if target is None:
        target = float(train["total"].mean())
    season = MLBStatsClient().league_total_runs(Date.today().year)
    print(
        f"\nleague runs per game: target {target:.3f} from {len(train)} games"
        f" on or before {args.split}"
        + (f"   (season to date, standings: {season:.3f})" if season is not None else "")
    )

    _, base = elasticity(args.sims, 0.96, 1.04)
    scale = run_env.scale_for_total(target)
    print(f"scale that reproduces the target: {scale:.4f}")
    deltas = logit_deltas(scale, args.sims, base)

    grade_props(graded_rows(args.ledger), deltas, args.split)
    grade_totals(games, args.split, (scale - 1.0) * run_env.RUNS_PER_SCALE)


if __name__ == "__main__":
    main()
