"""Validate the possession simulator's *shape* against 27 seasons of results.

The point of separating the score distribution from the forecast is that this
study needs no ratings at all. It hands the simulator the market's own closing
spread and total as the means for every historical game, so any mismatch between
the simulated and actual margin distributions is a fault in the distribution --
not in a rating, not in a feature.

Pass conditions, all measured on 2015-2025 (n=3,028) unless noted:

* P(margin = 3) near 14.8%, P(7) near 8.7%, P(6) near 6.9% -- a normal with the
  same moments puts 2.7% on a 3, which is what makes this the decisive test.
* ATS push rate 9.0% on a 3-point spread (n=1,153, 1999-2025), 6.5% on a 7,
  1.5% on totals.
* margin sd near 14.2, total sd near 13.9.
* over rate 48.7% and home cover rate 47.7% -- the boring pair, which catches
  sign errors that the shape tests would sail past.

Usage::

    python scripts/nfl/score_shape_study.py            # validate
    python scripts/nfl/score_shape_study.py --fit       # refit FORM_SD / anchor
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfl_engine.data import nflverse
from nfl_engine.models import drives as drives_mod
from nfl_engine.models.drives import DriveSim, ExpectedGame
from nfl_engine.models.montecarlo import NormalSim

KEY_NUMBERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17]
MODERN_FIRST_SEASON = 2015


def actual_margin_histogram(games: pd.DataFrame) -> dict[int, float]:
    margin = games.result.abs()
    return {k: float((margin == k).mean()) for k in KEY_NUMBERS}


def simulate_slate(games: pd.DataFrame, *, engine: str, n_sims: int, seed: int) -> pd.DataFrame:
    """One trial per game, means taken from that game's closing line.

    A single trial per game rather than a full 40k run each: the question is
    what the distribution looks like over the population of NFL games, and
    3,028 games x 300 trials answers it in a minute.
    """
    trials = max(1, n_sims)
    rows = []
    sim: DriveSim | NormalSim
    if engine == "drives":
        sim = DriveSim(n_sims=trials, seed=seed)
    else:
        sim = NormalSim(n_sims=trials, seed=seed)
    for row in games.itertuples():
        # spread_line is the home team's handicap with the sign flipped: a
        # positive spread_line means the home team is favoured by that much.
        exp_margin = float(row.spread_line)
        exp_total = float(row.total_line)
        home_points = (exp_total + exp_margin) / 2.0
        away_points = (exp_total - exp_margin) / 2.0
        dist = sim.simulate(ExpectedGame(home_points=home_points, away_points=away_points))
        rows.append(
            pd.DataFrame(
                {
                    "spread_line": exp_margin,
                    "total_line": exp_total,
                    "margin": dist.margins(),
                    "total": dist.totals(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


KEY_SPREADS = (3, 6, 7, 10, 14)


def _push_by_spread(margin: np.ndarray, spread: np.ndarray) -> dict[int, float]:
    """ATS push rate by key number.

    A push needs the margin to land on the spread *with its sign* -- the home
    team laying 3 and winning by 3, not either team winning by 3 -- so it runs
    around half of P(|margin| = 3).
    """
    whole = spread % 1 == 0
    out: dict[int, float] = {}
    for point in KEY_SPREADS:
        mask = whole & (np.abs(spread) == point)
        out[point] = float(np.mean(margin[mask] == spread[mask])) if mask.any() else float("nan")
    return out


def _summary(margin: np.ndarray, total: np.ndarray, spread: np.ndarray, line: np.ndarray) -> dict:
    absolute = np.abs(margin)
    cover = margin - spread
    whole_total = line % 1 == 0
    return {
        "push_by_spread": _push_by_spread(margin, spread),
        "whole_total_push": float(np.mean(total[whole_total] == line[whole_total])),
        "margin_sd": float(margin.std()),
        "total_sd": float(total.std()),
        "margin_mean": float(margin.mean()),
        "total_mean": float(total.mean()),
        "home_cover": float((cover > 0).mean()),
        "over": float((total > line).mean()),
        "ats_push": float((cover == 0).mean()),
        "total_push": float((total == line).mean()),
        "hist": {k: float((absolute == k).mean()) for k in KEY_NUMBERS},
    }


def report(games: pd.DataFrame, engine: str, n_sims: int, seed: int) -> dict:
    sim = simulate_slate(games, engine=engine, n_sims=n_sims, seed=seed)
    return _summary(
        sim.margin.to_numpy(),
        sim.total.to_numpy(),
        sim.spread_line.to_numpy(),
        sim.total_line.to_numpy(),
    )


def actual_summary(games: pd.DataFrame) -> dict:
    return _summary(
        games.result.to_numpy(dtype=float),
        games.total.to_numpy(dtype=float),
        games.spread_line.to_numpy(dtype=float),
        games.total_line.to_numpy(dtype=float),
    )


def _print_comparison(actual: dict, sims: dict[str, dict]) -> None:
    names = list(sims)
    head = "".join(f"{name:>10}" for name in names)
    print(f"\n{'metric':<16}{'actual':>10}{head}")
    for key in ("margin_mean", "margin_sd", "total_mean", "total_sd", "home_cover", "over"):
        line = f"{key:<16}{actual[key]:>10.3f}"
        for name in names:
            line += f"{sims[name][key]:>10.3f}"
        print(line)
    print(f"\n{'|margin|':<16}{'actual %':>10}{head}")
    for k in KEY_NUMBERS:
        line = f"{k:<16}{100 * actual['hist'][k]:>10.2f}"
        for name in names:
            line += f"{100 * sims[name]['hist'][k]:>10.2f}"
        print(line)
    print(f"\n{'ATS push %':<16}{'actual':>10}{head}")
    for point in KEY_SPREADS:
        line = f"{'spread ' + str(point):<16}{100 * actual['push_by_spread'][point]:>10.2f}"
        for name in names:
            line += f"{100 * sims[name]['push_by_spread'][point]:>10.2f}"
        print(line)
    line = f"{'whole total':<16}{100 * actual['whole_total_push']:>10.2f}"
    for name in names:
        line += f"{100 * sims[name]['whole_total_push']:>10.2f}"
    print(line)


def _sample_sizes(games: pd.DataFrame) -> None:
    whole = games[games.spread_line % 1 == 0]
    counts = ", ".join(
        f"{point}: n={int((whole.spread_line.abs() == point).sum()):,}" for point in KEY_SPREADS
    )
    print(f"\nwhole-number spread sample -- {counts}")


def fit(games: pd.DataFrame, n_sims: int, seed: int) -> None:
    """Grid-search the two free parameters, then refit the anchor.

    Everything else in :mod:`nfl_engine.models.drives` is measured; these two are
    not identifiable from play-by-play, so they are fitted to the margin
    histogram and reported here rather than tuned silently.
    """
    best: tuple[float, float, int] | None = None
    for form_sd in (0.08, 0.10, 0.12, 0.14, 0.16):
        for late_slots in (2, 4, 6):
            drives_mod.FORM_SD = form_sd
            drives_mod.LATE_SLOTS = late_slots
            out = report(games, "drives", n_sims, seed)
            actual = actual_summary(games)
            loss = sum(
                (out["hist"][k] - actual["hist"][k]) ** 2 for k in KEY_NUMBERS
            ) * 100.0 + (out["margin_sd"] - actual["margin_sd"]) ** 2 / 100.0
            print(
                f"form_sd={form_sd:.2f} late_slots={late_slots}: loss={loss:.4f} "
                f"margin_sd={out['margin_sd']:.2f} P(3)={100 * out['hist'][3]:.2f} "
                f"total_mean={out['total_mean']:.2f}"
            )
            if best is None or loss < best[0]:
                best = (loss, form_sd, late_slots)
    if best is not None:
        print(f"\nbest: form_sd={best[1]} late_slots={best[2]} (loss {best[0]:.4f})")
        drives_mod.FORM_SD, drives_mod.LATE_SLOTS = best[1], best[2]

    print("\nanchors: requested -> realized")
    drives_mod.ANCHOR_TOTAL_A, drives_mod.ANCHOR_TOTAL_B = 0.0, 1.0
    drives_mod.ANCHOR_MARGIN_B = 1.0
    sim = DriveSim(n_sims=30000, seed=seed)
    requested = np.arange(15.0, 32.1, 4.0)
    realized = []
    for points in requested:
        dist = sim.simulate(ExpectedGame(home_points=float(points), away_points=float(points)))
        realized.append(dist.mean_total() / 2.0)
        print(f"  per-team points {points:5.1f} -> {realized[-1]:5.2f}")
    slope, intercept = np.polyfit(requested, np.array(realized), 1)
    print(f"  ANCHOR_TOTAL_A, ANCHOR_TOTAL_B = {intercept:.4f}, {slope:.4f}")
    drives_mod.ANCHOR_TOTAL_A, drives_mod.ANCHOR_TOTAL_B = float(intercept), float(slope)

    margins_in = np.array([1.0, 3.0, 6.0, 10.0, 14.0])
    margins_out = []
    for margin in margins_in:
        dist = sim.simulate(
            ExpectedGame(home_points=22.8 + margin / 2.0, away_points=22.8 - margin / 2.0)
        )
        margins_out.append(dist.mean_margin())
        print(f"  margin {margin:5.1f} -> {margins_out[-1]:5.2f}")
    margin_slope = float(
        np.linalg.lstsq(margins_in.reshape(-1, 1), np.array(margins_out), rcond=None)[0][0]
    )
    print(f"  ANCHOR_MARGIN_B = {margin_slope:.4f}")
    drives_mod.ANCHOR_MARGIN_B = margin_slope



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=MODERN_FIRST_SEASON)
    parser.add_argument("--trials", type=int, default=300, help="simulated trials per game")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fit", action="store_true", help="grid-search the free parameters")
    args = parser.parse_args()

    games = nflverse.graded_games()
    if games.empty:
        raise SystemExit("no graded games available")
    games = games[games.season >= args.first_season].reset_index(drop=True)
    print(f"{len(games):,} graded games, {games.season.min()}-{games.season.max()}")

    if args.fit:
        fit(games, args.trials, args.seed)
        return

    actual = actual_summary(games)
    sims = {
        "drives": report(games, "drives", args.trials, args.seed),
        "normal": report(games, "normal", args.trials, args.seed),
    }
    _print_comparison(actual, sims)
    _sample_sizes(games)


if __name__ == "__main__":
    main()
