"""Which de-vig, measured on every closing moneyline nflverse carries.

Four ways to strip the hold off a two-way price, scored against what happened,
and -- the part that decides it -- bucketed by the favourite's booked price, since
the methods only diverge at the ends. Regenerates the table in the docstring of
:mod:`nfl_engine.market.fair`.

Usage::

    python scripts/nfl/devig_study.py
    python scripts/nfl/devig_study.py --first 2015
"""

from __future__ import annotations

import argparse

import numpy as np

from nfl_engine.data import nflverse
from nfl_engine.market.fair import METHODS, devig
from nfl_engine.market.odds import american_to_prob

BUCKETS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2006)
    args = parser.parse_args()

    games = nflverse.games()
    graded = games[
        games.result.notna()
        & games.home_moneyline.notna()
        & games.away_moneyline.notna()
        & (games.result != 0)
        & (games.season >= args.first)
    ].copy()
    home_implied = graded.home_moneyline.map(american_to_prob).to_numpy(dtype=float)
    away_implied = graded.away_moneyline.map(american_to_prob).to_numpy(dtype=float)
    won = (graded.result > 0).to_numpy().astype(float)
    booked = home_implied + away_implied
    print(
        f"n={len(graded)}  {int(graded.season.min())}-{int(graded.season.max())}"
        f"  median hold {np.median(booked - 1.0):.4f}"
    )

    fair: dict[str, np.ndarray] = {}
    for name in METHODS:
        fair[name] = np.array(
            [devig([h, a], name)[0] for h, a in zip(home_implied, away_implied, strict=True)]
        )
        probs = np.clip(fair[name], 1e-6, 1.0 - 1e-6)
        brier = float(np.mean((probs - won) ** 2))
        logloss = float(-np.mean(won * np.log(probs) + (1 - won) * np.log(1 - probs)))
        print(f"  {name:12s} Brier {brier:.5f}  log loss {logloss:.5f}")

    print("\n  fair minus realised, by the favourite's booked price:")
    header = "    favourite       n  realised" + "".join(f"  {n[:5]:>7s}" for n in METHODS)
    print(header)
    favourite = np.maximum(home_implied, away_implied)
    home_is_favourite = home_implied >= away_implied
    for low, high in zip(BUCKETS[:-1], BUCKETS[1:], strict=True):
        pick = (favourite >= low) & (favourite < high)
        if pick.sum() < 60:
            continue
        realised = float(np.where(home_is_favourite, won, 1 - won)[pick].mean())
        line = f"    {low:.1f}-{high:.1f}   {int(pick.sum()):6d}    {realised:.4f}"
        for name in METHODS:
            side = np.where(home_is_favourite, fair[name], 1 - fair[name])[pick]
            line += f"  {side.mean() - realised:+7.4f}"
        print(line)


if __name__ == "__main__":
    main()
