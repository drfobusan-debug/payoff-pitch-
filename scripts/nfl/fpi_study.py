"""Is ESPN's FPI worth reading beside our own number, and is it worth pricing off?

Two different questions, and the answer to the second decides whether FPI is
allowed anywhere near a price. Measured on played seasons, per game, against the
closing moneyline the market itself settled on:

* FPI's Brier score and hit rate on its own pick;
* the de-vigged closing line's Brier score on the same games;
* how often FPI and the close disagree on the favourite, and who wins those.

A benchmark that loses to the close is a display column, nothing more: reading it
as an input would be paying to be told what the market already knew, worse.

Usage::

    python scripts/nfl/fpi_study.py --first 2022 --last 2024
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nfl_engine.data import espn, nflverse
from nfl_engine.market.fair import devig
from nfl_engine.market.odds import american_to_prob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2022)
    parser.add_argument("--last", type=int, default=2024)
    args = parser.parse_args()

    games = nflverse.games()
    rows: list[dict[str, float]] = []
    for season in range(args.first, args.last + 1):
        weeks = sorted({int(w) for w in games[games.season == season].week})
        for week in weeks:
            for game in espn.projections(season, week):
                frame = games[
                    (games.season == season) & (games.week == week) & (games.home_team == game.home)
                ]
                if frame.empty:
                    continue
                row = frame.iloc[0]
                if pd.isna(row.result) or pd.isna(row.home_moneyline):
                    continue
                if float(row.result) == 0.0:
                    continue
                market = devig(
                    [
                        american_to_prob(float(row.home_moneyline)),
                        american_to_prob(float(row.away_moneyline)),
                    ]
                )[0]
                home_won = 1.0 if float(row.result) > 0 else 0.0
                rows.append(
                    {
                        "fpi": game.home_prob / 100.0,
                        "market": market,
                        "home_won": home_won,
                        "margin": game.home_margin,
                        "actual_margin": float(row.result),
                    }
                )
        print(f"  {season}: {len(rows)} games so far", flush=True)

    if not rows:
        print("no games with both an FPI projection and a closing moneyline")
        return
    frame = pd.DataFrame(rows)
    fpi = frame.fpi.to_numpy()
    market = frame.market.to_numpy()
    won = frame.home_won.to_numpy()

    print(f"\ngames {len(frame)} ({args.first}-{args.last})")
    for name, prob in (("FPI", fpi), ("close", market)):
        brier = float(np.mean((prob - won) ** 2))
        picked = np.where(prob >= 0.5, won, 1.0 - won)
        print(
            f"{name:6s} brier {brier:.5f}  hit {picked.mean():.4f}"
            f"  mean prob on pick {np.maximum(prob, 1 - prob).mean():.4f}"
        )
    diff = (fpi >= 0.5) != (market >= 0.5)
    if diff.any():
        fpi_right = np.where(fpi[diff] >= 0.5, won[diff], 1.0 - won[diff])
        print(
            f"\ndisagree on the favourite: {int(diff.sum())} games"
            f" ({diff.mean():.3f}); FPI right {fpi_right.mean():.4f}"
        )
    err_fpi = float(np.mean(np.abs(frame.margin - frame.actual_margin)))
    print(f"\nFPI predicted margin: mean absolute error {err_fpi:.3f} points")


if __name__ == "__main__":
    main()
