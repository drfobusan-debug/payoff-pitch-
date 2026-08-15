"""Does the simulator price the rungs the market did not quote?

Phase 5's only forecasting claim is narrow: anchored to the market's own closing
number, the possession distribution can say what a *different* rung of the same
ladder is worth. That is testable without a historical multi-book archive --
anchor to the closing spread and total, price every neighbouring rung, and compare
the model's cover probability to the realised cover rate at that rung.

Reported per offset, over every game with a closing spread and total:

* model mean probability against realised, with a standard error;
* the same for totals;
* the value of each half-point step, model against realised, since that is the
  number the ladder-shopping is actually spending.

Usage::

    python scripts/nfl/rung_study.py --first 2015 --sims 20000
"""

from __future__ import annotations

import argparse

import numpy as np

from nfl_engine.data import nflverse
from nfl_engine.models.distribution import ScoreDistribution
from nfl_engine.models.drives import DriveSim, ExpectedGame

OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2015)
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()

    games = nflverse.games()
    graded = games[
        games.result.notna()
        & games.spread_line.notna()
        & games.total_line.notna()
        & (games.season >= args.first)
    ].copy()
    print(f"n={len(graded)}  {int(graded.season.min())}-{int(graded.season.max())}")

    sim = DriveSim(n_sims=args.sims, seed=17)
    cache: dict[tuple[float, float], ScoreDistribution] = {}
    spread_model: dict[float, list[float]] = {off: [] for off in OFFSETS}
    spread_real: dict[float, list[float]] = {off: [] for off in OFFSETS}
    total_model: dict[float, list[float]] = {off: [] for off in OFFSETS}
    total_real: dict[float, list[float]] = {off: [] for off in OFFSETS}

    for row in graded.itertuples():
        margin = float(row.spread_line)  # positive = home favoured, nflverse convention
        total = float(row.total_line)
        key = (round(margin * 2) / 2, round(total * 2) / 2)
        if key not in cache:
            mean_margin, mean_total = key
            cache[key] = sim.simulate(
                ExpectedGame(
                    home_points=(mean_total + mean_margin) / 2.0,
                    away_points=(mean_total - mean_margin) / 2.0,
                )
            )
        dist = cache[key]
        actual_margin = float(row.result)
        actual_total = float(row.home_score) + float(row.away_score)
        for offset in OFFSETS:
            # Home laying its own closing number, moved by ``offset``.
            home_point = -(margin + offset)
            prob = dist.spread(home_point)
            if prob.push < 1.0:
                spread_model[offset].append(prob.conditional)
                covered = actual_margin + home_point
                if covered != 0:
                    spread_real[offset].append(1.0 if covered > 0 else 0.0)
                else:
                    spread_model[offset].pop()
            line = total + offset
            over = dist.total(line, over=True)
            if actual_total != line:
                total_model[offset].append(over.conditional)
                total_real[offset].append(1.0 if actual_total > line else 0.0)

    print("\n  home side, its closing number moved by the offset:")
    _report(spread_model, spread_real)
    print("\n  over, closing total moved by the offset:")
    _report(total_model, total_real)

    print("\n  half-point steps, model against realised:")
    _steps(spread_model, spread_real, "spread")
    _steps(total_model, total_real, "total")


def _report(model: dict[float, list[float]], real: dict[float, list[float]]) -> None:
    print(f"    {'offset':>7s} {'n':>6s} {'model':>8s} {'realised':>9s} {'diff':>8s} {'t':>6s}")
    for offset, probs in model.items():
        outcomes = real[offset]
        if len(outcomes) < 100:
            continue
        mean_model = float(np.mean(probs))
        mean_real = float(np.mean(outcomes))
        se = float(np.std(outcomes, ddof=1) / np.sqrt(len(outcomes)))
        print(
            f"    {offset:+7.1f} {len(outcomes):6d} {mean_model:8.4f} {mean_real:9.4f}"
            f" {mean_model - mean_real:+8.4f} {(mean_model - mean_real)/se:+6.2f}"
        )


def _steps(
    model: dict[float, list[float]], real: dict[float, list[float]], label: str
) -> None:
    offsets = sorted(model)
    for low, high in zip(offsets[:-1], offsets[1:], strict=True):
        if not model[low] or not model[high]:
            continue
        model_step = float(np.mean(model[high])) - float(np.mean(model[low]))
        real_step = float(np.mean(real[high])) - float(np.mean(real[low]))
        print(
            f"    {label:7s} {low:+.1f} -> {high:+.1f}"
            f"  model {model_step:+.4f}  realised {real_step:+.4f}"
        )


if __name__ == "__main__":
    main()
