"""Fit the wall logistic expected home runs are summed from.

``hr_probability`` turns a batted ball into a fraction of a home run with a
logistic on (projected distance - wall). Both of its parameters were hand-set:
centred on the wall itself, with a 14-foot spread standing in for projection
error and wind. Neither had ever been measured against whether the ball actually
left the park -- and the whole home-run rate the simulator prices rests on this
curve, because :func:`blend_hr_rate` pulls a hitter's HR/PA toward the xHR/PA it
produces at 200 equivalent plate appearances.

    python -m scripts.xhr_wall_fit

Finding, 4,071 measured live balls over 2026 (fit on the first 60% by date,
scored on the rest): **the shipped curve over-counts expected home runs by 42%,
and the error is all in the near misses.**

    projected past the wall      n     actual   shipped   fitted
      40+ feet short           780      0.000     0.014    0.000
      20-40 feet short         305      0.000     0.111    0.001
      10-20 feet short         126      0.040     0.256    0.012
      0-10 feet short           96      0.073     0.405    0.073
      0-10 feet past            80      0.325     0.579    0.323
      10-20 feet past           80      0.738     0.737    0.743
      20+ feet past            176      0.983     0.919    0.982

    held out:  shipped 383 xHR vs 270 actual (1.42x), Brier .0438
               fitted  267 xHR vs 270 actual (0.99x), Brier .0264

The fit moves the curve **+8.6 feet past the wall** and narrows it to a
**5.2-foot** spread. The offset is physical rather than statistical: Statcast's
distance projects where the ball *lands*, and a ball has to clear the wall's
height on the way, so landing on the wall is not a coin flip -- it is an out.

Refit and update ``WALL_OFFSET``/``CARRY_SIGMA`` in ``mlb_engine.features.xhr``
when a season's worth of new tracking has accumulated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mlb_engine.data.fences import fence_for_team, wall_distance
from mlb_engine.features.xhr import (
    CARRY_SIGMA,
    MAX_HR_ANGLE,
    MIN_HR_ANGLE,
    MIN_HR_DISTANCE,
    WALL_OFFSET,
    _fence_lookup,
    spray_angle,
)
from scripts.xk_refit_study import load_pitches

# Where the shipped curve sat before it was fitted, for the comparison.
PRIOR_OFFSET = 0.0
PRIOR_SIGMA = 14.0
GAP_BINS = [-1e9, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 1e9]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def live_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Measured balls in the home-run band, with their gap to the wall."""
    bip = df[df["launch_angle"].notna()]
    live = bip[
        (bip["hit_distance_sc"] >= MIN_HR_DISTANCE)
        & bip["launch_angle"].between(MIN_HR_ANGLE, MAX_HR_ANGLE)
        & bip["hc_x"].notna()
        & bip["hc_y"].notna()
    ].copy()
    fences = _fence_lookup(live["home_team"])
    sprays = spray_angle(live["hc_x"], live["hc_y"]).to_numpy()
    live["wall"] = [
        wall_distance(fences.get(str(t), fence_for_team(None)), float(s))
        for t, s in zip(live["home_team"].to_numpy(), sprays, strict=True)
    ]
    live["gap"] = live["hit_distance_sc"].astype(float) - live["wall"]
    live["is_hr"] = live["events"].eq("home_run").astype(float)
    return live


def nll(params: np.ndarray, gap: np.ndarray, y: np.ndarray) -> float:
    offset, log_sigma = params
    p = np.clip(sigmoid((gap - offset) / np.exp(log_sigma)), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def main() -> None:
    live = live_balls(load_pitches())
    print(f"{len(live):,} measured live balls, {live.is_hr.sum():.0f} home runs, "
          f"{live.game_date.min().date()}..{live.game_date.max().date()}")

    cut = live.game_date.quantile(0.6)
    train, test = live[live.game_date < cut], live[live.game_date >= cut]
    fit = minimize(
        nll,
        np.array([PRIOR_OFFSET, np.log(PRIOR_SIGMA)]),
        args=(train.gap.to_numpy(), train.is_hr.to_numpy()),
    )
    offset, sigma = float(fit.x[0]), float(np.exp(fit.x[1]))
    print(f"\nbefore the fit: offset {PRIOR_OFFSET:+.1f} ft, sigma {PRIOR_SIGMA:.1f}")
    print(f"shipped now   : offset {WALL_OFFSET:+.1f} ft, sigma {CARRY_SIGMA:.1f}")
    print(f"this fit      : offset {offset:+.1f} ft, sigma {sigma:.1f}"
          f"   (train {len(train):,}, from {cut.date()})")

    for label, part in (("train", train), ("test", test)):
        gap, y = part.gap.to_numpy(), part.is_hr.to_numpy()
        print(f"\n{label}: actual HR rate {y.mean():.4f} over {len(part):,} balls")
        for name, (off, sig) in (
            ("hand-set", (PRIOR_OFFSET, PRIOR_SIGMA)),
            ("shipped", (WALL_OFFSET, CARRY_SIGMA)),
            ("this fit", (offset, sigma)),
        ):
            p = sigmoid((gap - off) / sig)
            print(f"  {name:<9} mean {p.mean():.4f}  sum {p.sum():.0f} vs {y.sum():.0f}"
                  f" actual ({p.sum() / max(y.sum(), 1):.2f}x)"
                  f"  Brier {np.mean((p - y) ** 2):.4f}")

    print("\nby projected distance past the wall (held out):")
    t = test.assign(
        hand_set=sigmoid((test.gap.to_numpy() - PRIOR_OFFSET) / PRIOR_SIGMA),
        shipped=sigmoid((test.gap.to_numpy() - WALL_OFFSET) / CARRY_SIGMA),
    )
    table = t.groupby(pd.cut(t.gap, GAP_BINS), observed=True).agg(
        n=("is_hr", "size"),
        actual=("is_hr", "mean"),
        hand_set=("hand_set", "mean"),
        shipped=("shipped", "mean"),
    )
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
