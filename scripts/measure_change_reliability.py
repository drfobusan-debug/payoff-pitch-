"""Is a hitter's *move* in max EV or fastball whiff% a move, or is it the sample?

``measure_metric_reliability.py`` sizes the window a *level* needs. A change
needs two windows, and two noisy reads differenced are noisier than either, so a
report that prints "max EV up 3 mph" has to be able to say how much of that 3 mph
a hitter who changed nothing would show anyway. Three numbers per block size:

* **noise sd** -- the same block of plate appearances split at random into two
  halves and differenced. Both halves come from the same weeks, so this is what
  a hitter who did not change looks like.
* **adjacent sd** -- the recent block minus the block immediately before it, in
  time. Noise plus whatever really moved.
* **band** -- ``1.96 x noise sd``, the move a level read has to clear before it
  is distinguishable from its own measurement error at all.

Then the only question that matters for a report: does the move say anything
about what comes next? The next block of the same size is held out and regressed
on the trailing level and the move together, so the move is credited only with
what the level does not already carry.

    python scripts/measure_change_reliability.py --days 182

Re-run it when a season has accumulated; the numbers
``mlb_engine/features/power_change.py`` stores are one season and should move.
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.statcast import StatcastRepository
from scripts.measure_metric_reliability import PA_KEY, _pa_frame, _rate

log = logging.getLogger("change")

BLOCKS = (25, 40, 60, 90, 130)
METRICS = ("max_ev", "fb_whiff")


def _ordered(pa: pd.DataFrame) -> pd.DataFrame:
    """Plate appearances in the order they were taken."""
    out = pa.copy()
    out["_d"] = pd.to_datetime(out["game_date"])
    return out.sort_values(["batter", "_d", "inning"], kind="stable")


def _blocks(
    pa: pd.DataFrame, metric: str, n: int, rng: np.random.Generator
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Per hitter: prior level, recent level, next-block level, and a noise delta."""
    prior: list[float] = []
    recent: list[float] = []
    nxt: list[float] = []
    noise: list[float] = []
    for _, g in pa.groupby("batter", sort=False):
        if len(g) < 3 * n:
            continue
        tail = g.iloc[-3 * n :]
        a, b, c = tail.iloc[:n], tail.iloc[n : 2 * n], tail.iloc[2 * n :]
        va, vb, vc = _rate(a, metric), _rate(b, metric), _rate(c, metric)
        if not any(math.isnan(v) for v in (va, vb, vc)):
            prior.append(va)
            recent.append(vb)
            nxt.append(vc)
        both = pd.concat([a, b])
        idx = rng.permutation(len(both))
        h1 = _rate(both.iloc[np.sort(idx[:n])], metric)
        h2 = _rate(both.iloc[np.sort(idx[n:])], metric)
        if not (math.isnan(h1) or math.isnan(h2)):
            noise.append(h1 - h2)
    return prior, recent, nxt, noise


def _ols_t(y: list[float], x1: list[float], x2: list[float]) -> tuple[float, float, int]:
    """t on the level and on the move, predicting the held-out next block."""
    n = len(y)
    if n < 30:
        return math.nan, math.nan, n
    x = np.column_stack([np.ones(n), np.asarray(x1), np.asarray(x2)])
    yv = np.asarray(y)
    beta, *_ = np.linalg.lstsq(x, yv, rcond=None)
    resid = yv - x @ beta
    dof = n - x.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    return float(beta[1] / se[1]), float(beta[2] / se[2]), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", type=Date.fromisoformat, default=Date.today())
    ap.add_argument("--days", type=int, default=182)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config()
    raw = StatcastRepository(cfg.cache_dir).load_trailing(args.date, args.days)
    pa = _ordered(_pa_frame(raw))
    log.info("plate appearances: %d over %d hitters", len(pa), pa["batter"].nunique())
    log.info("keyed on %s", ", ".join(PA_KEY))

    rng = np.random.default_rng(args.seed)
    for metric in METRICS:
        print(f"\n{metric}")
        print(f"{'block':>6}{'hitters':>9}{'noise sd':>10}{'adj sd':>9}"
              f"{'band':>8}{'true sd':>9}{'t level':>9}{'t move':>8}")
        for n in BLOCKS:
            prior, recent, nxt, noise = _blocks(pa, metric, n, rng)
            if len(prior) < 30:
                print(f"{n:>6}{len(prior):>9}   too few hitters")
                continue
            noise_sd = float(np.std(noise, ddof=1) / math.sqrt(2))
            adj = np.asarray(recent) - np.asarray(prior)
            adj_sd = float(np.std(adj, ddof=1) / math.sqrt(2))
            true_sd = math.sqrt(max(0.0, adj_sd**2 - noise_sd**2))
            level = [(p + r) / 2 for p, r in zip(prior, recent, strict=True)]
            t_level, t_move, n_h = _ols_t(nxt, level, list(adj))
            print(f"{n:>6}{n_h:>9}{noise_sd:>10.3f}{adj_sd:>9.3f}"
                  f"{1.96 * noise_sd * math.sqrt(2):>8.2f}{true_sd:>9.3f}"
                  f"{t_level:>9.1f}{t_move:>8.1f}")


if __name__ == "__main__":
    main()
