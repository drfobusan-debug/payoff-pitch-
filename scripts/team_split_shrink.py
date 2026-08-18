"""Fit the equivalent-PA prior strength for each team split.

For a club's split measured over ``n`` plate appearances, the observed spread
across the thirty clubs is talent plus sampling noise:

    var(observed) = var(talent) + sigma_pa^2 / n

so the shrinkage that minimises squared error is ``n / (n + k)`` toward the
baseline, with ``k = sigma_pa^2 / var(talent)`` in plate appearances. Both terms
are measured here off the same Statcast frame the previews read: ``sigma_pa`` is
the spread of per-PA xwOBA within the league, and ``var(talent)`` is the spread
across clubs with the noise subtracted.

Run against the widest window available, because subtracting noise from a spread
that is nearly all noise is exactly the estimate that needs the most sample:

    python -m scripts.team_split_shrink <statcast.pkl> [--as-of YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import math
from datetime import date as Date
from pathlib import Path

import pandas as pd

from mlb_engine.features.team_splits import (
    SPLITS,
    XWOBA_COL,
    _priced_pa,
    _window,
)


def _split_rows(d: pd.DataFrame, key: str) -> pd.DataFrame:
    return d[SPLITS[key](d)] if key in SPLITS else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", type=Path)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--days", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_pickle(args.frame)
    as_of = Date.fromisoformat(args.as_of) if args.as_of else Date.today()
    d = _priced_pa(_window(df, as_of, args.days))
    sigma = float(d[XWOBA_COL].astype(float).std())
    print(f"{len(d):,} priced PA, per-PA xwOBA sd = {sigma:.4f}\n")
    print(f"{'split':<8}{'clubs':>6}{'median PA':>11}{'observed sd':>13}{'noise sd':>10}{'talent sd':>11}{'k':>8}")

    for key in ("all", *SPLITS):
        rows = _split_rows(d, key)
        g = rows.groupby("bat_team")[XWOBA_COL].agg(["mean", "size"])
        vals = [float(v) for v in g["mean"]]
        sizes = sorted(int(s) for s in g["size"])
        n = len(vals)
        med = sizes[n // 2]
        mean = sum(vals) / n
        obs = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        noise = sigma / math.sqrt(med)
        talent = math.sqrt(max(obs**2 - noise**2, 0.0))
        k = sigma**2 / talent**2 if talent else float("inf")
        shown = f"{k:.0f}" if math.isfinite(k) else "inf"
        print(f"{key:<8}{n:>6}{med:>11}{obs:>13.4f}{noise:>10.4f}{talent:>11.4f}{shown:>8}")


if __name__ == "__main__":
    main()
