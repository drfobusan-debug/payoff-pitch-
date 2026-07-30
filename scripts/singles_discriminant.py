"""Discriminant profile for batter singles (batter_1b).

Splits the graded singles picks into the two groups requested and asks *which
engine metric actually separates them* — a real statistic, not a vibe.

Groups (confusion-matrix framing: favored = model_prob >= 0.5, positive = the
selection won, pushes dropped):

  * WINS  = TP + FN  (favored-won + faded-won)   -> outcome the bet won
  * LOSSES = FP + TN (favored-lost + faded-lost) -> outcome the bet lost

For every numeric engine metric it prints each group's profile (n, mean,
median, sd) and three significance tests: Welch t-test, Mann-Whitney U, and
point-biserial correlation with the win/loss outcome. It also runs the two
within-side splits the audit uses: TP vs FP among favored picks (why buys win
or lose) and FN vs TN among faded picks (why fades hit or get away).

Reads the cumulative metric store the nightly audit maintains
(``~/.mlb_engine/audit/graded_metrics.csv``). Run on the machine that owns it:

    python scripts/singles_discriminant.py
    python scripts/singles_discriminant.py --market batter_1b --store /path/to/graded_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr, ttest_ind

from mlb_engine.config import load_config
from mlb_engine.output.audit_insight import (
    METRIC_COLS,
    METRIC_LABELS,
    STORE_NAME,
    classify,
    market_label,
)

MIN_GROUP = 6  # per-group sample needed before a metric is tested


def _profile(sub: pd.DataFrame, label_a: str, label_b: str, y: np.ndarray) -> None:
    """y: 1 = group B (e.g. wins), 0 = group A (e.g. losses)."""
    nb, na = int(y.sum()), int(len(y) - y.sum())
    print(f"\n  {label_b}: n={nb}   {label_a}: n={na}")
    if nb < MIN_GROUP or na < MIN_GROUP:
        print(f"  (need >= {MIN_GROUP} per group to test — skipping)")
        return
    header = (
        f"  {'metric':<28} {label_a[:10]:>11} {label_b[:10]:>11} "
        f"{'d(B-A)':>9} {'t p':>8} {'MW p':>8} {'r_pb':>7} {'sig':>4}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for col in METRIC_COLS:
        if col not in sub.columns:
            continue
        x = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(x)
        xv, yv = x[m], y[m]
        if int(yv.sum()) < MIN_GROUP or int(len(yv) - yv.sum()) < MIN_GROUP:
            continue
        xa, xb = xv[yv == 0], xv[yv == 1]
        if np.nanstd(xv) < 1e-9:
            continue
        try:
            _, tp = ttest_ind(xb, xa, equal_var=False)
            _, mw = mannwhitneyu(xb, xa, alternative="two-sided")
            r, rp = pointbiserialr(yv, xv)
        except Exception:  # noqa: BLE001
            continue
        rows.append((min(tp, mw, rp), col, xa.mean(), xb.mean(), rp, tp, mw, r))
    rows.sort(key=lambda t: t[0])
    for _, col, ma, mb, rp, tp, mw, r in rows:
        sig = "***" if min(tp, mw, rp) < 0.05 else ""
        lab = METRIC_LABELS.get(col, col)[:28]
        print(
            f"  {lab:<28} {ma:>11.4f} {mb:>11.4f} {mb - ma:>9.4f} "
            f"{tp:>8.3f} {mw:>8.3f} {r:>7.3f} {sig:>4}"
        )
    if not rows:
        print("  (no metric had enough non-null signal in both groups)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="batter_1b")
    ap.add_argument("--store", type=Path, default=None)
    args = ap.parse_args()

    store = args.store or (load_config().audit_dir / STORE_NAME)
    if not store.exists():
        raise SystemExit(
            f"No metric store at {store}. It is written by `mlb-engine audit` "
            "(audit insight); run an audit first."
        )

    df = classify(pd.read_csv(store))
    sub = df[df["market"] == args.market].copy()
    if sub.empty:
        raise SystemExit(f"No graded rows for market {args.market} in {store}")

    tp = int(((sub["favored"] == 1) & (sub["won"] == 1)).sum())
    fp = int(((sub["favored"] == 1) & (sub["won"] == 0)).sum())
    fn = int(((sub["favored"] == 0) & (sub["won"] == 1)).sum())
    tn = int(((sub["favored"] == 0) & (sub["won"] == 0)).sum())
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    print(f"Store: {store}")
    print(f"Market: {args.market} ({market_label(args.market)})   graded n={len(sub)}")
    print("\nConfusion matrix (favored = model_prob >= 0.5, positive = won):")
    print(f"  TP (favored & won)  = {tp:>4}    FP (favored & lost) = {fp:>4}   PPV = {ppv:.3f}")
    print(f"  FN (faded & won)    = {fn:>4}    TN (faded & lost)   = {tn:>4}   NPV = {npv:.3f}")

    print("\n=== GROUP A (LOSSES: FP+TN) vs GROUP B (WINS: TP+FN) — the two groups ===")
    _profile(sub, "LOSSES", "WINS", sub["won"].to_numpy())

    print("\n=== Within FAVORED buys — TP (won) vs FP (lost): why a buy wins or loses ===")
    fav = sub[sub["favored"] == 1]
    _profile(fav, "FP lost", "TP won", fav["won"].to_numpy())

    print("\n=== Within FADED picks — TN (lost) vs FN (won): why a fade hits or gets away ===")
    fade = sub[sub["favored"] == 0]
    _profile(fade, "TN lost", "FN won", fade["won"].to_numpy())

    print(
        "\nd(B-A) = mean(group B) - mean(group A).  t p = Welch t-test.  "
        "MW p = Mann-Whitney U.  r_pb = point-biserial corr with win(1)/loss(0).\n"
        "*** = significant at p < 0.05 on all three tests."
    )


if __name__ == "__main__":
    main()
