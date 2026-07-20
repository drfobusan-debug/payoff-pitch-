"""Analyze pickled backtest picks: calibration, PPV/NPV, FP/FN bias, workbook."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

from mlb_engine.backtest import (
    confidence_gap,
    confusion,
    risk_factors,
    summarize,
    write_backtest_workbook,
)


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/backtest_graded.pkl")
    with src.open("rb") as f:
        graded = pickle.load(f)
    print(f"Loaded {len(graded)} graded picks from {src}")

    groups = summarize(graded)
    conf = confusion(graded)
    gaps = confidence_gap(graded)
    findings = risk_factors(groups, conf, gaps)

    print("\n== Accuracy / calibration by market group ==")
    for g in groups:
        print(
            f"{g.group:<16} n={g.n:<7} win%={g.win_pct * 100:5.1f} "
            f"avgP={g.avg_model_prob * 100:5.1f} brier={g.brier:.4f} "
            f"calibGap={gaps.get(g.group, 0.0) * 100:+5.1f}pts"
        )

    print("\n== Confusion (PPV / NPV / FP-rate / FN-rate) ==")
    for c in conf:
        print(
            f"{c.group:<16} PPV={c.ppv:.3f} NPV={c.npv:.3f} "
            f"sens={c.sensitivity:.3f} spec={c.specificity:.3f} "
            f"FPr={c.fp_rate:.3f} FNr={c.fn_rate:.3f} (n={c.n})"
        )

    print("\n== Risk factors ==")
    for line in findings:
        print(" - " + line)

    out = Path.home() / ".mlb_engine" / "output" / "backtest_2024.xlsx"
    write_backtest_workbook(groups, conf, gaps, findings, out)
    print(f"\nWorkbook -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
