"""Driver: run the 2024 accuracy/calibration backtest and pickle graded picks."""

from __future__ import annotations

import logging
import pickle
import sys
import time
from datetime import date
from pathlib import Path

from mlb_engine.backtest import load_season_frame, run_backtest, sample_dates, summarize
from mlb_engine.config import load_config


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("mlb_engine.backtest").setLevel(logging.INFO)
    cfg = load_config()
    frame = load_season_frame(cfg.cache_dir, date(2024, 4, 15), date(2024, 9, 30))
    dates = sample_dates(date(2024, 6, 1), date(2024, 9, 28), 2)
    print(f"Backtesting {len(dates)} dates ({dates[0]}..{dates[-1]}), {len(frame)} statcast rows")
    t = time.time()
    graded = run_backtest(cfg, frame, dates)
    print(f"Done in {(time.time() - t) / 60:.1f} min -> {len(graded)} graded picks")
    out = Path("/tmp/backtest_graded.pkl")
    with out.open("wb") as f:
        pickle.dump(graded, f)
    print(f"Pickled -> {out}")
    for r in summarize(graded):
        print(
            f"{r.group:<16} n={r.n:<7} win%={r.win_pct * 100:5.1f} "
            f"avgP={r.avg_model_prob * 100:5.1f} brier={r.brier:.4f} pushes={r.pushes}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
