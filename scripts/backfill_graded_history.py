"""Backfill a graded pick history over a date range, sharded across processes.

The live ledger only holds the slates the engine actually ran, which makes every
*game-level* market (moneyline, F5, run line, totals) sample-starved: one outcome
per game, so five slates is ~50 games. `mlb_engine.backtest` can replay any
historical date off cached Statcast and grade against the final score, which
turns a season into ~1,200 games.

Runs with no odds feed, so rows carry model probabilities and outcomes but no
prices, EV tiers or handle splits. Use it for "does this metric predict the
outcome", not for "was this a good price".

    # one shard per core, then concatenate the CSVs
    for s in 0 1 2 3 4 5; do
        python scripts/backfill_graded_history.py 2026-04-20 2026-07-22 $s 6 &
    done

Each shard appends to ~/.mlb_engine/audit/backfill_shard<N>.csv and skips dates
already present, so it is safe to re-run after an interruption.
"""

from __future__ import annotations

import csv
import logging
import sys
from datetime import date, timedelta

from mlb_engine.backtest import load_season_frame, run_backtest
from mlb_engine.config import load_config

HEADER = ["date", "game_pk", "matchup", "market", "selection", "line", "model_prob",
          "tier", "result"]
# trailing-window lead-in: the batter/pitcher windows look back up to 6 weeks
LEADIN_DAYS = 45


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    shard = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nshards = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config()
    frame = load_season_frame(cfg.cache_dir, start - timedelta(days=LEADIN_DAYS), end)

    out = cfg.audit_dir / f"backfill_shard{shard}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open() as fh:
            done = {r["date"] for r in csv.DictReader(fh)}

    span = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    dates = [d for i, d in enumerate(span)
             if i % nshards == shard and d.isoformat() not in done]
    print(f"shard {shard}/{nshards}: {len(dates)} dates to grade -> {out}", flush=True)

    fresh = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(HEADER)
        for d in dates:
            for rec, result in run_backtest(cfg, frame, [d]):
                w.writerow([d.isoformat(), rec.game_pk, rec.matchup, rec.market,
                            rec.selection, rec.line, f"{rec.model_prob:.6f}",
                            rec.tier, result])
            fh.flush()
            print(f"shard {shard} {d}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
