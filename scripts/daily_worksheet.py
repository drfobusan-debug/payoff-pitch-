"""Write the daily MLB worksheet, record its prices, and grade what is pending.

Writes ``worksheet_<day>.xlsx`` (matchups + ranked gaps, audit by gap band, the
ledger, and the bullpen / offense / starter tables) and ``worksheet_<day>.txt``
(the lines the morning email prints) to the output dir, and appends the day's
games to ``audit/worksheet_ledger.csv``.

Usage:
    python -m scripts.daily_worksheet                 # today
    python -m scripts.daily_worksheet 2026-09-14
    python -m scripts.daily_worksheet 2026-09-14 --if-incomplete   # rewrite only a sheet missing a starter, before first pitch
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as Date

from mlb_engine.config import load_config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.output.daily_worksheet import (
    day_incomplete,
    ledger_path,
    load_ledger,
    run_worksheet,
)
from mlb_engine.output.totals_sheet import slate_started


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", type=Date.fromisoformat, default=Date.today())
    ap.add_argument(
        "--if-incomplete", action="store_true",
        help="keep a written sheet unless one of its games lacks a starter score and no game has started",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    slate = MLBStatsClient().get_slate(args.date)
    if not slate.games:
        print(f"no games on {args.date}; nothing written")
        return 0
    if args.if_incomplete and (cfg.output_dir / f"worksheet_{args.date.isoformat()}.xlsx").exists():
        if not day_incomplete(load_ledger(ledger_path(cfg)), args.date):
            print(f"worksheet {args.date}: every game scored; kept")
            return 0
        if slate_started(slate):
            print(f"worksheet {args.date}: a starter is missing but the slate has started; kept")
            return 0
    path, text = run_worksheet(cfg, args.date, slate)
    print(text)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
