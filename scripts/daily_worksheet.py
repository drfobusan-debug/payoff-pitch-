"""Write the daily MLB worksheet, record its prices, and grade what is pending.

Writes ``worksheet_<day>.xlsx`` (matchups + ranked gaps, audit by gap band, the
ledger, and the bullpen / offense / starter tables) and ``worksheet_<day>.txt``
(the lines the morning email prints) to the output dir, and appends the day's
games to ``audit/worksheet_ledger.csv``.

Usage:
    python -m scripts.daily_worksheet                 # today
    python -m scripts.daily_worksheet 2026-09-14
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as Date

from mlb_engine.config import load_config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.output.daily_worksheet import run_worksheet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", type=Date.fromisoformat, default=Date.today())
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    slate = MLBStatsClient().get_slate(args.date)
    if not slate.games:
        print(f"no games on {args.date}; nothing written")
        return 0
    path, text = run_worksheet(cfg, args.date, slate)
    print(text)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
