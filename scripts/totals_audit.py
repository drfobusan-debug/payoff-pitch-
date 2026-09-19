"""Grade yesterday's hand totals sheet and refresh the running ledger.

Writes ``totals_audit_<day>.xlsx`` (yesterday graded, cumulative summary, full
ledger) and ``totals_audit_<day>.txt`` (the lines the morning email prints).

Usage:
    python -m scripts.totals_audit                       # grade yesterday, file under today
    python -m scripts.totals_audit 2026-09-10            # grade 2026-09-09, file under 09-10
    python -m scripts.totals_audit 2026-09-10 --sheet 2026-09-08
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as Date
from datetime import timedelta

from mlb_engine.config import load_config
from mlb_engine.output.totals_audit import run_audit
from mlb_engine.state import auto_push


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", type=Date.fromisoformat, default=Date.today())
    ap.add_argument("--sheet", type=Date.fromisoformat, default=None, help="sheet date to grade (default: day before)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    path, text = run_audit(cfg, args.date, args.sheet)
    print(text)
    if path is None:
        print("no totals ledger yet; nothing written", file=sys.stderr)
        return 0
    print(path)
    # The grade lands on the shared branch now rather than with whichever
    # capture next happens to push, so the record is readable the same morning.
    if cfg.state_sync:
        sheet_day = args.sheet or (args.date - timedelta(days=1))
        report = auto_push(cfg.data_dir, f"totals audit {sheet_day.isoformat()} graded", branch=cfg.state_branch)
        if report is not None:
            print(f"State: {report.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
