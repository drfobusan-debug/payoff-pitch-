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

from mlb_engine.config import load_config
from mlb_engine.output.totals_audit import run_audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", type=Date.fromisoformat, default=Date.today())
    ap.add_argument("--sheet", type=Date.fromisoformat, default=None, help="sheet date to grade (default: day before)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path, text = run_audit(load_config(), args.date, args.sheet)
    print(text)
    if path is None:
        print("no totals ledger yet; nothing written", file=sys.stderr)
    else:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
