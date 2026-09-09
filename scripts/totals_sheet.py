"""Write the day's hand-method totals sheet as a workbook.

    python -m scripts.totals_sheet                # today
    python -m scripts.totals_sheet 2026-09-10

Writes ``totals_sheet_<date>.xlsx`` to the engine's output directory, where the
morning package (``scripts.email_daily_package --with-daily``) picks it up. Every
rule the sheet scores is on its Legend tab; the bands themselves live in
``mlb_engine/output/totals_sheet.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as Date

from mlb_engine.config import load_config
from mlb_engine.output.totals_sheet import build_totals_sheet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", nargs="?", type=Date.fromisoformat, default=Date.today())
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path = build_totals_sheet(load_config(), args.date)
    if path is None:
        print(f"no games on {args.date}; no totals sheet written", file=sys.stderr)
        return 0
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
