"""Is a prop market paying because the model is right, or because it got lucky?

For batter hits (H), batter walks (BB), a starter's walks (SP BB) and a starter's
outs (SP outs) -- and every other market beside them for comparison -- this
script scores the model's probability, the printed probability and the two-sided
no-vig market against the outcome, bootstraps the model-minus-market gap, and
cuts the record by side, price, tier, edge and line so accuracy can be read apart
from selection and price. See :mod:`mlb_engine.audit.market_isolation`.

Two ledgers can feed it:

* ``--source power`` (default): the power screen's ``power_screen_ledger.csv``,
  graded here off the box scores through ``fetch_result`` (cached under the
  audit dir), one run per day unless ``--all-runs``.
* ``--source engine``: the main audit ledger ``ledger.csv``, which already
  carries a result and a P&L per row; only priced prop rows are read.

Both are read from the ``engine-state`` branch by default (``--ref``), so the
script needs no local state and never writes any. ``--ledger`` reads a file
instead. Output is a console report plus a Markdown and a CSV file in the audit
dir. Nothing here changes a price, a tier, a gate or the screen.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from datetime import date as Date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlb_engine.audit.market_isolation import (  # noqa: E402
    FOCUS_MARKETS,
    MarketReport,
    Row,
    build_report,
    last_run_per_day,
    markets_present,
    render_console,
    render_markdown,
    rows_from_engine_ledger,
    rows_from_power,
    summary_rows,
    write_csv,
)
from mlb_engine.audit.power_ledger import grade_positions, load  # noqa: E402
from mlb_engine.config import load_config  # noqa: E402
from mlb_engine.data.results import GameResult, fetch_result  # noqa: E402
from mlb_engine.output.power_board import DISPLAY_ONLY  # noqa: E402

log = logging.getLogger("market_accuracy_isolation")

STATE_PATHS = {"power": "mlb/power_screen_ledger.csv", "engine": "mlb/ledger.csv"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", type=Date.fromisoformat, default=None, help="first ledger day")
    p.add_argument("--end", type=Date.fromisoformat, default=None, help="last ledger day")
    p.add_argument("--source", choices=("power", "engine"), default="power")
    p.add_argument("--ledger", default=None, help="read this ledger file instead of the branch")
    p.add_argument(
        "--ref",
        default="origin/engine-state",
        help="git ref the ledger is read from when --ledger is not given",
    )
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--all-runs", action="store_true", help="power: grade every run of a day")
    p.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default=None, help="where the md/csv land (default: audit dir)")
    p.add_argument("--markets", nargs="*", default=None, help="restrict the detailed sections")
    p.add_argument("--summary-only", action="store_true", help="print the tables only")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _ledger_text(args: argparse.Namespace) -> str:
    if args.ledger:
        return Path(args.ledger).read_text(encoding="utf-8")
    spec = f"{args.ref}:{STATE_PATHS[args.source]}"
    out = subprocess.run(
        ["git", "-C", args.repo, "show", spec], capture_output=True, text=True, check=True
    )
    return out.stdout


def _results(game_pks: set[int], cache_dir: Path) -> dict[int, GameResult]:
    out: dict[int, GameResult] = {}
    for pk in sorted(game_pks):
        try:
            out[pk] = fetch_result(pk, cache_dir=cache_dir)
        except Exception as exc:  # noqa: BLE001 - one missing box score voids one game
            log.warning("could not fetch the box score for %s: %s", pk, exc)
    return out


def _power_rows(text: str, args: argparse.Namespace, cache_dir: Path) -> list[Row]:
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "power_screen_ledger.csv"
        ledger.write_text(text, encoding="utf-8")
        positions = load(ledger)
    positions = [
        p
        for p in positions
        if p.stat not in DISPLAY_ONLY and p.date and _within(p.date, args.start, args.end)
    ]
    if positions and not args.all_runs:
        positions = last_run_per_day(positions)
    results = _results({p.game_pk for p in positions if p.game_pk is not None}, cache_dir)
    graded, voided = grade_positions(positions, results)
    print(f"power screen ledger: {len(positions)} positions, {len(graded)} graded, {voided} voided")
    return rows_from_power(graded)


def _within(day: str, start: Date | None, end: Date | None) -> bool:
    d = Date.fromisoformat(day)
    return (start is None or d >= start) and (end is None or d <= end)


def _print_summary(reports: list[MarketReport]) -> None:
    cols = (
        ("market", 9),
        ("n", 5),
        ("hit", 7),
        ("units", 8),
        ("roi", 8),
        ("scored", 6),
        ("brier_model", 11),
        ("brier_printed", 13),
        ("brier_market", 12),
        ("gap", 8),
        ("ci_lo", 8),
        ("ci_hi", 8),
        ("p_model_better", 6),
        ("verdict", 28),
    )
    rows = summary_rows(reports)
    print("  ".join(f"{c:<{w}}" for c, w in cols))
    for r in rows:
        print("  ".join(f"{r[c]:<{w}}" for c, w in cols))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    cfg = load_config()
    text = _ledger_text(args)
    if args.source == "power":
        rows = _power_rows(text, args, cfg.cache_dir)
    else:
        rows = rows_from_engine_ledger(text.splitlines(), args.start, args.end)
        print(f"engine ledger: {len(rows)} priced, graded prop rows")
    if not rows:
        print("nothing to score")
        return 1
    days = sorted({r.date for r in rows})
    print(f"{days[0]} .. {days[-1]}, {len(days)} days\n")

    markets = markets_present(rows)
    reports = [build_report(m, rows, resamples=args.resamples, seed=args.seed) for m in markets]

    print("=== four-market summary ===")
    _print_summary([r for r in reports if r.market in FOCUS_MARKETS])
    print("\n=== every market ===")
    _print_summary(reports)
    print()
    for rep in reports:
        if rep.market in FOCUS_MARKETS:
            print(f"{rep.market:<9} {rep.verdict}" + (" [thin: n < 100]" if rep.thin else ""))

    if not args.summary_only:
        wanted = set(args.markets) if args.markets else set(markets)
        for rep in reports:
            if rep.market in wanted:
                print()
                print(render_console(rep))

    out_dir = Path(args.out_dir) if args.out_dir else cfg.audit_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"market_accuracy_{args.source}_{days[0]}_{days[-1]}"
    title = f"Market accuracy isolation ({args.source} ledger, {days[0]} .. {days[-1]})"
    md = out_dir / f"{stem}.md"
    md.write_text(render_markdown(reports, title), encoding="utf-8")
    csv_path = out_dir / f"{stem}.csv"
    write_csv(reports, csv_path)
    print(f"\nwrote {md}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
