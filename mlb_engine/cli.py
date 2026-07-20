"""Command-line entrypoint: `mlb-engine run` and `mlb-engine audit`."""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from mlb_engine.audit.grade import grade
from mlb_engine.audit.ledger import (
    daily_rollup,
    entries_from_graded,
    overall_metrics,
    update_ledger,
)
from mlb_engine.audit.scorecard import append_scorecard, build_scorecard
from mlb_engine.config import load_config
from mlb_engine.data.fangraphs import FanGraphsClient
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.data.results import fetch_result
from mlb_engine.data.rotowire import RotowireClient
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import VSINClient
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market.tiers import Tier
from mlb_engine.output.excel import write_ledger_workbook, write_workbook
from mlb_engine.pipeline import Pipeline, PipelineDeps
from mlb_engine.recommendations import load_json, save_json


def _parse_date(s: str | None, default: Date) -> Date:
    return Date.fromisoformat(s) if s else default


def _write_quotes_template(recs, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matchup", "market", "selection", "book", "american", "handle_pct", "bets_pct"])
        for r in recs:
            key = (r.matchup, r.market, r.selection)
            if key in seen:
                continue
            seen.add(key)
            w.writerow([r.matchup, r.market, r.selection, "draftkings", "", "", ""])


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    deps = PipelineDeps(
        stats=MLBStatsClient(),
        statcast=StatcastRepository(cfg.cache_dir),
        weather=WeatherProvider(),
        vsin=VSINClient(cfg.creds),
        oddsapi=OddsAPIClient(cfg.creds.odds_api_key),
        rotowire=RotowireClient(cfg.creds),
        fangraphs=FanGraphsClient(cfg.creds),
    )
    if args.sims:
        import os

        os.environ["MLBE_MC_SIMS"] = str(args.sims)
        cfg = load_config()

    slate_date = _parse_date(args.date, Date.today())
    pipe = Pipeline(cfg, deps)
    vsin_csv = Path(args.vsin_csv) if args.vsin_csv else None
    fg_dir = cfg.fangraphs_dir
    has_drop_ins = fg_dir.is_dir() and any(
        f.suffix.lower() in (".csv", ".xlsx", ".xls") for f in fg_dir.iterdir()
    )
    if args.fangraphs_csv:
        fangraphs_csv: Path | None = Path(args.fangraphs_csv)
    elif has_drop_ins:
        fangraphs_csv = fg_dir
    else:
        fangraphs_csv = None
    recs = pipe.run(slate_date, vsin_csv=vsin_csv, fangraphs_csv=fangraphs_csv)

    xlsx = args.out or str(cfg.output_dir / f"mlb_recommendations_{slate_date.isoformat()}.xlsx")
    write_workbook(recs, Path(xlsx), slate_date)
    save_json(recs, cfg.audit_dir / f"predictions_{slate_date.isoformat()}.json")

    # Emit a blank VSIN quotes template so odds/handle can be filled and re-run.
    if not vsin_csv:
        _write_quotes_template(
            recs, cfg.output_dir / f"vsin_template_{slate_date.isoformat()}.csv"
        )

    strong = sum(1 for r in recs if r.tier == Tier.STRONG)
    mod = sum(1 for r in recs if r.tier == Tier.MODERATE)
    print(f"Priced {len(recs)} markets: {strong} strong buys, {mod} moderate buys")
    print(f"Excel: {xlsx}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_config()
    audit_date = _parse_date(args.date, Date.today() - timedelta(days=1))
    pred_path = cfg.audit_dir / f"predictions_{audit_date.isoformat()}.json"
    if not pred_path.exists():
        print(f"No predictions found for {audit_date} at {pred_path}")
        return 1

    recs = load_json(pred_path)
    game_pks = {r.game_pk for r in recs}
    results = {}
    for pk in game_pks:
        try:
            results[pk] = fetch_result(pk)
        except Exception as exc:  # noqa: BLE001
            logging.warning("could not fetch result for %s: %s", pk, exc)

    graded = []
    for r in recs:
        res = results.get(r.game_pk)
        if res is None or not res.final:
            continue
        g = grade(r, res)
        if g is not None:
            graded.append((r, g))

    rows = build_scorecard(graded, audit_date)
    append_scorecard(rows, cfg.audit_dir / "scorecard.csv")

    entries = entries_from_graded(graded, audit_date)
    all_entries = update_ledger(cfg.audit_dir / "ledger.csv", entries, audit_date)
    overall = overall_metrics(all_entries)
    ledger_xlsx = cfg.output_dir / "ledger.xlsx"
    write_ledger_workbook(all_entries, overall, daily_rollup(all_entries), ledger_xlsx)

    print(f"Graded {len(graded)} markets for {audit_date}")
    for row in rows:
        print(
            f"  {row.tier:<12} n={row.n:<4} PPV={row.ppv:.3f} "
            f"sens={row.sensitivity:.3f} spec={row.specificity:.3f} "
            f"NPV={row.npv:.3f} ROI={row.roi:+.3f}"
        )
    print(f"\nLedger: {len(all_entries)} graded bets across all dates -> {ledger_xlsx}")
    for m in overall:
        print(
            f"  OVERALL {m.tier:<12} n={m.n:<5} win%={m.win_pct * 100:5.1f} "
            f"PPV={m.ppv:.3f} sens={m.sensitivity:.3f} spec={m.specificity:.3f} "
            f"NPV={m.npv:.3f} ROI={m.roi * 100:+.1f}% units={m.units:+.1f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="mlb-engine", description="MLB prediction engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run daily predictions -> Excel")
    r.add_argument("--date", help="slate date YYYY-MM-DD (default: today)")
    r.add_argument("--vsin-csv", help="path to VSIN odds/handle CSV")
    r.add_argument(
        "--fangraphs-csv",
        help="FanGraphs custom-report CSV/XLSX (or a folder of them) for "
        "SIERA/Stuff+/wRC+/xSLG tails; defaults to ~/.mlb_engine/fangraphs/ if present",
    )
    r.add_argument("--sims", type=int, help="Monte Carlo sims per game")
    r.add_argument("--out", help="output .xlsx path")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("audit", help="grade a prior slate and update scorecard")
    a.add_argument("--date", help="slate date to audit YYYY-MM-DD (default: yesterday)")
    a.set_defaults(func=cmd_audit)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
