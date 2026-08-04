"""Console entry point for the college-football engine.

Commands mirror the MLB engine:

    cfb-engine run       price today's slate -> Excel + article/PDF + MP3 (+email)
    cfb-engine card      rebuild the article/PDF/MP3 from saved predictions
    cfb-engine close     snapshot the closing market for closing-line value
    cfb-engine audit     grade a past slate, update the ledger, email the recap
    cfb-engine report    rebuild the ledger workbook from history (no grading)
    cfb-engine calibrate refit the probability calibration from the ledger
    cfb-engine backtest  A/B the score engines (normal vs markov) on a season
    cfb-engine scorecard print the rolling PPV/NPV-by-market scorecard
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date as Date
from datetime import datetime
from zoneinfo import ZoneInfo

from cfb_engine.audit.clv import (
    closing_quotes,
    clv_summary,
    compute_clv,
    load_closing,
    save_closing,
)
from cfb_engine.audit.grade import build_result_index, grade, result_for
from cfb_engine.audit.ledger import (
    LedgerEntry,
    daily_rollup,
    engine_metrics,
    entries_from_graded,
    load_ledger,
    market_metrics,
    overall_metrics,
    update_ledger,
)
from cfb_engine.audit.scorecard import append_scorecard, build_scorecard
from cfb_engine.config import Config, load_config
from cfb_engine.data.cfbd import CFBDClient
from cfb_engine.data.oddsapi import OddsAPIClient
from cfb_engine.output.audit_report import generate_audit_report
from cfb_engine.output.card import generate_daily_card
from cfb_engine.output.excel import write_ledger_workbook, write_workbook
from cfb_engine.pipeline import Pipeline
from cfb_engine.recommendations import Recommendation, load_json, save_json

_EASTERN = ZoneInfo("America/New_York")


def _today() -> Date:
    return datetime.now(_EASTERN).date()


def _day(args: argparse.Namespace) -> Date:
    if args.date:
        return Date.fromisoformat(args.date)
    return _today()


def _season(cfg: Config, day: Date) -> int:
    if cfg.season:
        return cfg.season
    return day.year - 1 if day.month <= 2 else day.year


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    day = _day(args)
    pipe = Pipeline(cfg)
    recs = pipe.run(day)
    if not recs:
        print(f"No NCAAF recommendations for {day}.")
        return 0

    save_json(recs, cfg.predictions_file(day))
    xlsx = write_workbook(recs, cfg.output_dir / f"PayoffPitch_CFB_{day.isoformat()}.xlsx", day)
    print(f"Wrote {len(recs)} recommendations -> {xlsx}")

    extra = [(xlsx.name, xlsx.read_bytes())]
    generate_daily_card(
        recs, day, cfg, email=not args.no_email, to=args.to, extra_attachments=extra
    )
    return 0


def cmd_card(cfg: Config, args: argparse.Namespace) -> int:
    day = _day(args)
    path = cfg.predictions_file(day)
    if not path.exists():
        print(f"No saved predictions for {day} ({path}); run `cfb-engine run` first.")
        return 1
    recs = load_json(path)
    xlsx = cfg.output_dir / f"PayoffPitch_CFB_{day.isoformat()}.xlsx"
    extra = [(xlsx.name, xlsx.read_bytes())] if xlsx.exists() else None
    generate_daily_card(
        recs, day, cfg, email=not args.no_email, to=args.to, extra_attachments=extra
    )
    return 0


def cmd_close(cfg: Config, args: argparse.Namespace) -> int:
    day = _day(args)
    odds = OddsAPIClient(
        cfg.creds.odds_api_key, regions=cfg.odds_regions, cache_dir=cfg.odds_cache_dir, cache_ttl=0
    )
    slate, board = odds.fetch_board(day)
    if not slate.games:
        print(f"No NCAAF games to snapshot for {day}.")
        return 0
    quotes = closing_quotes(slate, board)
    save_closing(quotes, cfg.closing_file(day))
    print(f"Captured {len(quotes)} closing quotes -> {cfg.closing_file(day)}")
    return 0


def _grade_slate(cfg: Config, day: Date) -> list[tuple[Recommendation, str]]:
    recs = load_json(cfg.predictions_file(day))
    cfbd = CFBDClient(cfg.creds.cfbd_api_key)
    index = build_result_index(cfbd.fetch_results(_season(cfg, day), day))
    graded: list[tuple[Recommendation, str]] = []
    for rec in recs:
        res = result_for(rec, index)
        if res is None:
            continue
        outcome = grade(rec, res)
        if outcome is not None:
            graded.append((rec, outcome))
    return graded


def cmd_audit(cfg: Config, args: argparse.Namespace) -> int:
    day = _day(args)
    if not cfg.predictions_file(day).exists():
        print(f"No saved predictions for {day}; nothing to grade.")
        return 1
    graded = _grade_slate(cfg, day)
    if not graded:
        print(f"No graded markets for {day} yet (results may not be final).")
        return 0

    entries = entries_from_graded(graded, day)
    closing = load_closing(cfg.closing_file(day))
    if closing:
        for e in entries:
            e.close_odds, e.close_prob, e.clv, e.clv_ev = compute_clv(
                e.market, e.selection, e.odds, e.fair_prob, closing
            )
    merged = update_ledger(cfg.ledger_file, entries, day)
    print(f"Graded {len(entries)} markets; ledger now {len(merged)} rows.")

    scorecard = build_scorecard(graded, day)
    append_scorecard(scorecard, cfg.scorecard_file)
    buy = next((r for r in scorecard if r.market == "All" and r.tier == "Buy (S+M)"), None)
    if buy is not None and buy.n:
        print(
            f"Buy PPV {buy.ppv:.1%} vs break-even {buy.breakeven:.1%} "
            f"(edge {buy.edge_vs_be:+.1%}); ROI {buy.roi:+.1%} on {buy.n} bets."
        )

    _emit_ledger(cfg, merged, day, n_graded=len(entries), email=not args.no_email, to=args.to)
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    entries = load_ledger(cfg.ledger_file)
    if not entries:
        print("Ledger is empty; run `cfb-engine audit` first.")
        return 1
    day = _day(args)
    _emit_ledger(cfg, entries, day, n_graded=0, email=not args.no_email, to=args.to)
    return 0


def cmd_calibrate(cfg: Config, args: argparse.Namespace) -> int:
    from cfb_engine.audit.grade import PUSH
    from cfb_engine.calibration import Calibrator

    entries = load_ledger(cfg.ledger_file)
    rows: list[tuple[str, float, int]] = []
    for e in entries:
        if e.result == PUSH:
            continue
        prob = e.raw_prob if e.raw_prob is not None else e.model_prob
        rows.append((e.market, prob, 1 if e.result == "win" else 0))
    if not rows:
        print("No graded, non-push bets in the ledger to calibrate from.")
        return 1
    calib = Calibrator.fit(rows)
    calib.to_json(cfg.calibration_file)
    print(f"Fit calibration from {len(rows)} graded bets -> {cfg.calibration_file}")
    return 0


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> int:
    from cfb_engine.backtest import run_backtest

    season = args.season or _season(cfg, _day(args))
    scores = run_backtest(cfg, season)
    if not scores:
        print(f"No completed games / ratings for {season} to backtest.")
        return 1
    print(f"Engine A/B on {season} ({scores[0].n} games), same ratings-implied means:")
    print(f"{'engine':>8}  {'Brier':>7}  {'logloss':>8}  {'margin RMSE':>12}  {'total RMSE':>11}")
    for s in scores:
        print(
            f"{s.engine:>8}  {s.brier:>7.4f}  {s.logloss:>8.4f}  "
            f"{s.margin_rmse:>12.2f}  {s.total_rmse:>11.2f}"
        )
    print(
        "note: means are shared, so RMSE is identical by design; the engines differ\n"
        "only in distribution shape, which shows up in Brier/logloss (moneyline)."
    )
    return 0


def cmd_scorecard(cfg: Config, args: argparse.Namespace) -> int:
    if not cfg.scorecard_file.exists():
        print("No scorecard yet; run `cfb-engine audit` on graded slates first.")
        return 1
    print(cfg.scorecard_file.read_text().rstrip())
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _emit_ledger(
    cfg: Config,
    entries: list[LedgerEntry],
    day: Date,
    *,
    n_graded: int,
    email: bool,
    to: str | None,
) -> None:
    overall = [engine_metrics(entries), *overall_metrics(entries)]
    daily = daily_rollup(entries)
    markets = market_metrics(entries)
    clv_rows = clv_summary([(e.category, e.clv, e.clv_ev) for e in entries])
    xlsx = write_ledger_workbook(
        entries,
        overall,
        daily,
        cfg.audit_dir / f"PayoffPitch_CFB_Ledger_{day.isoformat()}.xlsx",
        market_rows=markets,
        clv_rows=clv_rows,
    )
    print(f"Wrote ledger workbook -> {xlsx}")
    extra = [(xlsx.name, xlsx.read_bytes())]
    generate_audit_report(
        day, overall, clv_rows, n_graded, cfg, email=email, to=to, extra_attachments=extra
    )


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cfb-engine", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, *, email: bool = False) -> None:
        sp.add_argument("--date", help="slate date YYYY-MM-DD (default: today, US/Eastern)")
        if email:
            sp.add_argument("--no-email", action="store_true", help="write files but do not email")
            sp.add_argument("--to", help="override the email recipient")

    add_common(sub.add_parser("run", help="price today's slate"), email=True)
    add_common(sub.add_parser("card", help="rebuild article/PDF/MP3 from saved predictions"), email=True)
    add_common(sub.add_parser("close", help="snapshot the closing market"))
    add_common(sub.add_parser("audit", help="grade a slate and update the ledger"), email=True)
    add_common(sub.add_parser("report", help="rebuild the ledger workbook/report"), email=True)
    add_common(sub.add_parser("calibrate", help="refit probability calibration"))
    bt = sub.add_parser("backtest", help="A/B the score engines (normal vs markov)")
    bt.add_argument("--season", type=int, help="season year (default: inferred)")
    bt.add_argument("--date", help="slate date used to infer the season")
    add_common(sub.add_parser("scorecard", help="print the PPV/NPV-by-market scorecard"))
    return p


_DISPATCH = {
    "run": cmd_run,
    "card": cmd_card,
    "close": cmd_close,
    "audit": cmd_audit,
    "report": cmd_report,
    "calibrate": cmd_calibrate,
    "backtest": cmd_backtest,
    "scorecard": cmd_scorecard,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not hasattr(args, "no_email"):
        args.no_email = False
    if not hasattr(args, "to"):
        args.to = None
    cfg = load_config()
    return _DISPATCH[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
