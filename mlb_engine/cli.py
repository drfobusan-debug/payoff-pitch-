"""Command-line entrypoint: `mlb-engine run` and `mlb-engine audit`."""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from mlb_engine.audit.analysis import prop_insights
from mlb_engine.audit.email import send_audit_summary
from mlb_engine.audit.grade import LOSS, WIN, grade
from mlb_engine.audit.ledger import (
    LedgerEntry,
    daily_engine_metrics,
    daily_rollup,
    engine_metrics,
    entries_from_graded,
    load_ledger,
    overall_metrics,
    prop_metrics,
    runline_metrics,
    update_ledger,
)
from mlb_engine.audit.scorecard import append_scorecard, build_scorecard
from mlb_engine.calibration import Calibrator
from mlb_engine.config import Config, load_config
from mlb_engine.data.fangraphs import FanGraphsClient
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.data.results import fetch_result
from mlb_engine.data.rotowire import RotowireClient
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import VSINClient
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market.tiers import Tier
from mlb_engine.output.card import build_cards, render_html, render_markdown
from mlb_engine.output.email import EmailNotConfigured, send_card_email
from mlb_engine.output.excel import write_ledger_workbook, write_workbook
from mlb_engine.pipeline import Pipeline, PipelineDeps, load_calibrator
from mlb_engine.recommendations import Recommendation, load_json, save_json


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


def _generate_card(
    recs: list[Recommendation],
    slate_date: Date,
    cfg: Config,
    *,
    email: bool,
    to: str | None,
) -> tuple[Path, Path]:
    """Write the card's Markdown + HTML and optionally email it."""
    cards = build_cards(recs)
    md = render_markdown(cards, slate_date)
    html_body = render_html(cards, slate_date)
    md_path = cfg.output_dir / f"card_{slate_date.isoformat()}.md"
    html_path = cfg.output_dir / f"card_{slate_date.isoformat()}.html"
    md_path.write_text(md)
    html_path.write_text(html_body)
    print(f"Card: {len(cards)} games -> {md_path}")

    if email:
        subject = f"PayoffPitch Card — {slate_date.isoformat()} ({len(cards)} games)"
        try:
            recipient = send_card_email(
                cfg,
                subject=subject,
                html_body=html_body,
                text_body="Your PayoffPitch card is attached; HTML in the body.\n",
                to=to,
                attachments=[(md_path.name, md.encode(), "markdown")],
            )
            print(f"Emailed card to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Email not sent: {exc}")
    return md_path, html_path


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

    if getattr(args, "card", False) or getattr(args, "email", False):
        _generate_card(recs, slate_date, cfg, email=args.email, to=args.to)
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate_date = _parse_date(args.date, Date.today())
    pred_path = cfg.audit_dir / f"predictions_{slate_date.isoformat()}.json"
    if not pred_path.exists():
        print(f"No predictions found for {slate_date} at {pred_path}; run `mlb-engine run` first")
        return 1
    recs = load_json(pred_path)
    _generate_card(recs, slate_date, cfg, email=args.email, to=args.to)
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
            results[pk] = fetch_result(pk, cache_dir=cfg.cache_dir)
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
    engine = engine_metrics(all_entries)
    overall = [engine, *overall_metrics(all_entries)]
    daily_engine = daily_engine_metrics(all_entries)
    props = prop_metrics(all_entries)
    runlines = runline_metrics(all_entries)
    insights = prop_insights(all_entries)
    ledger_xlsx = cfg.output_dir / "ledger.xlsx"
    write_ledger_workbook(
        all_entries,
        overall,
        daily_rollup(all_entries),
        ledger_xlsx,
        daily_engine=daily_engine,
        prop_rows=props,
        insights=insights,
        runline_rows=runlines,
    )

    print(f"Graded {len(graded)} markets for {audit_date}")
    for row in rows:
        print(
            f"  {row.tier:<12} n={row.n:<4} PPV={row.ppv:.3f} "
            f"sens={row.sensitivity:.3f} spec={row.specificity:.3f} "
            f"NPV={row.npv:.3f} ROI={row.roi:+.3f}"
        )
    print(f"\nLedger: {len(all_entries)} graded bets across all dates -> {ledger_xlsx}")
    print(
        f"  WHOLE ENGINE   n={engine.n:<5} PPV={engine.ppv:.3f} NPV={engine.npv:.3f} "
        f"sens={engine.sensitivity:.3f} spec={engine.specificity:.3f} "
        f"(model-favored side wins {engine.ppv * 100:.1f}%)"
    )
    for m in overall:
        print(
            f"  OVERALL {m.tier:<14} n={m.n:<5} win%={m.win_pct * 100:5.1f} "
            f"PPV={m.ppv:.3f} sens={m.sensitivity:.3f} spec={m.specificity:.3f} "
            f"NPV={m.npv:.3f} ROI={m.roi * 100:+.1f}% units={m.units:+.1f}"
        )

    if props:
        print("\nProps PPV/NPV:")
        for m in props:
            print(
                f"  {m.tier:<14} n={m.n:<5} PPV={m.ppv:.3f} NPV={m.npv:.3f} "
                f"sens={m.sensitivity:.3f} spec={m.specificity:.3f}"
            )
    if runlines:
        print("\nRun lines PPV/NPV (VETO rows = what each gate removed):")
        for m in runlines:
            print(
                f"  {m.tier:<18} n={m.n:<5} win%={m.win_pct * 100:5.1f} "
                f"PPV={m.ppv:.3f} NPV={m.npv:.3f}"
            )

    if insights:
        print(f"\nProp insights ({len(insights)}):")
        for ins in insights:
            print(f"  [{ins.kind}] {ins.finding}")

    if getattr(args, "email", True):
        send_audit_summary(ledger_xlsx, audit_date.isoformat(), cfg)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Refit the isotonic calibration map from the audit ledger.

    Trains on every graded row (pushes dropped) and holds out the most recent
    ``--holdout`` slates to check, market by market, whether the refit map
    actually beats the packaged 2024 fit out of sample. Only the markets that
    win are adopted; the rest keep the packaged map.
    """
    cfg = load_config()
    ledger_path = cfg.audit_dir / "ledger.csv"
    entries = [
        e for e in load_ledger(ledger_path) if e.result in (WIN, LOSS) and e.raw_prob is not None
    ]
    if not entries:
        print(
            f"No graded rows with a raw probability in {ledger_path}; "
            "re-run `mlb-engine audit` to backfill the raw_prob column"
        )
        return 1

    dates = sorted({e.date for e in entries})
    if len(dates) <= args.holdout:
        print(f"Need more than {args.holdout} graded slates to validate; have {len(dates)}")
        return 1
    split = dates[-args.holdout]

    def rows_of(subset: list[LedgerEntry]) -> list[tuple[str, float, int]]:
        return [
            (e.market, e.raw_prob, 1 if e.result == WIN else 0)
            for e in subset
            if e.raw_prob is not None
        ]

    rows = rows_of(entries)
    train = rows_of([e for e in entries if e.date < split])
    test = rows_of([e for e in entries if e.date >= split])

    packaged = load_calibrator()
    refit = Calibrator.fit(train)

    def brier(cal: Calibrator, subset: list[tuple[str, float, int]]) -> float:
        return sum((cal.apply(m, p) - w) ** 2 for m, p, w in subset) / len(subset)

    by_market: dict[str, list[tuple[str, float, int]]] = {}
    for row in test:
        by_market.setdefault(row[0], []).append(row)

    # Adopt the refit map per market rather than wholesale. Local history is
    # thin, and on the eight-slate ledger it beat the packaged 2024 fit exactly
    # where that fit was stale or absent (batter_tb had no map at all) while
    # losing on the low-volume game-level markets.
    print(f"Holdout: {split}..{dates[-1]} ({len(test)} rows), trained on {len(train)}")
    print(f"{'market':<14}{'n':>7}{'packaged':>11}{'refit':>10}{'delta':>9}  adopt")
    adopt: list[str] = []
    for market in sorted(by_market):
        subset = by_market[market]
        b_old, b_new = brier(packaged, subset), brier(refit, subset)
        take = len(subset) >= args.min_holdout and b_new < b_old
        if take:
            adopt.append(market)
        print(
            f"{market:<14}{len(subset):>7}{b_old:>11.4f}{b_new:>10.4f}"
            f"{b_new - b_old:>+9.4f}  {'yes' if take else 'no'}"
        )

    if not adopt and not args.force:
        print("\nNo market improved out of sample; nothing written")
        return 1

    final = Calibrator.fit(rows)
    merged = Calibrator(
        maps={**packaged.maps, **{m: final.maps[m] for m in adopt if m in final.maps}},
        default=packaged.default,
    )
    merged.to_json(cfg.calibration_file)
    print(f"\nAdopted refit maps for {len(adopt)}: {', '.join(adopt) or 'none'}")
    print(f"Wrote {cfg.calibration_file} (other markets keep the packaged fit)")
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
    r.add_argument("--card", action="store_true", help="also write the daily card (md + html)")
    r.add_argument("--email", action="store_true", help="email the daily card after the run")
    r.add_argument("--to", help="email recipient (default: MLBE_EMAIL_TO)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("card", help="build the daily card from a prior run's predictions")
    c.add_argument("--date", help="slate date YYYY-MM-DD (default: today)")
    c.add_argument("--email", action="store_true", help="email the card")
    c.add_argument("--to", help="email recipient (default: MLBE_EMAIL_TO)")
    c.set_defaults(func=cmd_card)

    a = sub.add_parser("audit", help="grade a prior slate and update scorecard")
    a.add_argument("--date", help="slate date to audit YYYY-MM-DD (default: yesterday)")
    a.add_argument("--no-email", action="store_false", dest="email",
                  help="skip sending the nightly audit email")
    a.set_defaults(func=cmd_audit, email=True)

    cal = sub.add_parser("calibrate", help="refit the calibration map from the audit ledger")
    cal.add_argument("--holdout", type=int, default=2, help="slates held out for validation")
    cal.add_argument(
        "--min-holdout", type=int, default=200, help="holdout rows a market needs to be adopted"
    )
    cal.add_argument("--force", action="store_true", help="write even if no market improves")
    cal.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
