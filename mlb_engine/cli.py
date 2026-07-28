"""Command-line entrypoint: `mlb-engine run` and `mlb-engine audit`."""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from mlb_engine.audit.analysis import prop_insights
from mlb_engine.audit.grade import grade
from mlb_engine.audit.ledger import (
    LedgerEntry,
    daily_engine_metrics,
    daily_rollup,
    engine_metrics,
    entries_from_graded,
    load_ledger,
    overall_metrics,
    prop_metrics,
    update_ledger,
)
from mlb_engine.audit.scorecard import append_scorecard, build_scorecard
from mlb_engine.config import Config, load_config
from mlb_engine.data.fangraphs import FanGraphsClient
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.data.results import fetch_result
from mlb_engine.data.rotowire import RotowireClient
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.vsin import VSINClient
from mlb_engine.features.team_form import build_team_forms, compute_luck_gaps, save_team_forms
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market.tiers import Tier
from mlb_engine.output.card import build_cards, render_html, render_markdown
from mlb_engine.output.email import EmailNotConfigured, send_card_email
from mlb_engine.output.excel import write_ledger_workbook, write_workbook
from mlb_engine.output.report import (
    PdfNotAvailable,
    build_report_data,
    daily_entries,
    render_html_report,
    render_markdown_report,
    render_pdf,
    weekly_entries,
)
from mlb_engine.pipeline import Pipeline, PipelineDeps
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
                attachments=[(md_path.name, md.encode(), "text", "markdown")],
            )
            print(f"Emailed card to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Email not sent: {exc}")
    return md_path, html_path


def _generate_report(
    entries: list[LedgerEntry],
    cfg: Config,
    *,
    period_label: str,
    subtitle: str,
    slug: str,
    email: bool,
    to: str | None,
) -> tuple[Path, Path, Path | None]:
    """Build the audit report (md + html + pdf) and optionally email it."""
    data = build_report_data(entries, period_label=period_label, subtitle=subtitle)
    md = render_markdown_report(data)
    html_body = render_html_report(data)
    md_path = cfg.output_dir / f"audit_report_{slug}.md"
    html_path = cfg.output_dir / f"audit_report_{slug}.html"
    md_path.write_text(md)
    html_path.write_text(html_body)

    pdf_path: Path | None = cfg.output_dir / f"audit_report_{slug}.pdf"
    pdf_bytes: bytes | None = None
    try:
        pdf_bytes = render_pdf(html_body)
        assert pdf_path is not None
        pdf_path.write_bytes(pdf_bytes)
    except PdfNotAvailable as exc:
        print(f"PDF not written: {exc}")
        pdf_path = None

    print(f"Audit report -> {md_path}" + (f" (+ {pdf_path})" if pdf_path else ""))

    if email:
        subject = f"PayoffPitch Audit — {period_label} ({subtitle})"
        attachments: list[tuple[str, bytes, str, str]] = [
            (md_path.name, md.encode(), "text", "markdown")
        ]
        if pdf_bytes is not None and pdf_path is not None:
            attachments.insert(0, (pdf_path.name, pdf_bytes, "application", "pdf"))
        try:
            recipient = send_card_email(
                cfg,
                subject=subject,
                html_body=html_body,
                text_body="Your PayoffPitch audit report is attached; HTML in the body.\n",
                to=to,
                attachments=attachments,
            )
            print(f"Emailed audit report to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Email not sent: {exc}")
    return md_path, html_path, pdf_path


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


def cmd_team_form(args: argparse.Namespace) -> int:
    """Build the season team-form (luck-gap) baseline cache -- run once daily."""
    cfg = load_config()
    as_of = _parse_date(args.date, Date.today())
    statcast = StatcastRepository(cfg.cache_dir)
    df = statcast.load_trailing(as_of, args.days, refresh=args.refresh)
    run_diffs = MLBStatsClient().team_run_differentials(as_of.year)
    forms = build_team_forms(df, run_diffs)
    save_team_forms(forms, cfg.team_form_path)
    gaps = compute_luck_gaps(forms)
    print(f"Built team-form baseline for {len(forms)} teams -> {cfg.team_form_path}")
    for team, gap in sorted(gaps.items(), key=lambda kv: kv[1], reverse=True):
        f = forms[team]
        rd = f"{f.actual_rd_g:+.2f}" if f.actual_rd_g is not None else "  n/a"
        print(f"  {team:4s} luck_gap {gap:+.2f}  actual_rd/g {rd}  xrd_proxy {f.xrd_proxy:+.3f}")
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

    entries = entries_from_graded(graded, audit_date, results)
    all_entries = update_ledger(cfg.audit_dir / "ledger.csv", entries, audit_date)
    engine = engine_metrics(all_entries)
    overall = [engine, *overall_metrics(all_entries)]
    daily_engine = daily_engine_metrics(all_entries)
    props = prop_metrics(all_entries)
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
    if insights:
        print(f"\nProp insights ({len(insights)}):")
        for ins in insights:
            print(f"  [{ins.kind}] {ins.finding}")

    if getattr(args, "report", False) or getattr(args, "email", False):
        day_entries = daily_entries(all_entries, audit_date)
        _generate_report(
            day_entries,
            cfg,
            period_label="Daily",
            subtitle=f"slate graded {audit_date.isoformat()}",
            slug=audit_date.isoformat(),
            email=getattr(args, "email", False),
            to=getattr(args, "to", None),
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config()
    ledger_path = cfg.audit_dir / "ledger.csv"
    if not ledger_path.exists():
        print(f"No ledger found at {ledger_path}; run `mlb-engine audit` first")
        return 1
    all_entries = load_ledger(ledger_path)
    end = _parse_date(args.date, Date.today() - timedelta(days=1))

    if args.period == "weekly":
        entries = weekly_entries(all_entries, end)
        start = end - timedelta(days=6)
        period_label = "Weekly"
        subtitle = f"{start.isoformat()} to {end.isoformat()}"
        slug = f"week_{end.isoformat()}"
    else:
        entries = daily_entries(all_entries, end)
        period_label = "Daily"
        subtitle = f"slate graded {end.isoformat()}"
        slug = end.isoformat()

    if not entries:
        print(f"No graded bets in the ledger for the {args.period} window ending {end}")
        return 1

    _generate_report(
        entries,
        cfg,
        period_label=period_label,
        subtitle=subtitle,
        slug=slug,
        email=args.email,
        to=args.to,
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
    a.add_argument(
        "--report", action="store_true", help="also write the daily audit report (md/html/pdf)"
    )
    a.add_argument(
        "--email", action="store_true", help="email the daily audit report after grading"
    )
    a.add_argument("--to", help="email recipient (default: MLBE_EMAIL_TO)")
    a.set_defaults(func=cmd_audit)

    rp = sub.add_parser("report", help="render a daily/weekly audit report from the ledger")
    rp.add_argument(
        "--period", choices=("daily", "weekly"), default="daily", help="report window"
    )
    rp.add_argument("--date", help="end date YYYY-MM-DD (default: yesterday)")
    rp.add_argument("--email", action="store_true", help="email the report")
    rp.add_argument("--to", help="email recipient (default: MLBE_EMAIL_TO)")
    rp.set_defaults(func=cmd_report)

    tf = sub.add_parser(
        "team-form", help="build the season team-form (luck-gap) baseline cache"
    )
    tf.add_argument("--date", help="as-of date YYYY-MM-DD (default: today)")
    tf.add_argument("--days", type=int, default=180, help="season look-back window (days)")
    tf.add_argument("--refresh", action="store_true", help="re-download Statcast for the window")
    tf.set_defaults(func=cmd_team_form)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
