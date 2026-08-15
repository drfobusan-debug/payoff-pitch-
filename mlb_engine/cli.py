"""Command-line entrypoint: `mlb-engine run` and `mlb-engine audit`."""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from mlb_engine.audit.analysis import (
    dog_vs_favorite,
    price_bucket_findings,
    price_buckets,
    prop_insights,
)
from mlb_engine.audit.clv import (
    attach_clv,
    closing_quotes,
    clv_rows,
    load_closing,
    merge_closing,
    save_closing,
    summarize,
)
from mlb_engine.audit.grade import LOSS, WIN, grade
from mlb_engine.audit.ledger import (
    LedgerEntry,
    daily_engine_metrics,
    daily_rollup,
    engine_metrics,
    engine_rows,
    entries_from_graded,
    gate_metrics,
    load_ledger,
    one_side_per_prop,
    overall_metrics,
    prop_metrics,
    runline_metrics,
    update_ledger,
)
from mlb_engine.audit.outside import entries_from_picks, head_to_head
from mlb_engine.audit.probation import (
    WATCHING,
    market_probation,
    screen_probation,
)
from mlb_engine.audit.scorecard import append_scorecard, build_scorecard
from mlb_engine.calibration import FEATURE_BASIS, FEATURE_BASIS_SINCE, Calibrator
from mlb_engine.config import Config, load_config
from mlb_engine.data.batx import annotate as annotate_batx
from mlb_engine.data.batx import load_rows as load_batx_rows
from mlb_engine.data.collapse import capture_slate
from mlb_engine.data.fangraphs import FanGraphsClient
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.data.oddsapi import DEFAULT_PROP_MARKETS, OddsAPIClient
from mlb_engine.data.opta import (
    OptaClient,
    annotate,
    load_rows,
    merge_rows,
    save_rows,
)
from mlb_engine.data.results import GameResult, fetch_result
from mlb_engine.data.rotowire import RotowireClient
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.data.teamrankings import (
    TeamRankingsClient,
    load_picks,
    merge_picks,
    save_picks,
)
from mlb_engine.data.vsin import VSINClient
from mlb_engine.features.team_form import build_team_forms, compute_luck_gaps, save_team_forms
from mlb_engine.filters.weather import WeatherProvider
from mlb_engine.market.tiers import Tier
from mlb_engine.output.audit_insight import generate_audit_insight
from mlb_engine.output.card import build_cards, render_html, render_markdown, render_pdf
from mlb_engine.output.daily_preview import generate_daily_preview
from mlb_engine.output.email import EmailNotConfigured, send_card_email
from mlb_engine.output.excel import write_ledger_workbook, write_workbook
from mlb_engine.output.regression_radar import generate_radar_pdf
from mlb_engine.output.regression_radar import render_html as render_radar_html
from mlb_engine.output.regression_radar import render_markdown as render_radar_markdown
from mlb_engine.output.report import (
    PdfNotAvailable,
    build_report_data,
    daily_entries,
    render_html_report,
    render_markdown_report,
    weekly_entries,
)
from mlb_engine.output.report import render_pdf as render_report_pdf
from mlb_engine.pipeline import Pipeline, PipelineDeps, load_calibrator
from mlb_engine.preview import save_previews
from mlb_engine.recommendations import Recommendation, load_json, save_json
from mlb_engine.state import (
    PREGAME_SUFFIX,
    STATE_BRANCH,
    auto_pull,
    auto_push,
    pull_state,
    push_state,
)


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
    workbook: Path | None = None,
) -> tuple[Path, Path]:
    """Write the card's Markdown + HTML + PDF and optionally email it.

    When ``email`` is set the message carries the written card as a PDF plus,
    when available, the master Excel bet sheet (``workbook``).
    """
    cards = build_cards(recs)
    md = render_markdown(cards, slate_date)
    html_body = render_html(cards, slate_date)
    md_path = cfg.output_dir / f"card_{slate_date.isoformat()}.md"
    html_path = cfg.output_dir / f"card_{slate_date.isoformat()}.html"
    pdf_path = cfg.output_dir / f"card_{slate_date.isoformat()}.pdf"
    md_path.write_text(md)
    html_path.write_text(html_body)
    try:
        rendered = render_pdf(html_body)
        pdf_path.write_bytes(rendered)
        pdf_bytes: bytes | None = rendered
    except Exception as exc:  # noqa: BLE001 - PDF is best-effort; fall back to markdown
        pdf_bytes = None
        print(f"Card PDF not rendered ({exc}); attaching markdown instead")
    print(f"Card: {len(cards)} games -> {md_path}")

    if email:
        subject = f"PayoffPitch Card — {slate_date.isoformat()} ({len(cards)} games)"
        if pdf_bytes is not None:
            attachments = [(pdf_path.name, pdf_bytes)]
        else:
            attachments = [(md_path.name, md.encode())]
        if workbook is not None and workbook.exists():
            attachments.append((workbook.name, workbook.read_bytes()))
        try:
            recipient = send_card_email(
                cfg,
                subject=subject,
                html_body=html_body,
                text_body=(
                    "Your PayoffPitch daily slate card (PDF) and the master Excel "
                    "bet sheet are attached; the card is also in the HTML body.\n"
                ),
                to=to,
                attachments=attachments,
            )
            print(f"Emailed card to {recipient}")
        except EmailNotConfigured as exc:
            print(f"Email not sent: {exc}")
    return md_path, html_path


def _odds_client(cfg: Config, *, cache_ttl: int | None = None) -> OddsAPIClient:
    """Odds API client on the configured credit budget.

    ``cache_ttl`` overrides the configured TTL. The closing snapshot passes 0:
    serving it a cached board from the pre-slate run would silently report zero
    closing line value on every bet.
    """
    props = cfg.odds_props if cfg.odds_props is not None else DEFAULT_PROP_MARKETS
    return OddsAPIClient(
        cfg.creds.odds_api_key,
        prop_markets=props,
        include_f5=cfg.odds_f5,
        cache_dir=cfg.odds_cache_dir,
        cache_ttl=cfg.odds_cache_ttl if cache_ttl is None else cache_ttl,
        min_credits=cfg.odds_min_credits,
    )


def _generate_report(
    entries: list[LedgerEntry],
    cfg: Config,
    *,
    period_label: str,
    subtitle: str,
    slug: str,
    email: bool,
    to: str | None,
    history: list[LedgerEntry] | None = None,
) -> tuple[Path, Path, Path | None]:
    """Build the audit report (md + html + pdf) and optionally email it."""
    data = build_report_data(
        entries, period_label=period_label, subtitle=subtitle, history=history
    )
    md = render_markdown_report(data)
    html_body = render_html_report(data)
    md_path = cfg.output_dir / f"audit_report_{slug}.md"
    html_path = cfg.output_dir / f"audit_report_{slug}.html"
    md_path.write_text(md)
    html_path.write_text(html_body)

    pdf_path: Path | None = cfg.output_dir / f"audit_report_{slug}.pdf"
    pdf_bytes: bytes | None = None
    try:
        pdf_bytes = render_report_pdf(html_body)
        assert pdf_path is not None
        pdf_path.write_bytes(pdf_bytes)
    except PdfNotAvailable as exc:
        print(f"PDF not written: {exc}")
        pdf_path = None

    print(f"Audit report -> {md_path}" + (f" (+ {pdf_path})" if pdf_path else ""))

    if email:
        subject = f"PayoffPitch Audit — {period_label} ({subtitle})"
        attachments: list[tuple[str, bytes]] = [(md_path.name, md.encode())]
        if pdf_bytes is not None and pdf_path is not None:
            attachments.insert(0, (pdf_path.name, pdf_bytes))
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


def _annotate_opta(cfg: Config, recs: list, slate_date: Date) -> None:
    """Put the outside model's read on the card, before the workbook is written.

    Prefers the capture already on disk and only goes to VSIN when the slate has
    none, so a morning that already ran ``mlb-engine opta`` costs nothing. The
    whole thing is best-effort: an annotation is worth strictly less than the
    card, and must never be able to stop it being produced.
    """
    try:
        rows = load_rows(_opta_path(cfg, slate_date.isoformat()))
        if not rows:
            client = OptaClient()
            day = next(
                (d for d, iso in client.slate_dates().items() if iso == slate_date.isoformat()),
                None,
            )
            if day is None:
                return
            rows = client.fetch(day=day, date=slate_date.isoformat())
        matched = annotate(recs, rows)
    except Exception:  # noqa: BLE001 - a benchmark must not break the slate
        logging.warning("Opta benchmark unavailable; card written without it", exc_info=True)
        return
    if matched:
        print(f"Opta benchmark: {matched} of {len(recs)} selections carry an outside projection")


def _annotate_batx(cfg: Config, recs: list, slate_date: Date) -> None:
    """Put THE BAT X's number beside ours, where a slate has been priced.

    Reads only what is already on disk -- the export is a manual download, so
    there is nothing to fetch -- and, like the Opta benchmark, is worth less
    than the card and must never be able to stop it being written.
    """
    try:
        rows = load_batx_rows(cfg.batx_dir / f"{slate_date.isoformat()}.csv")
        if not rows:
            return
        matched = annotate_batx(recs, rows)
    except Exception:  # noqa: BLE001 - a benchmark must not break the slate
        logging.warning("BAT X benchmark unavailable; card written without it", exc_info=True)
        return
    if matched:
        print(f"BAT X benchmark: {matched} of {len(recs)} selections carry an outside projection")


def _capture_teamrankings(cfg: Config, slate_date: Date) -> None:
    """Store the outside model's picks for tonight, so the audit can grade them.

    Runs as part of the slate because their grid keeps no archive: a pick not
    captured before the games is not recoverable afterwards, and a benchmark with
    holes in it cannot be compared over a season. Best-effort, like Opta.
    """
    try:
        iso = slate_date.isoformat()
        picks = TeamRankingsClient().fetch(date=iso)
        if not picks:
            return
        path = _tr_path(cfg, iso)
        save_picks(path, merge_picks(load_picks(path), picks))
    except Exception:  # noqa: BLE001 - a benchmark must not break the slate
        logging.warning("TeamRankings benchmark unavailable", exc_info=True)
        return
    print(f"TeamRankings benchmark: {len(picks)} picks captured for {iso}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    deps = PipelineDeps(
        stats=MLBStatsClient(),
        statcast=StatcastRepository(cfg.cache_dir),
        weather=WeatherProvider(),
        vsin=VSINClient(cfg.creds),
        oddsapi=_odds_client(cfg),
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
    _annotate_opta(cfg, recs, slate_date)
    _annotate_batx(cfg, recs, slate_date)
    _capture_teamrankings(cfg, slate_date)

    xlsx = args.out or str(cfg.output_dir / f"mlb_recommendations_{slate_date.isoformat()}.xlsx")
    write_workbook(recs, Path(xlsx), slate_date)
    save_json(recs, cfg.audit_dir / f"predictions_{slate_date.isoformat()}.json")
    previews = pipe.previews
    save_previews(previews, cfg.audit_dir / f"previews_{slate_date.isoformat()}.json")

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
        # Write the card (md/html/pdf) for the record, but do NOT email it here:
        # the slate preview below owns email delivery so a single message carries
        # the Morningstar article + audio + Excel bet sheet (mirrors the audit).
        _generate_card(
            recs, slate_date, cfg, email=False, to=args.to, workbook=Path(xlsx)
        )
        attachments: list[tuple[str, bytes]] = []
        if Path(xlsx).exists():
            attachments.append((Path(xlsx).name, Path(xlsx).read_bytes()))
        if args.email:
            radar_pdf = _build_radar_pdf(pipe, slate_date, cfg)
            if radar_pdf is not None:
                attachments.append(
                    (f"regression_radar_{slate_date.isoformat()}.pdf", radar_pdf)
                )
        generate_daily_preview(
            previews,
            slate_date,
            cfg,
            email=args.email,
            to=args.to,
            recs=recs,
            extra_attachments=attachments or None,
        )
    # Publish the picks at the prices they were priced at, so whichever machine
    # grades this slate grades what was actually sent.
    _state_push(cfg, f"run {slate_date.isoformat()}: {len(recs)} markets priced")
    return 0


def _build_radar_pdf(pipe: Pipeline, slate_date: Date, cfg: Config) -> bytes | None:
    """Build the Regression Radar PDF for the slate and persist md/html/pdf.

    Returns the PDF bytes for the preview email, or ``None`` when the slate
    isn't available or the radar can't be rendered.
    """
    slate = pipe.slate
    if slate is None:
        return None
    pitcher_names: set[str] = set()
    batter_names: set[str] = set()
    for game in slate.games:
        for side in (game.home, game.away):
            if side.probable_pitcher:
                pitcher_names.add(side.probable_pitcher.name)
            for slot in side.lineup:
                batter_names.add(slot.player.name)
    radar_pdf, radar = generate_radar_pdf(
        pitcher_names, batter_names, slate_date, year=slate_date.year
    )
    if radar_pdf is None:
        return None
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"regression_radar_{slate_date.isoformat()}"
    (cfg.output_dir / f"{stem}.pdf").write_bytes(radar_pdf)
    (cfg.output_dir / f"{stem}.md").write_text(render_radar_markdown(radar, slate_date))
    (cfg.output_dir / f"{stem}.html").write_text(render_radar_html(radar, slate_date))
    return radar_pdf


def cmd_card(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate_date = _parse_date(args.date, Date.today())
    pred_path = cfg.audit_dir / f"predictions_{slate_date.isoformat()}.json"
    if not pred_path.exists():
        print(f"No predictions found for {slate_date} at {pred_path}; run `mlb-engine run` first")
        return 1
    recs = load_json(pred_path)
    workbook = cfg.output_dir / f"mlb_recommendations_{slate_date.isoformat()}.xlsx"
    _generate_card(
        recs, slate_date, cfg, email=args.email, to=args.to, workbook=workbook
    )
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


def _closing_path(cfg: Config, slate_date: Date) -> Path:
    return cfg.audit_dir / f"closing_{slate_date.isoformat()}.json"


def _state_pull(cfg: Config, slate_date: Date | None = None) -> None:
    """Recover state written by an earlier run, possibly on another machine."""
    if not cfg.state_sync:
        return
    dates = (slate_date.isoformat(),) if slate_date is not None else None
    report = auto_pull(cfg.data_dir, branch=cfg.state_branch, dates=dates)
    if report is not None:
        print(f"State: {report.describe()}")


def _state_push(cfg: Config, message: str) -> None:
    if not cfg.state_sync:
        return
    report = auto_push(cfg.data_dir, message, branch=cfg.state_branch)
    if report is not None:
        print(f"State: {report.describe()}")


def cmd_close(args: argparse.Namespace) -> int:
    """Snapshot the closing market so the next audit can score closing line value.

    Run this as late as possible before a game starts: the closing price is the
    sharpest forecast available, and beating it is the only fast evidence that a
    pick was good. Cheap by design -- one bulk request for the game markets, and
    props only if the credit budget allows.

    Safe to run more than once a day, and worth doing: a game that has started
    has left the pre-match board, so an evening capture alone never sees the
    afternoon slate's close. Repeat captures merge, keeping the last price seen
    for each selection.
    """
    cfg = load_config()
    cfg.ensure_dirs()
    slate_date = _parse_date(args.date, Date.today())
    client = _odds_client(cfg, cache_ttl=0)
    if not client.available():
        print("No Odds API key configured; cannot capture the close")
        return 1
    # An earlier capture of this slate may live on another machine entirely.
    _state_pull(cfg, slate_date)
    slate = MLBStatsClient().get_slate(slate_date)
    quotes = client.fetch(slate, include_props=not args.game_only, pregame_only=True)
    if not quotes:
        print(
            f"No closing prices returned for {slate_date}: every game has started, "
            "or the board is empty. Nothing captured -- an in-play price is not a close."
        )
        return 1
    path = _closing_path(cfg, slate_date)
    already = load_closing(path)
    fresh = closing_quotes(quotes)
    closing = merge_closing(already, fresh)
    save_closing(path, closing)
    markets = len({q.market for q in closing})
    kept = len(closing) - len(fresh)
    detail = f", {kept} carried over from an earlier capture" if kept > 0 else ""
    print(
        f"Captured {len(fresh)} closing prices; {len(closing)} total across "
        f"{markets} markets{detail} -> {path}"
    )
    _state_push(cfg, f"close {slate_date.isoformat()}: {len(closing)} prices")
    return 0


def _opta_path(cfg: Config, slate_date: str) -> Path:
    return cfg.audit_dir / f"opta_{slate_date}.json"


def _tr_path(cfg: Config, slate_date: str) -> Path:
    return cfg.audit_dir / f"teamrankings_{slate_date}.json"


def cmd_teamrankings(args: argparse.Namespace) -> int:
    """Capture TeamRankings' picks for a slate, as an outside benchmark.

    Free and uncredited, but only ever the current grid: there is no date
    parameter and no archive, so a slate not captured before it is replaced is
    gone. The audit grades whatever was captured against the box score, so this
    only has to run once, before the games.
    """
    cfg = load_config()
    cfg.ensure_dirs()
    picks = TeamRankingsClient().fetch()
    if not picks:
        print("TeamRankings' picks grid returned nothing; captured no benchmark.")
        return 1
    wanted = args.date or max(p.date for p in picks)
    slate = [p for p in picks if p.date == wanted]
    if not slate:
        published = ", ".join(sorted({p.date for p in picks}))
        print(f"TeamRankings is showing {published}; {wanted} is not on the grid.")
        return 1
    _state_pull(cfg)
    path = _tr_path(cfg, wanted)
    merged = merge_picks(load_picks(path), slate)
    save_picks(path, merged)
    games = len({p.matchup for p in merged})
    print(f"Captured {len(slate)} TeamRankings picks over {games} games for {wanted} -> {path}")
    _state_push(cfg, f"teamrankings {wanted}: {len(merged)} picks, {games} games")
    return 0


def cmd_opta(args: argparse.Namespace) -> int:
    """Capture VSIN's Opta projections for a slate, as an outside benchmark.

    Free and uncredited, but only ever three days wide: ``day`` clamps at
    yesterday, so a slate not captured within a day of being played is gone.
    Run it twice -- once before the games for the projections and prices, once
    after for the graded outcomes, which the second capture merges in.
    """
    cfg = load_config()
    cfg.ensure_dirs()
    client = OptaClient()
    dates = client.slate_dates()
    if not dates:
        print("VSIN's projection page returned nothing; captured no benchmark.")
        return 1
    day = args.day
    if args.date:
        offsets = [d for d, iso in dates.items() if iso == args.date]
        if not offsets:
            available = ", ".join(sorted(dates.values()))
            print(f"VSIN only publishes {available}; {args.date} is out of reach.")
            return 1
        day = offsets[0]
    slate_date = dates.get(day, "")
    rows = client.fetch(day=day, date=slate_date)
    if not rows:
        print(f"No Opta projections published for {slate_date or day}.")
        return 1
    _state_pull(cfg)
    path = _opta_path(cfg, slate_date)
    merged = merge_rows(load_rows(path), rows)
    save_rows(path, merged)
    graded = sum(1 for r in merged if r.result is not None)
    print(
        f"Captured {len(rows)} Opta projections for {slate_date}; "
        f"{len(merged)} total, {graded} graded -> {path}"
    )
    _state_push(cfg, f"opta {slate_date}: {len(merged)} projections, {graded} graded")
    return 0


def _outside_entries(
    cfg: Config,
    audit_date: Date,
    recs: list[Recommendation],
    results: dict[int, GameResult],
) -> list[LedgerEntry]:
    """Grade the captured outside picks for a slate, best-effort.

    A benchmark must be gradeable on games we did not price -- most of the value
    of a second model is what it says where we said nothing -- so a matchup our
    own recommendations do not cover is looked up on the schedule and its box
    score fetched. The whole thing is wrapped: a benchmark is worth less than
    the audit and must never be able to stop it.
    """
    picks = load_picks(_tr_path(cfg, audit_date.isoformat()))
    if not picks:
        return []
    game_pks = {r.matchup: r.game_pk for r in recs}
    graded = dict(results)
    missing = {p.matchup for p in picks} - set(game_pks)
    if missing:
        try:
            slate = MLBStatsClient().get_slate(audit_date)
            for game in slate.games:
                if game.matchup() in missing:
                    game_pks[game.matchup()] = game.game_pk
                    graded[game.game_pk] = fetch_result(game.game_pk, cache_dir=cfg.cache_dir)
        except Exception:  # noqa: BLE001 - the benchmark never blocks the audit
            logging.warning("could not extend the slate for outside picks", exc_info=True)
    return entries_from_picks(picks, graded, game_pks, audit_date)


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    audit_date = _parse_date(args.date, Date.today() - timedelta(days=1))
    _state_pull(cfg, audit_date)
    pred_path = cfg.audit_dir / f"predictions_{audit_date.isoformat()}.json"
    # A pregame copy is what the card actually sent, at the prices it was sent
    # at. Anything regenerated after the games finished grades different picks
    # against quotes that no longer exist, so it loses to the real thing.
    pregame = cfg.audit_dir / f"predictions_{audit_date.isoformat()}{PREGAME_SUFFIX}"
    if pregame.exists():
        if pred_path.exists():
            print(f"Grading the pregame predictions from {pregame.name}, not the local re-price")
        pred_path = pregame
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

    # Observe-only: log per-pitcher, per-inning run attribution (inherited-runner
    # credit) so pitcher collapse/volatility can be measured. No effect on grading.
    final_pks = [pk for pk, res in results.items() if res is not None and res.final]
    try:
        collapse_lines = capture_slate(final_pks, audit_date.isoformat(), cfg.cache_dir, cfg.audit_dir)
        if collapse_lines:
            print(f"Captured {len(collapse_lines)} pitcher-inning rows -> {cfg.audit_dir / 'collapse_ledger.csv'}")
    except Exception as exc:  # noqa: BLE001 -- capture must never break the audit
        logging.warning("collapse capture failed: %s", exc)

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
    closing = load_closing(_closing_path(cfg, audit_date))
    n_clv = attach_clv(entries, closing)
    # The outside model's picks are graded off the same box scores and written
    # beside ours, on their own rows, in the same `update_ledger` call: a second
    # write for the same date would be treated as a re-audit and drop them.
    outside = _outside_entries(cfg, audit_date, recs, results)
    all_entries = update_ledger(
        cfg.audit_dir / "ledger.csv", entries + outside, audit_date
    )
    # The ledger keeps both sides of every prop so the fade stays graded, and
    # the outside model's picks beside ours; a measurement of the engine takes
    # one row per wager of ours (see `one_side_per_prop`, `engine_rows`).
    # Counting a benchmark inside our own PPV, ROI or CLV would corrupt the
    # numbers it exists to check. The workbook below still writes every row.
    measured = one_side_per_prop(engine_rows(all_entries))
    # CLV especially: the two sides of a prop are devigged complements, so their
    # CLV is an exact negation and counting both drives every market's mean to
    # zero and "beat the close" to 50% arithmetically.
    clv_summary = summarize(clv_rows(measured))
    engine = engine_metrics(measured)
    overall = [engine, *overall_metrics(measured)]
    daily_engine = daily_engine_metrics(measured)
    props = prop_metrics(measured)
    runlines = runline_metrics(measured)
    gates = gate_metrics(measured)
    insights = prop_insights(measured)
    ledger_xlsx = cfg.output_dir / "ledger.xlsx"
    write_ledger_workbook(
        all_entries,
        overall,
        daily_rollup(measured),
        ledger_xlsx,
        daily_engine=daily_engine,
        prop_rows=props,
        insights=insights,
        runline_rows=runlines,
        clv_rows=clv_summary,
    )

    if outside:
        _print_head_to_head(entries, outside)

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

    if closing:
        print(f"\nClosing line value ({n_clv} of {len(entries)} rows had a captured close):")
        for c in clv_summary:
            print(
                f"  {c.label:<14} n={c.n:<5} CLV={c.mean_clv * 100:+.2f} pts "
                f"beat close={c.beat_close_pct * 100:5.1f}% "
                f"CLV EV={c.mean_clv_ev:+.4f}/unit"
            )
    else:
        print(
            "\nNo closing snapshot for this slate: run `mlb-engine close` just before "
            "first pitch to score closing line value (~3 credits)."
        )

    price_rows = [*dog_vs_favorite(measured), *price_buckets(measured)]
    if price_rows:
        print("\nReal-priced buys by price length (Need = win rate the price demands):")
        for pb in price_rows:
            print(
                f"  {pb.label:<34} n={pb.n:<4} win%={pb.win_rate * 100:5.1f} "
                f"need={pb.breakeven * 100:5.1f} gap={pb.shortfall * 100:+5.1f}pts "
                f"ROI={pb.roi * 100:+6.1f}% units={pb.units:+.2f}"
            )
        for finding in price_bucket_findings(measured):
            print(f"  - {finding}")

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

    if gates:
        print("\nScreens: how the picks each one rejected actually finished")
        print("  (win% below Need = the screen deleted losers and is earning its keep)")
        for m in gates:
            print(
                f"  {m.tier:<24} n={m.n:<6} win%={m.win_pct * 100:5.1f} "
                f"need={m.required_win_pct * 100:5.1f} "
                f"gap={(m.win_pct - m.required_win_pct) * 100:+5.1f}pts "
                f"ROI={m.roi * 100:+6.1f}% units={m.units:+9.2f}"
            )

    probation = [*market_probation(measured), *screen_probation(measured)]
    if probation:
        print("\nProbation: markets on their own buys, screens on what they refused")
        print("  (acts only on volume + size + both halves agreeing; see audit/probation.py)")
        for p in probation:
            flag = "  " if p.status == WATCHING else "->"
            print(
                f"  {flag} {p.status:<9} {p.name:<22} n={p.n:<5} "
                f"ROI={p.roi * 100:+6.1f}% se={p.se * 100:4.1f} "
                f"halves {p.first_half * 100:+6.1f}% / {p.second_half * 100:+6.1f}%"
            )
        for p in probation:
            if p.actionable:
                print(f"  - {p.finding}")

    if insights:
        print(f"\nProp insights ({len(insights)}):")
        for ins in insights:
            print(f"  [{ins.kind}] {ins.finding}")

    if getattr(args, "report", False) or getattr(args, "email", False):
        day_entries = daily_entries(all_entries, audit_date)
        # The classic md/HTML/PDF audit report is still written to disk for
        # reference, but email delivery is owned by the insight report below so
        # a single email carries the Excel ledger + article + audio.
        _generate_report(
            day_entries,
            cfg,
            period_label="Daily",
            subtitle=f"slate graded {audit_date.isoformat()}",
            slug=audit_date.isoformat(),
            email=False,
            to=getattr(args, "to", None),
            history=all_entries,
        )
        try:
            ledger_bytes = ledger_xlsx.read_bytes() if ledger_xlsx.exists() else None
            extra = [(ledger_xlsx.name, ledger_bytes)] if ledger_bytes else None
            generate_audit_insight(
                graded,
                audit_date,
                cfg,
                email=getattr(args, "email", False),
                to=getattr(args, "to", None),
                extra_attachments=extra,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("audit insight report failed: %s", exc)
    _state_push(cfg, f"audit {audit_date.isoformat()}: {len(graded)} graded")
    return 0


def _print_head_to_head(ours: list, theirs: list) -> None:
    """Our game-market calls beside the outside model's, for the slate."""
    rows = head_to_head(ours, theirs)
    if not rows:
        return
    both = [r for r in rows if r.ours and r.theirs]
    agreed = [r for r in both if r.agree]
    print(f"\nTeamRankings head-to-head ({len(both)} markets both of us bet):")
    for r in rows:
        ours_txt = f"{r.ours} [{r.our_result}]" if r.ours else "(pass)"
        theirs_txt = f"{r.theirs} {r.their_tier} [{r.their_result}]" if r.theirs else "(lay off)"
        mark = "=" if r.ours and r.theirs and r.agree else " "
        print(f"  {mark} {r.matchup:<12} {r.market:<11} us {ours_txt:<28} them {theirs_txt}")
    if both:
        print(f"  agreed on {len(agreed)} of {len(both)}")


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config()
    _state_pull(cfg)
    ledger_path = cfg.audit_dir / "ledger.csv"
    if not ledger_path.exists():
        print(f"No ledger found at {ledger_path}; run `mlb-engine audit` first")
        return 1
    # The report grades the engine, so the benchmark's rows are not part of it.
    all_entries = engine_rows(load_ledger(ledger_path))
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


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Refit the isotonic calibration map from the audit ledger.

    Trains on every graded row (pushes dropped) and holds out the most recent
    ``--holdout`` slates to check, market by market, whether the refit map
    actually beats the packaged 2024 fit out of sample. Only the markets that
    win are adopted; the rest keep the packaged map.

    Rows priced before ``FEATURE_BASIS_SINCE`` are dropped: a map learns what
    this engine's probabilities mean, so rows produced by a materially different
    feature basis would teach it to undo a correction rather than measure one.
    """
    cfg = load_config()
    ledger_path = cfg.audit_dir / "ledger.csv"
    graded = [
        e
        for e in engine_rows(load_ledger(ledger_path))
        if e.result in (WIN, LOSS) and e.raw_prob is not None
    ]
    since = FEATURE_BASIS_SINCE.isoformat()
    entries = [e for e in graded if e.date >= since]
    if not entries and graded:
        print(
            f"{len(graded)} graded row(s) in {ledger_path}, none on or after "
            f"{FEATURE_BASIS_SINCE}: they were priced on a previous feature basis "
            f"({FEATURE_BASIS} is current). Grade a few slates from this engine first."
        )
        return 1
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


def cmd_state(args: argparse.Namespace) -> int:
    """Sync the audit's memory with the state branch.

    Scheduled runs are separate machines: the capture that snapshots the close,
    the morning card that records what was actually recommended and the audit
    that grades it all start with an empty data directory. Pull before a run
    and push after, or CLV has nothing to compare against and the ledger
    resets to a single slate every night.
    """
    cfg = load_config()
    cfg.ensure_dirs()
    if args.direction == "pull":
        dates = (args.date,) if args.date else None
        report = pull_state(cfg.data_dir, branch=args.branch, dates=dates)
    else:
        message = args.message or f"engine state {Date.today().isoformat()}"
        report = push_state(cfg.data_dir, message, branch=args.branch)
    print(report.describe())
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

    cl = sub.add_parser(
        "close", help="snapshot closing prices for tonight's slate (for CLV scoring)"
    )
    cl.add_argument("--date", help="slate date YYYY-MM-DD (default: today)")
    cl.add_argument(
        "--game-only",
        action="store_true",
        help="skip per-event F5/prop markets: 3 credits for the whole slate",
    )
    cl.set_defaults(func=cmd_close)

    op = sub.add_parser(
        "opta", help="capture VSIN's Opta prop projections as an outside benchmark"
    )
    op.add_argument(
        "--day",
        type=int,
        default=0,
        help="VSIN slate offset: -1 yesterday, 0 today, 1 tomorrow (default: 0)",
    )
    op.add_argument(
        "--date",
        help="slate date YYYY-MM-DD; must be one of the three VSIN publishes",
    )
    op.set_defaults(func=cmd_opta)

    tr = sub.add_parser(
        "teamrankings",
        help="capture TeamRankings' game-market picks and star ratings as a benchmark",
    )
    tr.add_argument(
        "--date",
        help="slate date YYYY-MM-DD; defaults to the latest slate on their grid",
    )
    tr.set_defaults(func=cmd_teamrankings)

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

    cal = sub.add_parser("calibrate", help="refit the calibration map from the audit ledger")
    cal.add_argument("--holdout", type=int, default=2, help="slates held out for validation")
    cal.add_argument(
        "--min-holdout", type=int, default=200, help="holdout rows a market needs to be adopted"
    )
    cal.add_argument("--force", action="store_true", help="write even if no market improves")
    cal.set_defaults(func=cmd_calibrate)

    st = sub.add_parser(
        "state",
        help="sync the audit's memory (predictions, closes, ledger) with the engine-state branch",
    )
    st.add_argument("direction", choices=("pull", "push"))
    st.add_argument("--branch", default=STATE_BRANCH, help=f"state branch (default: {STATE_BRANCH})")
    st.add_argument(
        "--date",
        help="on a pull, restore the pregame predictions for this slate date "
        "(default: the most recent couple on the branch)",
    )
    st.add_argument("--message", help="commit message for a push")
    st.set_defaults(func=cmd_state)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
