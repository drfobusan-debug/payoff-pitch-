"""``nba-engine`` command line (Phase 0: capture and historical archive).

* ``capture`` -- archive the featured board (ML / ATS / total) and, unless
  ``--board-only``, every game's first-half and five prop markets. ``--close``
  captures only games tipping within the close window, once each: the T-5
  close the ledger will grade CLV against.
* ``watch`` -- one tick of the news alarm (``alarm.py``): compare the board and
  both injury feeds with the last look, re-capture any game that moved, record
  the alert, take any close now due, then fast-poll every alerted game until
  its line settles. Scheduled every five minutes.
* ``history`` -- pull genuine historical quotes for whole seasons at chosen
  anchors before tip, within a credit budget, resumably.
* ``coverage`` -- what the historical archive holds per season and anchor.
* ``injuries`` -- archive (and parse) the day's official injury reports.
* ``results`` -- ESPN finals by quarter with player box lines.
* ``archive`` -- summarise captured live prices for a date.
* ``boxes`` -- archive ESPN finals and box lines for whole seasons (free).
* ``fit-over-bias`` -- fit the Over bias per market and book on the archived
  closes and finals; writes a new ``params/over_bias`` version.
* ``board`` -- the day's selections: consensus fair (Over bias removed), the
  better DraftKings/BetMGM price, one- or two-book, price vs fair.
* ``capture --record TAG`` -- after capturing, write the day's untipped
  selections to the ledger as one immutable priced pass.
* ``grade`` -- settle a slate's ledger on the ESPN finals and stamp each row's
  pre-tip close at its book; written once, when every game is settled.
* ``audit`` -- the graded ledger's tables: buys, tiers, markets, gates (false
  negatives), book, one/two-book, the board's base rates, CLV and integrity.
* ``card`` -- the daily package from the day's recorded ledger passes: slate
  PDF, betting workbook and (``--email``) the email carrying both. No Odds API
  call; context is read off disk (plus free ESPN finals) and prices nothing.

Nothing here prices a game yet: without a model every row is refused
(``no_model``) and recorded as the board's base rate (docs/nba/master_plan.md §13).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, replace
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine_common.podcasts import state as podcast_state
from engine_common.podcasts.episodes import store_dir
from engine_common.podcasts.run import read_feeds
from engine_common.podcasts.shows import NBA
from nba_engine import alarm, podcast, state
from nba_engine.audit import ledger, scorecard
from nba_engine.config import cache_dir, data_dir, load_config, output_dir
from nba_engine.data import boxes, capture, espn_injuries, history, injuries
from nba_engine.data.capture import ALL_MARKETS, EVENT_MARKETS, GAME_MARKETS
from nba_engine.data.espn import ESPNClient, parse_box
from nba_engine.data.oddsapi import SLATE_TZ, OddsAPIClient, parse_utc
from nba_engine.market import overbias, params
from nba_engine.market.board import ev_per_unit, selections
from nba_engine.models import minutes, minutes_fit, rating_fit, ratings
from nba_engine.models.schedule import schedule
from nba_engine.output import context
from nba_engine.output.card import SlateCard, build_card, render_html, render_pdf, render_text
from nba_engine.output.email import EmailNotConfigured, send_package
from nba_engine.output.excel import build_workbook
from nba_engine.schemas import GameResult

log = logging.getLogger("nba_engine")


def _today() -> Date:
    return datetime.now(SLATE_TZ).date()


def _parse_date(raw: str | None) -> Date:
    return Date.fromisoformat(raw) if raw else _today()


_sleep = time.sleep


def _client(cache_ttl: int = 0) -> OddsAPIClient | None:
    cfg = load_config()
    client = OddsAPIClient(
        cfg.creds.odds_api_key,
        cache_dir=cache_dir() / "oddsapi",
        cache_ttl=cache_ttl,
        sport_key=cfg.sport_key,
    )
    if not client.available():
        print("no Odds API key (THE_ODDS_API_KEY / ODDS_API_KEY)", file=sys.stderr)
        return None
    return client


def _close_label(event_id: str) -> str:
    return f"close-{event_id[:12]}"


def _capture_closes(client: OddsAPIClient, root: Path, slate_date: Date, taken: str) -> list[str]:
    """Every market for games tipping within the close window, once per game."""
    cfg = load_config()
    slate = client.fetch_events(slate_date=slate_date, horizon_hours=cfg.capture.horizon_hours)
    now = datetime.now(timezone.utc)
    window = timedelta(minutes=cfg.capture.close_window_min)
    held = {p.name.split("_")[0] for p in capture.snapshot_paths(root, slate_date)}
    written: list[str] = []
    for game in slate.games:
        tip = parse_utc(game.start_utc)
        if tip is None or not now < tip <= now + window or _close_label(game.event_id) in held:
            continue
        rows = client.fetch_event_markets(
            [game], markets=ALL_MARKETS, captured_at=taken, max_events=1
        )
        path = capture.write_snapshot(
            rows, root, slate_date, label=_close_label(game.event_id), captured_at=taken
        )
        if path is not None:
            written.append(path.name)
    return written


def cmd_capture(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate_date = _parse_date(args.date)
    root = data_dir()
    sync = cfg.state_sync and not args.no_sync
    if sync:
        state.auto_pull(root)
    client = _client(args.cache_ttl)
    if client is None:
        return 2
    taken = capture.now_utc()
    written: list[str] = []
    if args.close:
        written.extend(_capture_closes(client, root, slate_date, taken))
        if not written:
            print(f"{slate_date}: no game tips in the close window without a close")
    else:
        slate, rows = client.fetch_board(
            slate_date=slate_date, horizon_hours=cfg.capture.horizon_hours, captured_at=taken
        )
        board = [r for r in rows if r.game_date == slate_date.isoformat()]
        path = capture.write_snapshot(board, root, slate_date, label="board", captured_at=taken)
        if path is not None:
            written.append(path.name)
        games = [g for g in slate.games if g.game_date == slate_date]
        if not args.board_only and games:
            ev = client.fetch_event_markets(
                games,
                markets=EVENT_MARKETS,
                captured_at=taken,
                max_events=min(args.max_events, cfg.capture.max_events),
            )
            path = capture.write_snapshot(ev, root, slate_date, label="events", captured_at=taken)
            if path is not None:
                written.append(path.name)
        if args.record:
            tips = {g.event_id: g.start_utc for g in games}
            path = _record_pass(root, slate_date, tips, args.record, taken)
            if path is not None:
                written.append(path.name)
    print(
        f"{slate_date}: wrote {len(written)} snapshot(s) {written}; "
        f"credits remaining {client.credits_remaining}"
    )
    if sync and written:
        state.auto_push(root, f"nba capture {slate_date} {taken}")
    return 0


def _versions(root: Path) -> tuple[str, dict]:
    """The params versions pricing would use now, and the Over-bias payload."""
    bias = params.latest(root, overbias.NAME)
    rated = params.latest(root, ratings.NAME)
    stamp = ";".join(
        f"{name}={held['version']}"
        for name, held in ((overbias.NAME, bias), (ratings.NAME, rated))
        if held
    )
    return stamp, bias or {}


def _record_pass(root: Path, day: Date, tips: dict[str, str], tag: str, taken: str) -> Path | None:
    """Price every selection on a game that has not tipped, from pre-tip quotes only."""
    now = parse_utc(taken)
    live = {
        e: t
        for e, t in tips.items()
        if (tip := parse_utc(t)) is not None and now is not None and tip > now
    }
    rows = ledger.pregame(capture.read_day(root, day), live)
    versions, bias = _versions(root)
    priced = ledger.price(
        selections(rows),
        ledger.quote_index(rows),
        tips=live,
        shift=overbias.shifts(bias),
        versions=versions,
        pass_tag=tag,
    )
    print(f"{day}: priced {len(priced)} row(s) for the ledger ({tag})")
    return ledger.write_once(priced, ledger.pass_path(root, day, taken, tag))


def _bias_version(row: ledger.LedgerRow) -> str:
    for part in row.versions.split(";"):
        name, _, version = part.partition("=")
        if name == overbias.NAME:
            return version
    return ""


def cmd_grade(args: argparse.Namespace) -> int:
    cfg = load_config()
    root = data_dir()
    day = _parse_date(args.date) if args.date else _today() - timedelta(days=1)
    sync = cfg.state_sync and not args.no_sync
    if sync:
        state.auto_pull(root, trees=("prices", "ledger", "params"))
    out = ledger.graded_path(root, day)
    if out.exists():
        print(f"{day}: already graded ({out.name})")
        return 0
    rows = ledger.of_record(ledger.passes(root, day))
    if not rows:
        print(f"{day}: no priced passes in the ledger")
        return 1
    games = boxes.ensure_results(root, day)
    finals = {f"{g.away} @ {g.home}": g for g in games}
    quotes = capture.read_day(root, day)
    graded: list[ledger.LedgerRow] = []
    for version in sorted({_bias_version(r) for r in rows}):
        held = params.read(root, overbias.NAME, version) if version else None
        if version and held is None:
            print(f"{day}: over_bias {version} not on this machine; not graded", file=sys.stderr)
            return 1
        group = [r for r in rows if _bias_version(r) == version]
        graded += ledger.grade(
            group, finals, quotes, shift=overbias.shifts(held), graded_at=capture.now_utc()
        )
    settled = {f"{g.away} @ {g.home}" for g in games if g.not_played or g.is_final}
    waiting = sorted({r.matchup for r in graded} - settled)
    counts = scorecard.integrity(graded).by_outcome
    print(f"{day}: {len(graded)} row(s) of record; {counts}")
    if waiting:
        print(f"{day}: not written, waiting on {waiting}")
        return 1
    path = ledger.write_once(graded, out)
    print(f"wrote {out}" if path else f"{day}: nothing written")
    if sync and path is not None:
        state.auto_push(root, f"nba grade {day}", trees=("ledger",))
    return 0


def _ledger_days(root: Path, since: str | None, until: str | None) -> list[Date]:
    base = root / "ledger"
    days = (
        sorted(Date.fromisoformat(p.name) for p in base.iterdir() if p.is_dir())
        if base.is_dir()
        else []
    )
    if since:
        days = [d for d in days if d >= Date.fromisoformat(since)]
    if until:
        days = [d for d in days if d <= Date.fromisoformat(until)]
    return days


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_config()
    root = data_dir()
    if cfg.state_sync and not args.no_sync:
        state.auto_pull(root, trees=("ledger",))
    days = _ledger_days(root, args.since, args.until)
    rows = ledger.graded_rows(root, days)
    if not rows:
        print("no graded ledger rows in range")
        return 1
    print(f"{len(rows)} graded row(s) over {len(days)} slate(s), {days[0]} .. {days[-1]}")
    print(f"integrity {asdict(scorecard.integrity(rows))}")
    print(f"absent from the box (games): {scorecard.absent_names(rows)}")
    head = (
        f"{'cut':26} {'n':>6} {'W-L-P':>14} {'win':>6} {'need':>6} {'base':>6} {'PPV':>6} "
        f"{'NPV':>6} {'ROI':>7} {'ROI 95%':>17} {'CLV':>7} {'beat':>5} {'CLV EV':>7} "
        f"{'pulled':>6} {'prepull':>7}"
    )
    for name, table in scorecard.tables(rows, draws=args.draws).items():
        table = [m for m in table if m.n or name == "buys"]
        if not table:
            continue
        print(f"\n[{name}]\n{head}")
        for m in table:
            print(
                f"{m.label[:26]:26} {m.n:6} {f'{m.wins}-{m.losses}-{m.pushes}':>14} "
                f"{100 * m.win_pct:5.1f}% {100 * m.required_win_pct:5.1f}% "
                f"{100 * m.base_rate:5.1f}% {100 * m.ppv:5.1f}% {100 * m.npv:5.1f}% "
                f"{_pct(m.roi):>7} ({_pct(m.roi_lo):>7},{_pct(m.roi_hi):>7}) "
                f"{100 * m.mean_clv:+6.2f} {100 * m.clv_beat_pct:4.0f}% {_pct(m.mean_clv_ev):>7} "
                f"{m.pulled + m.moved:6} {100 * m.mean_pre_pull_clv:+7.2f}"
            )
    print("\n[calibration: consensus fair]")
    for label, n, p, hit in scorecard.calibration(rows, lambda r: r.fair):
        print(f"{label:10} n={n:6} fair {100 * p:5.1f}% hit {100 * hit:5.1f}%")
    return 0


def _write(path: Path, data: bytes) -> bool:
    try:
        path.write_bytes(data)
    except OSError as exc:
        print(f"  {path.name} not written ({exc})")
        return False
    return True


def _podcast_view(card: SlateCard) -> podcast.PodcastView | None:
    """The slate's podcast bets for the PDF; a failure here never stops the card."""
    try:
        return podcast.view(store_dir(), card)
    except Exception as exc:  # noqa: BLE001 - podcasts are display only
        print(f"  podcast bets left off the card ({exc})")
        return None


def _grade_podcasts(root: Path, days: list[Date]) -> int:
    """Grade the shows' NBA picks on ``days`` into the podcast ledger. Never raises."""
    n = 0
    for d in days:
        try:
            n += len(podcast.grade_day(store_dir(), d, ledger.graded_rows(root, [d])))
        except Exception as exc:  # noqa: BLE001 - podcasts are display only
            print(f"  {d}: podcast picks not graded ({exc})")
    return n


def cmd_podcast(args: argparse.Namespace) -> int:
    """Read the shows' new NBA episodes and grade their picks. Prices nothing.

    ``--no-read`` only grades what has already been read.
    """
    cfg = load_config()
    root = data_dir()
    sync = cfg.state_sync and not args.no_sync
    if sync:
        podcast_state.auto_pull(store_dir())
        state.auto_pull(root, trees=("ledger",))
    if not args.no_read:
        now = datetime.now(timezone.utc)
        rep = read_feeds(
            store_dir(),
            now - timedelta(days=args.days),
            now,
            league=NBA,
            transcribe_missing=not args.no_transcribe,
        )
        print(f"  {rep.by_league.get(NBA, 0)} NBA picks read")
    today = _today()
    days = [today - timedelta(days=k) for k in range(args.days, 0, -1)]
    n = _grade_podcasts(root, days)
    print(f"  {n} NBA podcast picks graded over the last {args.days} day(s)")
    if sync:
        podcast_state.auto_push(store_dir(), f"nba podcasts: through {today - timedelta(days=1)}")
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    """Write the day's package -- workbook, card HTML, PDF -- and optionally email it.

    Built from the ledger passes already recorded (``capture --record``), so it
    spends no Odds API credit and re-renders any priced day. Each artifact is
    guarded on its own: the workbook is written first, a missing WeasyPrint
    costs the PDF only (the HTML is attached instead), and a machine without
    SMTP credentials keeps everything on disk and still exits 0.
    """
    cfg = load_config()
    root = data_dir()
    day = _parse_date(args.date)
    for noisy in ("weasyprint", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if cfg.state_sync and not args.no_sync:
        state.auto_pull(root, trees=("prices", "injuries", "alerts", "ledger"))
    rows = ledger.of_record(ledger.passes(root, day))
    if not rows:
        print(f"{day}: no priced ledger pass to card; record one with `capture --record TAG`")
        return 1
    try:
        fetch = None if args.offline else context.espn_fetch(root)
        ctx = context.gather(root, day, rows, fetch=fetch)
    except Exception as exc:  # noqa: BLE001 - context is colour; the prices go out regardless
        print(f"  context not read ({exc}); card shows prices only")
        ctx = {}
    before = (day - timedelta(days=1)).isoformat()
    graded = ledger.graded_rows(root, _ledger_days(root, None, before))
    card = build_card(rows, day=day, context=ctx, graded=graded, preseason=cfg.preseason)
    if cfg.state_sync and not args.no_sync:
        podcast_state.auto_pull(store_dir())
    page, text = render_html(card, _podcast_view(card)), render_text(card)
    out = output_dir()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"card not written ({exc}); {out} is not a writable directory")
        return 1
    stem = card.stem()
    attachments: list[tuple[str, bytes]] = []
    try:
        workbook = build_workbook(card)
    except Exception as exc:  # noqa: BLE001 - report the failure, keep the card
        print(f"  workbook not built ({exc})")
    else:
        if _write(out / f"{stem}.xlsx", workbook):
            attachments.append((f"{stem}.xlsx", workbook))
    html_ok = _write(out / f"{stem}.html", page.encode("utf-8"))
    try:
        pdf = render_pdf(page)
    except Exception as exc:  # noqa: BLE001 - the PDF is the optional artifact
        print(f"  card PDF not rendered ({exc}); HTML attached instead")
        if html_ok:
            attachments.insert(0, (f"{stem}.html", page.encode("utf-8")))
    else:
        if _write(out / f"{stem}.pdf", pdf):
            attachments.insert(0, (f"{stem}.pdf", pdf))
    if not attachments:
        print(f"card: nothing could be written to {out}")
        return 1
    buys = len(card.buys())
    print(
        f"card: {len(card.games)} games, {len(card.rows)} rows, {buys} buys"
        f" -> {out / stem}.* ({', '.join(name for name, _ in attachments)})"
    )
    if not args.email:
        return 0
    try:
        sent = send_package(
            cfg,
            subject=f"{card.title()} -- {buys} buy{'s' if buys != 1 else ''} [paper]",
            html_body=page,
            text_body=text,
            to=args.to,
            attachments=attachments,
        )
    except EmailNotConfigured as exc:
        print(f"  email not sent ({exc}); artifacts are in {out}")
        return 0
    except Exception as exc:  # noqa: BLE001 - SMTP down: the files are already on disk
        print(f"  email failed ({type(exc).__name__}: {exc}); artifacts are in {out}")
        return 1
    left = f"; too large, left on disk: {', '.join(sent.left_out)}" if sent.left_out else ""
    print(f"  emailed {', '.join(sent.attached)} to {sent.recipient} ({sent.size:,} bytes){left}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """One tick of the news alarm (scheduled every few minutes on game days)."""
    cfg = load_config()
    day = _parse_date(args.date)
    root = data_dir()
    sync = cfg.state_sync and not args.no_sync
    if sync:
        state.auto_pull(root)
    client = _client()
    if client is None:
        return 2
    taken = capture.now_utc()
    written: list[str] = []

    boards = [capture.read_snapshot(p) for p in capture.snapshot_paths(root, day, label="board")]
    prior = alarm.read_alerts(root, day)
    slate, rows = client.fetch_board(
        slate_date=day, horizon_hours=cfg.capture.horizon_hours, captured_at=taken
    )
    games = [g for g in slate.games if g.game_date == day]
    by_id = {g.event_id: g for g in games}
    board = [r for r in rows if r.game_date == day.isoformat()]
    found: list[alarm.Alert] = []
    path = capture.write_snapshot(board, root, day, label="board", captured_at=taken)
    if path is not None:
        written.append(path.name)
        found += alarm.move_alerts(
            alarm.reference(boards, prior), alarm.consensus(board), by_id, cfg.alarm, taken
        )

    feed = espn_injuries.fetch_feed()
    if feed is not None:
        prev_feed = espn_injuries.latest_feed_path(root, day)
        path = espn_injuries.write_feed(feed, root, day, taken)
        if path is not None:
            written.append(path.name)
            if prev_feed is not None:
                found += alarm.feed_alerts(espn_injuries.read_feed(prev_feed), feed, games, taken)

    held = injuries.report_csvs(root, day)
    new_reports = injuries.InjuryClient().capture(root, day)
    written += [p.name for p in new_reports]
    reports = injuries.report_csvs(root, day)
    if held and reports and reports[-1] != held[-1]:
        found += alarm.official_alerts(
            injuries.read_report(held[-1]), injuries.read_report(reports[-1]), games, taken
        )

    now = datetime.now(timezone.utc)
    live = [
        a
        for a in found
        if (g := by_id.get(a.event_id)) is not None
        and (tip := parse_utc(g.start_utc)) is not None
        and tip > now
    ]
    for eid in sorted({a.event_id for a in live}):
        ev = client.fetch_event_markets(
            [by_id[eid]], markets=EVENT_MARKETS, captured_at=taken, max_events=1
        )
        path = capture.write_snapshot(ev, root, day, label=f"alarm-{eid[:12]}", captured_at=taken)
        if path is not None:
            written.append(path.name)
    path = alarm.write_alerts(live, root, day, taken)
    if path is not None:
        written.append(path.name)
    written += _capture_closes(client, root, day, taken)

    future = {
        eid
        for eid, g in by_id.items()
        if (tip := parse_utc(g.start_utc)) is not None and tip > datetime.now(timezone.utc)
    }
    waiting = alarm.unsettled(prior + live) & future
    if waiting:
        first = {e: c for e, c in alarm.consensus(board).items() if e in waiting}

        def poll(events: set[str]) -> dict[str, dict[str, float]]:
            at = capture.now_utc()
            got = client.fetch_event_markets(
                [by_id[e] for e in sorted(events)],
                markets=GAME_MARKETS,
                captured_at=at,
                max_events=len(events),
            )
            snap = capture.write_snapshot(got, root, day, label="settle", captured_at=at)
            if snap is not None:
                written.append(snap.name)
            return alarm.consensus(got)

        done = alarm.settle(waiting, first, poll, cfg.alarm, _sleep)
        stamp = capture.now_utc()
        news = max(a.detected_at for a in prior + live if a.event_id in done) if done else ""
        if stamp <= news and (heard := parse_utc(news)) is not None:
            stamp = (heard + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        calm = alarm.settled_alerts(done, by_id, stamp)
        if calm:
            path = alarm.write_alerts(calm, root, day, calm[0].detected_at)
            if path is not None:
                written.append(path.name)
        live += calm
        still = sorted(waiting - set(done))
    else:
        still = []

    for a in live:
        print(f"ALERT {a.matchup:10} {a.kind:16} {a.detail}")
    for eid in still:
        print(f"PENDING {by_id[eid].matchup:10} line still moving")
    print(
        f"{day}: {len(live)} alert(s), {len(written)} file(s) written; "
        f"credits remaining {client.credits_remaining}"
    )
    if sync and (live or any(n.startswith("close-") for n in written)):
        state.auto_push(root, f"nba watch {day} {taken}")
    return 0


def _days(args: argparse.Namespace) -> list[Date]:
    days: list[Date] = []
    for season in args.season:
        days.extend(history.season_days(season))
    if args.since:
        days = [d for d in days if d >= Date.fromisoformat(args.since)]
    if args.until:
        days = [d for d in days if d <= Date.fromisoformat(args.until)]
    return days


def cmd_history(args: argparse.Namespace) -> int:
    cfg = load_config()
    client = _client()
    if client is None:
        return 2
    anchors = tuple(a.strip() for a in args.anchors.split(",") if a.strip())
    unknown = [a for a in anchors if a not in history.ANCHORS]
    if unknown:
        print(f"unknown anchor(s) {unknown}; choose from {list(history.ANCHORS)}", file=sys.stderr)
        return 2
    markets = tuple(m.strip() for m in args.markets.split(",")) if args.markets else ALL_MARKETS
    report = history.pull(
        client,
        data_dir(),
        _days(args),
        anchors=anchors,
        markets=markets,
        budget=args.budget,
        min_credits=args.min_credits if args.min_credits is not None else cfg.history.min_credits,
    )
    print(json.dumps(asdict(report), indent=2))
    if args.push:
        state.auto_push(data_dir(), "nba history pull", trees=("history",))
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    for season in args.season:
        print(season, json.dumps(history.coverage(data_dir(), history.season_days(season))))
    return 0


def cmd_injuries(args: argparse.Namespace) -> int:
    cfg = load_config()
    root = data_dir()
    sync = cfg.state_sync and not args.no_sync
    day = _parse_date(args.date)
    if sync:
        state.auto_pull(root)
    paths = injuries.InjuryClient().capture(root, day)
    rows = injuries.read_latest(root, day)
    print(f"{day}: archived {len(paths)} new report(s); latest has {len(rows)} rows")
    for row in rows if args.show else []:
        print(f"  {row.matchup:8} {row.team:24} {row.player:28} {row.status:12} {row.reason}")
    if sync and paths:
        state.auto_push(root, f"nba injuries {day}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    for game in ESPNClient().results(day):
        line = (
            f"{game.away} {game.final_away} @ {game.home} {game.final_home}  "
            f"1H {game.h1_away}-{game.h1_home}  OT {game.overtimes}  {game.state}"
        )
        print(line)
        if args.players:
            for p in game.players:
                if not p.dnp:
                    print(
                        f"   {p.team} {p.name:24} {p.minutes:>2}m  {p.points:>2}p "
                        f"{p.threes}x3 {p.rebounds}r {p.assists}a  PRA {p.pra}"
                    )
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    print(
        json.dumps(
            capture.archive_summary(capture.read_day(data_dir(), day)), indent=2, default=str
        )
    )
    return 0


def cmd_boxes(args: argparse.Namespace) -> int:
    root, client, held, fetched = data_dir(), ESPNClient(), 0, 0
    for day in _days(args):
        if day >= _today():
            break
        if boxes.read_results(root, day) is not None:
            held += 1
            continue
        boxes.ensure_results(root, day, client)
        fetched += 1
    print(f"results: {held} day(s) already held, {fetched} fetched")
    return 0


def cmd_fit_over_bias(args: argparse.Namespace) -> int:
    root = data_dir()
    graded: list[overbias.Graded] = []
    missing = 0
    for day in _days(args):
        day_rows = history.history_rows(root, day)
        if not day_rows:
            continue
        games = boxes.read_results(root, day)
        if games is None:
            missing += 1
            continue
        graded.extend(overbias.grade(day_rows, boxes.by_matchup(games)))
    if not graded:
        print("nothing graded (no archived closes with finals); params left unchanged")
        return 1
    gaps = overbias.fit(graded, draws=args.draws)
    n_games = len({g.game for g in graded})
    print(f"graded {len(graded)} Overs over {n_games} games; {missing} day(s) with no results")
    print(f"{'market':11} {'book':11} {'n':>7} {'games':>5} {'gap pts':>8} {'95% CI':>16} applied")
    for g in gaps:
        print(
            f"{g.market:11} {g.book:11} {g.n:7} {g.games:5} {100 * g.gap:+8.2f} "
            f"({100 * g.lo:+6.2f},{100 * g.hi:+6.2f}) {100 * g.applied:+.2f}"
        )
    path = params.write(
        root,
        overbias.NAME,
        overbias.to_payload(
            gaps, seasons=args.season, anchor="close", n=len(graded), games=n_games
        ),
    )
    print(f"wrote {path}")
    if args.push:
        state.auto_push(root, f"nba params {path.stem}", trees=("params",))
    return 0


def _held_finals(root: Path, days: list[Date]) -> list[GameResult]:
    games: list[GameResult] = []
    for day in days:
        games.extend(g for g in boxes.read_results(root, day) or [] if ratings.rateable(g))
    return games


def cmd_fit_ratings(args: argparse.Namespace) -> int:
    root = data_dir()
    days = _days(args)
    games = _held_finals(root, days)
    if not games:
        print("no archived finals with team boxes; params left unchanged")
        return 1
    train, holdout = set(args.train), set(args.holdout or [])
    sched = schedule(games)
    finals = {g.espn_id: g for g in games}
    fitted, sse = rating_fit.fit(games, train, sched)
    b2b = rating_fit.fit_b2b(ratings.replay(games, fitted, sched)[0], finals, sched, train)
    fitted = replace(fitted, b2b_margin=b2b.applied)
    preds, book = ratings.replay(games, fitted, sched)
    lines = rating_fit.closing_lines(root, sorted({g.game_date for g in games}))
    print(f"{len(games)} finals; train {sorted(train)} score MSE {sse:.2f}")
    print(
        f"b2b cost {b2b.cost:+.2f} pts ({b2b.lo:+.2f},{b2b.hi:+.2f}) n={b2b.n} applied {b2b.applied:+.2f}"
    )
    grades: dict[str, list[dict]] = {}
    for label, seasons in (("train", train), ("holdout", holdout)):
        if not seasons:
            continue
        grades[label] = []
        for gr in rating_fit.vs_close(preds, finals, lines, seasons, draws=args.draws):
            grades[label].append(asdict(gr))
            print(
                f"{label:7} {gr.market:6} n={gr.n:5} MAE model {gr.model_mae:6.3f} "
                f"close {gr.close_mae:6.3f}  slope {gr.slope:+.3f} ({gr.lo:+.3f},{gr.hi:+.3f})"
                f"{'  adds information' if gr.adds_information else ''}"
            )
    path = params.write(
        root,
        ratings.NAME,
        {
            "params": ratings.params_payload(fitted),
            "train": sorted(train),
            "holdout": sorted(holdout),
            "games": len(games),
            "score_mse": sse,
            "b2b": asdict(b2b),
            "vs_close": grades,
            "league": {"ppp": book.league_ppp, "pace": book.league_pace, "hca": book.hca},
        },
    )
    print(f"wrote {path}")
    if args.push:
        state.auto_push(root, f"nba params {path.stem}", trees=("params",))
    return 0


def _with_reasons(root: Path, games: list[GameResult]) -> list[GameResult]:
    """Re-read each final's box from its raw summary, which carries the DNP reasons."""
    out: list[GameResult] = []
    for g in games:
        raw = boxes.read_raw_summary(root, g.game_date, g.espn_id)
        out.append(replace(g, players=parse_box(raw)) if raw else g)
    return out


def cmd_fit_minutes(args: argparse.Namespace) -> int:
    root = data_dir()
    games = _with_reasons(root, _held_finals(root, _days(args)))
    if not games:
        print("no archived finals; params left unchanged")
        return 1
    train, holdout = set(args.train), set(args.holdout or [])
    fitted, err = minutes_fit.fit(games, train)
    rows, _ = minutes.replay(games, fitted)
    print(f"{len(games)} finals; train {sorted(train)} minutes MAE {err:.3f}")
    print(" ".join(f"{k}={v}" for k, v in minutes.params_payload(fitted).items()))
    grades: dict[str, list[dict]] = {}
    for label, seasons in (("train", train), ("holdout", holdout)):
        if not seasons:
            continue
        grades[label] = []
        for gr in minutes_fit.grade(rows, seasons, draws=args.draws):
            grades[label].append(asdict(gr))
            print(
                f"{label:7} {gr.group:9} n={gr.n:6} MAE model {gr.model_mae:.3f} "
                f"last-{minutes_fit.RECENT} {gr.recent_mae:.3f} gain {gr.gain:+.3f} "
                f"({gr.lo:+.3f},{gr.hi:+.3f})"
            )
    band = minutes_fit.spread(rows, holdout or train)
    for b in band:
        print(
            f"projected {b['lo']:>2.0f}-{b['hi']:<2.0f} n={b['n']:6.0f} "
            f"miss mean {b['mean']:+.2f} sd {b['sd']:.2f}"
        )
    path = params.write(
        root,
        minutes.NAME,
        {
            "params": minutes.params_payload(fitted),
            "train": sorted(train),
            "holdout": sorted(holdout),
            "games": len(games),
            "train_mae": err,
            "vs_recent": grades,
            "spread": band,
        },
    )
    print(f"wrote {path}")
    if args.push:
        state.auto_push(root, f"nba params {path.stem}", trees=("params",))
    return 0


def cmd_ratings(args: argparse.Namespace) -> int:
    root = data_dir()
    as_of = _parse_date(args.date)
    held = params.latest(root, ratings.NAME)
    fitted = ratings.RatingParams.from_dict(held["params"]) if held else ratings.RatingParams()
    print(f"params {held['version'] if held else 'unfitted defaults'}; games before {as_of}")
    games = [g for g in _held_finals(root, _days(args)) if g.game_date < as_of]
    _, book = ratings.replay(games, fitted)
    print(f"{'team':4} {'off':>7} {'def':>7} {'net':>6} {'pace':>6} {'gp':>3}")
    for r in book.table():
        print(
            f"{r['team']:4} {r['off']:7.2f} {r['def']:7.2f} {r['net']:+6.2f} "
            f"{r['pace']:6.2f} {r['games']:3}"
        )
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    root = data_dir()
    day = _parse_date(args.date)
    fitted = params.latest(root, overbias.NAME)
    shift = overbias.shifts(fitted)
    sels = [s for s in selections(capture.read_day(root, day)) if args.market in ("", s.market)]
    print(
        f"{day}: {len(sels)} selection(s); over_bias {fitted['version'] if fitted else 'unfitted'}"
    )
    for s in sels:
        fair = overbias.adjusted_fair(s, shift)
        if fair is None or s.exec_american is None:
            continue
        line = (
            "" if s.line is None else f"{s.line:+g}" if s.market.endswith("ats") else f"{s.line:g}"
        )
        print(
            f"{s.matchup:10} {s.market:10} {s.entity[:20]:20} {s.side:6} {line:>6} "
            f"fair {fair:.3f} ({s.paired_books} bk)  {s.exec_book} {s.exec_american:+.0f} "
            f"{s.exec_books}-book  vs fair {100 * ev_per_unit(fair, s.exec_american):+.1f}%"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba-engine")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="archive the current NBA board")
    cap.add_argument("--date", help="slate date (ET), default today")
    cap.add_argument("--board-only", action="store_true", help="skip per-event markets")
    cap.add_argument("--close", action="store_true", help="only games tipping in the close window")
    cap.add_argument("--max-events", type=int, default=16)
    cap.add_argument("--cache-ttl", type=int, default=0, help="seconds; 0 = always fetch fresh")
    cap.add_argument("--no-sync", action="store_true", help="do not pull/push engine-state")
    cap.add_argument("--record", default="", metavar="TAG", help="write a priced ledger pass")
    cap.set_defaults(func=cmd_capture)

    wa = sub.add_parser("watch", help="one news-alarm tick: board, injury feeds, closes")
    wa.add_argument("--date", help="slate date (ET), default today")
    wa.add_argument("--no-sync", action="store_true")
    wa.set_defaults(func=cmd_watch)

    hi = sub.add_parser("history", help="pull historical quotes (spends shared credits)")
    hi.add_argument("--season", action="append", required=True, help="e.g. 2025-26 (repeatable)")
    hi.add_argument("--anchors", default="close", help=f"comma list of {list(history.ANCHORS)}")
    hi.add_argument("--markets", help="comma list of provider keys (default all eleven)")
    hi.add_argument("--budget", type=int, required=True, help="max credits this run may spend")
    hi.add_argument(
        "--min-credits", type=int, help="balance floor (default NBAE_HISTORY_MIN_CREDITS)"
    )
    hi.add_argument("--since", help="first slate date to pull")
    hi.add_argument("--until", help="last slate date to pull")
    hi.add_argument("--push", action="store_true", help="push the archive to engine-state")
    hi.set_defaults(func=cmd_history)

    co = sub.add_parser("coverage", help="what the historical archive holds")
    co.add_argument("--season", action="append", required=True)
    co.set_defaults(func=cmd_coverage)

    inj = sub.add_parser("injuries", help="archive today's official injury reports")
    inj.add_argument("--date")
    inj.add_argument("--show", action="store_true", help="print the latest report's rows")
    inj.add_argument("--no-sync", action="store_true")
    inj.set_defaults(func=cmd_injuries)

    res = sub.add_parser("results", help="ESPN finals by quarter")
    res.add_argument("--date")
    res.add_argument("--players", action="store_true")
    res.set_defaults(func=cmd_results)

    arc = sub.add_parser("archive", help="summarise captured prices for a date")
    arc.add_argument("--date")
    arc.set_defaults(func=cmd_archive)

    bx = sub.add_parser("boxes", help="archive ESPN finals and box lines (free)")
    bx.add_argument("--season", action="append", required=True)
    bx.add_argument("--since")
    bx.add_argument("--until")
    bx.set_defaults(func=cmd_boxes)

    ob = sub.add_parser("fit-over-bias", help="fit the Over bias on archived closes + finals")
    ob.add_argument("--season", action="append", required=True)
    ob.add_argument("--since")
    ob.add_argument("--until")
    ob.add_argument("--draws", type=int, default=400, help="game bootstrap draws")
    ob.add_argument("--push", action="store_true", help="push params/ to engine-state")
    ob.set_defaults(func=cmd_fit_over_bias)

    fr = sub.add_parser("fit-ratings", help="fit team-rating speeds on a replay; grade vs close")
    fr.add_argument("--season", action="append", required=True, help="finals to load")
    fr.add_argument("--since")
    fr.add_argument("--until")
    fr.add_argument("--train", type=int, action="append", required=True, help="start year")
    fr.add_argument("--holdout", type=int, action="append", help="start year, never fitted")
    fr.add_argument("--draws", type=int, default=1000, help="slate-date bootstrap draws")
    fr.add_argument("--push", action="store_true", help="push params/ to engine-state")
    fr.set_defaults(func=cmd_fit_ratings)

    fm = sub.add_parser("fit-minutes", help="fit the minutes book on a replay; grade vs recent avg")
    fm.add_argument("--season", action="append", required=True, help="finals to load")
    fm.add_argument("--since")
    fm.add_argument("--until")
    fm.add_argument("--train", type=int, action="append", required=True, help="start year")
    fm.add_argument("--holdout", type=int, action="append", help="start year, never fitted")
    fm.add_argument("--draws", type=int, default=1000, help="slate-date bootstrap draws")
    fm.add_argument("--push", action="store_true", help="push params/ to engine-state")
    fm.set_defaults(func=cmd_fit_minutes)

    rt = sub.add_parser("ratings", help="team ratings going into a date")
    rt.add_argument("--season", action="append", required=True)
    rt.add_argument("--since")
    rt.add_argument("--until")
    rt.add_argument("--date", help="ratings before this slate date (default today)")
    rt.set_defaults(func=cmd_ratings)

    bd = sub.add_parser("board", help="consensus fair vs the DraftKings/BetMGM price")
    bd.add_argument("--date")
    bd.add_argument("--market", default="", help="e.g. game_ml, pl_pts")
    bd.set_defaults(func=cmd_board)

    gr = sub.add_parser("grade", help="settle a slate's ledger and stamp its pre-tip closes")
    gr.add_argument("--date", help="slate date (ET), default yesterday")
    gr.add_argument("--no-sync", action="store_true", help="do not pull/push engine-state")
    gr.set_defaults(func=cmd_grade)

    au = sub.add_parser("audit", help="the graded ledger's tables")
    au.add_argument("--since")
    au.add_argument("--until")
    au.add_argument("--draws", type=int, default=400)
    au.add_argument("--no-sync", action="store_true", help="do not pull engine-state")
    au.set_defaults(func=cmd_audit)

    cd = sub.add_parser("card", help="slate PDF + workbook (+ email) from the day's ledger")
    cd.add_argument("--date", help="slate date (ET), default today")
    cd.add_argument("--email", action="store_true", help="email the PDF and workbook")
    cd.add_argument("--to", help="recipient (default NBAE_EMAIL_TO / GMAIL_USER)")
    cd.add_argument("--no-sync", action="store_true", help="do not pull engine-state")
    cd.add_argument(
        "--offline", action="store_true", help="context from disk only (no ESPN finals fetch)"
    )
    cd.set_defaults(func=cmd_card)

    pod = sub.add_parser(
        "podcast", help="read the shows' NBA picks and grade them (prices nothing)"
    )
    pod.add_argument("--days", type=int, default=3, help="episodes and slates this far back")
    pod.add_argument("--no-read", action="store_true", help="only grade picks already read")
    pod.add_argument("--no-transcribe", action="store_true", help="skip episodes not transcribed")
    pod.add_argument("--no-sync", action="store_true", help="skip the engine-state pull/push")
    pod.set_defaults(func=cmd_podcast)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
