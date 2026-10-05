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

Nothing here prices a game: the model, card, ledger and audit come later
(docs/nba/master_plan.md §13).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nba_engine import alarm, state
from nba_engine.config import cache_dir, data_dir, load_config
from nba_engine.data import boxes, capture, espn_injuries, history, injuries
from nba_engine.data.capture import ALL_MARKETS, EVENT_MARKETS, GAME_MARKETS
from nba_engine.data.espn import ESPNClient
from nba_engine.data.oddsapi import SLATE_TZ, OddsAPIClient, parse_utc
from nba_engine.market import overbias, params
from nba_engine.market.board import ev_per_unit, selections

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
    print(
        f"{slate_date}: wrote {len(written)} snapshot(s) {written}; "
        f"credits remaining {client.credits_remaining}"
    )
    if sync and written:
        state.auto_push(root, f"nba capture {slate_date} {taken}")
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

    bd = sub.add_parser("board", help="consensus fair vs the DraftKings/BetMGM price")
    bd.add_argument("--date")
    bd.add_argument("--market", default="", help="e.g. game_ml, pl_pts")
    bd.set_defaults(func=cmd_board)
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
