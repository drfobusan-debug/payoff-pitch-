"""``nhl-engine`` command line.

Phase 0 commands:

* ``capture`` -- archive today's featured board plus per-event period/prop
  markets. Idempotent: an unchanged board writes nothing.
* ``results`` -- pull official finals (per-period, OT/SO, starters) for a date.
* ``archive`` -- summarise what has been captured for a date.
* ``book-rules`` -- list, add or check per-book settlement rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date as Date
from datetime import datetime
from zoneinfo import ZoneInfo

from nhl_engine import state
from nhl_engine.config import cache_dir, data_dir, load_config
from nhl_engine.data import capture
from nhl_engine.data.book_rules import BookRule, BookRules, rules_path
from nhl_engine.data.nhlapi import NHLAPIClient
from nhl_engine.data.oddsapi import OddsAPIClient

log = logging.getLogger("nhl_engine")
SLATE_TZ = ZoneInfo("America/New_York")


def _today() -> Date:
    return datetime.now(SLATE_TZ).date()


def _parse_date(raw: str | None) -> Date:
    return Date.fromisoformat(raw) if raw else _today()


def cmd_capture(args: argparse.Namespace) -> int:
    cfg = load_config()
    slate_date = _parse_date(args.date)
    root = data_dir()
    if cfg.state_sync and not args.no_sync:
        state.auto_pull(root)
    client = OddsAPIClient(
        cfg.creds.odds_api_key, cache_dir=cache_dir() / "oddsapi", cache_ttl=args.cache_ttl
    )
    if not client.available():
        print(
            "no Odds API key (THE_ODDS_API_KEY / ODDS_API_KEY); nothing captured", file=sys.stderr
        )
        return 2
    taken = capture.now_utc()
    slate, rows = client.fetch_board(
        slate_date=slate_date, horizon_hours=cfg.capture.horizon_hours, captured_at=taken
    )
    written = [capture.write_snapshot(rows, root, slate_date, label="board", captured_at=taken)]
    if not args.board_only and slate.games:
        event_rows = client.fetch_event_markets(
            slate.games,
            captured_at=taken,
            max_events=min(cfg.capture.max_events, args.max_events),
        )
        written.append(
            capture.write_snapshot(event_rows, root, slate_date, label="events", captured_at=taken)
        )
    else:
        event_rows = []
    wrote = [p.name for p in written if p is not None]
    print(
        json.dumps(
            {
                "slate_date": slate_date.isoformat(),
                "games": len(slate.games),
                "board_rows": len(rows),
                "event_rows": len(event_rows),
                "written": wrote,
                "credits_remaining": client.credits_remaining,
            }
        )
    )
    if cfg.state_sync and not args.no_sync and wrote:
        state.auto_push(root, f"nhl: capture {slate_date} {taken}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    client = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")
    games = client.schedule(day)
    out = []
    for game in games:
        res = client.result(game)
        if res is None:
            continue
        out.append(
            {
                "game": game.matchup,
                "nhl_game_id": res.nhl_game_id,
                "state": res.state,
                "decided": res.decided,
                "periods": [asdict(p) for p in res.periods],
                "reg": [res.reg_away, res.reg_home],
                "final": [res.final_away, res.final_home],
                "starters": [res.away_starter, res.home_starter],
            }
        )
    print(json.dumps({"date": day.isoformat(), "games": out}, indent=None if args.compact else 2))
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    day = _parse_date(args.date)
    rows = capture.read_day(data_dir(), day)
    summary = capture.archive_summary(rows)
    summary["snapshots"] = [p.name for p in capture.snapshot_paths(data_dir(), day)]
    print(json.dumps(summary, indent=2))
    return 0


def cmd_book_rules(args: argparse.Namespace) -> int:
    path = rules_path(data_dir())
    rules = BookRules.load(path)
    today = _today()
    if args.add:
        book, market, ot_rule = args.add
        rules.add(
            BookRule(
                book=book,
                market=market,
                ot_rule=ot_rule,
                verified_on=today.isoformat(),
                source=args.source,
            )
        )
        rules.save(path)
        print(f"recorded {book} {market} -> {ot_rule} (verified {today})")
        return 0
    if args.check:
        book, market = args.check
        print(rules.resolve(book, market, today))
        return 0
    stale = {(r.book, r.market) for r in rules.stale(today)}
    for r in rules.all():
        flag = "  EXPIRED" if (r.book, r.market) in stale else ""
        print(f"{r.book:16} {r.market:16} {r.ot_rule:12} verified {r.verified_on}{flag}")
    if not len(rules):
        print(f"no book rules recorded ({path}); every team-total quote is settlement_unverified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nhl-engine")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="archive the current NHL board")
    cap.add_argument("--date", help="slate date (ET), default today")
    cap.add_argument("--board-only", action="store_true", help="skip per-event markets")
    cap.add_argument("--max-events", type=int, default=64)
    cap.add_argument("--cache-ttl", type=int, default=0, help="seconds; 0 = always fetch fresh")
    cap.add_argument("--no-sync", action="store_true", help="do not pull/push engine-state")
    cap.set_defaults(func=cmd_capture)

    res = sub.add_parser("results", help="official finals from the NHL API")
    res.add_argument("--date")
    res.add_argument("--compact", action="store_true")
    res.set_defaults(func=cmd_results)

    arc = sub.add_parser("archive", help="summarise captured prices for a date")
    arc.add_argument("--date")
    arc.set_defaults(func=cmd_archive)

    br = sub.add_parser("book-rules", help="per-book settlement rules")
    br.add_argument("--add", nargs=3, metavar=("BOOK", "MARKET", "OT_RULE"))
    br.add_argument("--source", default="")
    br.add_argument("--check", nargs=2, metavar=("BOOK", "MARKET"))
    br.set_defaults(func=cmd_book_rules)
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
