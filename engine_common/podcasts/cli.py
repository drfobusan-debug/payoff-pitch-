"""``payoff-podcasts``: read every show once, and audit every handicapper.

``read`` transcribes and extracts the registry's new episodes into the shared
store, every league at once. ``audit`` prints (and optionally writes as Excel)
each show's and host's record by league, market and kind. Each engine's own
``podcast`` command places and grades its league's picks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine_common.podcasts import audit, state
from engine_common.podcasts.episodes import store_dir
from engine_common.podcasts.run import read_feeds
from engine_common.podcasts.shows import BY_KEY, LEAGUES, SHOWS

DEFAULT_DAYS = 9


def _since(arg: str | None, days: int) -> datetime:
    if arg:
        return datetime.fromisoformat(arg).replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(days=days)


def cmd_read(args: argparse.Namespace) -> int:
    root = store_dir()
    if args.sync:
        state.auto_pull(root)
    shows = [BY_KEY[k] for k in args.shows.split(",")] if args.shows else list(SHOWS)
    read_feeds(
        root,
        _since(args.since, args.days),
        datetime.now(timezone.utc),
        shows=shows,
        league=args.league,
        transcribe_missing=not args.no_transcribe,
    )
    if args.sync:
        state.auto_push(root, "podcasts: read")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    root = store_dir()
    if args.sync:
        state.auto_pull(root)
    rows = audit.all_ledgers(root)
    if not rows:
        print(f"no graded podcast picks in {root}")
        return 0
    print(audit.table(rows, league=args.league))
    if args.xlsx:
        out = Path(args.xlsx).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(audit.workbook(rows))
        print(f"-> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="payoff-podcasts", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    rd = sub.add_parser("read", help="transcribe and extract new episodes, every league")
    rd.add_argument("--since", help="ISO date to read episodes from")
    rd.add_argument("--days", type=int, default=DEFAULT_DAYS, help="or this many days back")
    rd.add_argument("--shows", help="comma-separated show keys (default: all)")
    rd.add_argument("--league", choices=LEAGUES, help="only episodes that may carry it")
    rd.add_argument("--no-transcribe", action="store_true", help="only use saved transcripts")
    rd.add_argument("--no-sync", dest="sync", action="store_false")
    rd.set_defaults(func=cmd_read)
    au = sub.add_parser("audit", help="every handicapper's record by league and market")
    au.add_argument("--league", choices=LEAGUES)
    au.add_argument("--xlsx", help="also write the audit workbook here")
    au.add_argument("--no-sync", dest="sync", action="store_false")
    au.set_defaults(func=cmd_audit)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
