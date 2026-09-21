"""``live-edge`` -- watch-only in-play edge detector for NFL and CFB.

live-edge tick  --sport nfl            one poll: score + board -> tape, flags, grading
live-edge run   --sport nfl --interval 60 --hours 8
live-edge report                       flag ledger summary
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

from live_edge.engine import LiveEngine, Thresholds, summarize
from live_edge.espn import fetch_scoreboard
from live_edge.oddsapi import LiveOddsClient

log = logging.getLogger("live_edge")


def _api_key() -> str | None:
    return os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")


def _thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        ml_edge=args.ml_edge,
        spread_edge=args.spread_edge,
        total_edge=args.total_edge,
        min_ticks=args.min_ticks,
        min_books=args.min_books,
    )


def do_tick(engine: LiveEngine, client: LiveOddsClient, *, quiet: bool = False) -> int:
    games = fetch_scoreboard(engine.sport)
    if not games:
        print("ESPN: no games on the scoreboard")
        return 0
    live = [g for g in games if g.status == "in"]
    pre = [g for g in games if g.status == "pre"]
    # Only spend credits when there is something to price or a prior to capture.
    events = client.fetch(engine.sport) if (live or pre) else []
    rep = engine.tick(games, events)
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    credits = (
        f", {client.credits_remaining} credits left" if client.credits_remaining is not None else ""
    )
    if not quiet or rep.flags:
        print(
            f"[{stamp}] {engine.sport}: {rep.live} live, {rep.priced} priced, "
            f"{len(rep.flags)} new flag(s), {rep.graded} graded{credits}"
        )
    for m in rep.unmatched:
        log.info("no board match for %s", m)
    for m in rep.no_prior:
        log.info("no pregame prior captured for %s (started before the first tick)", m)
    for f in rep.flags:
        line = "" if f.line is None else f" {f.line:+g}" if f.market == "spread" else f" {f.line:g}"
        print(
            f"  FLAG {f.matchup} | {f.market} {f.side}{line} @ {f.book} {f.price:+.0f} | "
            f"model {f.p_model:.1%} vs market {f.p_market:.1%} (edge {f.edge:+.1%}, ev {f.ev:+.1%}) | "
            f"Q{f.period} {f.clock // 60}:{f.clock % 60:02d} {f.away_score}-{f.home_score}"
        )
    return len(live)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="live-edge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sport", choices=("nfl", "cfb"), default="nfl")
    ap.add_argument("--ml-edge", type=float, default=Thresholds.ml_edge)
    ap.add_argument("--spread-edge", type=float, default=Thresholds.spread_edge)
    ap.add_argument("--total-edge", type=float, default=Thresholds.total_edge)
    ap.add_argument("--min-ticks", type=int, default=Thresholds.min_ticks)
    ap.add_argument("--min-books", type=int, default=Thresholds.min_books)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick")
    run = sub.add_parser("run")
    run.add_argument("--interval", type=int, default=60, help="seconds between ticks")
    run.add_argument("--hours", type=float, default=8.0, help="stop after this long")
    run.add_argument(
        "--idle-exit",
        type=int,
        default=5,
        help="stop after this many ticks with no live game and none due within an hour",
    )
    sub.add_parser("report")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    engine = LiveEngine(args.sport, thresholds=_thresholds(args))

    if args.cmd == "report":
        print(summarize(engine.load_flags()))
        return 0

    client = LiveOddsClient(_api_key())
    if not client.available():
        print("THE_ODDS_API_KEY is not set; cannot price the live board", file=sys.stderr)
        return 2

    if args.cmd == "tick":
        do_tick(engine, client)
        return 0

    deadline = time.monotonic() + args.hours * 3600
    idle = 0
    while time.monotonic() < deadline:
        try:
            n_live = do_tick(engine, client, quiet=True)
        except Exception as exc:  # noqa: BLE001 - the loop must survive a bad tick
            log.warning("tick failed: %s", exc)
            n_live = 0
        idle = 0 if n_live else idle + 1
        if idle >= args.idle_exit and not _kickoff_soon(engine.sport):
            print("no live games and none within the hour; stopping")
            break
        time.sleep(args.interval)
    print(summarize(engine.load_flags()))
    return 0


def _kickoff_soon(sport: str, within_secs: int = 3600) -> bool:
    now = datetime.now(timezone.utc)
    for g in fetch_scoreboard(sport):
        if g.status != "pre" or not g.kickoff:
            continue
        try:
            ko = datetime.fromisoformat(g.kickoff.replace("Z", "+00:00"))
        except ValueError:
            continue
        if 0 <= (ko - now).total_seconds() <= within_secs:
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
