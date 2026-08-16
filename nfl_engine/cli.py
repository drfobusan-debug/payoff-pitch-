"""The week, as commands: `capture`, `price`, `close`, `grade`, `report`, plus
`replay` and the unattended `job`.

In the order a week actually happens: archive the board; price it and write every
selection *and rejection* to the ledger; archive and stamp the closing number so
CLV can be scored; settle against final scores; read the record back with PPV/NPV
against the base rate.

**Nothing here can stake money.** There is no stake, bankroll or Kelly argument
anywhere in the engine, and every ledger row is written ``mode=paper``. That is
not a flag that could be flipped by accident -- placing a bet would require code
that does not exist. `price` prints the fact on every run, so a screenshot of a
run can never be mistaken for a bet slip.

Two properties the dry run needs that the phase-5 commands did not have:

**Pricing appends; it never rewrites.** Re-running on a moved board adds only
new positions and keeps the first price seen, because keeping the latest number
would hand the paper record the best price of the week in hindsight.

**What gets priced is what got archived.** Every fetch writes a timestamped
snapshot before anything is priced, so a recommendation can be checked against the
board that existed at the time rather than against a memory of it.

`replay` runs a played week at its closing prices through these same functions,
which is the only way to exercise grading, CLV and the report out of season.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

from nfl_engine import replay as replay_mod
from nfl_engine.audit.ledger import (
    PAPER,
    LedgerEntry,
    apply_close,
    entry_from_bet,
    grade,
    load_ledger,
    market_metrics,
    merge_ledger,
    metrics,
    screen_metrics,
    tier_metrics,
    update_ledger,
)
from nfl_engine.config import data_dir, load_config
from nfl_engine.data import capture, nflverse
from nfl_engine.data.oddsapi import Board, OddsAPIClient
from nfl_engine.market.screens import tier_of
from nfl_engine.models.drives import DriveSim
from nfl_engine.pipeline import price_slate, slate_buys
from nfl_engine.schemas import Game

log = logging.getLogger(__name__)

LEDGER_NAME = "nfl_ledger.csv"
PAPER_BANNER = "paper only: no stake is placed and no bankroll exists in this engine"


def ledger_path(root: Path | None = None) -> Path:
    return (root or data_dir()) / LEDGER_NAME


def _client() -> OddsAPIClient:
    config = load_config()
    return OddsAPIClient(config.creds.odds_api_key, cache_dir=None)


def current_week(today: Date | None = None) -> tuple[int, int, Date]:
    """Season, week and that week's first kickoff date, from the schedule itself.

    Taken from nflverse rather than from the calendar because the NFL week is not
    a fixed seven days: week 1 starts on a Thursday, the international games move
    the window, and a hard-coded offset silently prices the wrong week.
    """
    day = today or Date.today()
    games = nflverse.games()
    upcoming = games[games.gameday.astype(str) >= day.isoformat()]
    frame = upcoming if len(upcoming) else games
    row = frame.sort_values("gameday").iloc[0]
    season, week = int(row.season), int(row.week)
    week_games = games[(games.season == season) & (games.week == week)]
    first = min(str(value) for value in week_games.gameday)
    return season, week, Date.fromisoformat(first)


@dataclass
class Fetched:
    season: int
    week: int
    captured_at: str
    games: list[Game]
    board: Board
    archived: Path | None = None


def _fetch(days: int, *, kind: str = capture.GAME_KIND, archive: bool = True) -> Fetched:
    """Pull the board and archive it before a probability is formed."""
    client = _client()
    season, week, first_day = current_week()
    taken = capture.stamp()
    if not client.available():
        log.warning("no Odds API key: nothing to fetch")
        return Fetched(season, week, taken, [], {})
    slate, board = client.fetch_board(
        season=season, week=week, first_day=first_day, days=days
    )
    games = list(slate.games)
    rows = capture.rows_from_board(
        board,
        season=season,
        week=week,
        captured_at=taken,
        dates={game.matchup(): game.game_date.isoformat() for game in games},
        event_ids={game.matchup(): game.game_id for game in games},
    )
    written = capture.write_snapshot(rows, season=season, week=week, kind=kind) if archive else None
    if rows:
        print(f"{capture.archive_summary(rows)}")
        print(f"  archived {written.name if written else '(unchanged: board had not moved)'}")
    if client.credits_remaining is not None:
        print(f"  Odds API credits remaining: {client.credits_remaining}")
    return Fetched(season, week, taken, games, board, written)


def cmd_capture(args: argparse.Namespace) -> int:
    """Archive prices and nothing else -- the command that has to run from now on.

    Game prices are recoverable from nflverse afterwards; prop prices are not
    recoverable from anywhere, at any price, which is why this runs through the
    preseason even though the props layer is blocked and nothing is bet.
    """
    fetched = _fetch(args.days)
    if args.props and fetched.games:
        client = _client()
        rows = client.fetch_props(
            fetched.games, captured_at=fetched.captured_at, max_events=args.max_events
        )
        path = capture.write_snapshot(
            rows, season=fetched.season, week=fetched.week, kind=capture.PROP_KIND
        )
        players = len({row.player for row in rows})
        markets = len({row.market for row in rows})
        print(f"props: {len(rows)} quotes, {players} players, {markets} markets")
        print(f"  archived {path.name if path else '(unchanged)'}")
    return 0


def _ledger_rows(pricings: list, captured_at: str) -> list[LedgerEntry]:
    return [
        entry_from_bet(
            bet,
            season=pricing.game.season,
            week=pricing.game.week,
            date=pricing.game.game_date.isoformat(),
            captured_at=captured_at,
            mode=PAPER,
        )
        for pricing in pricings
        for bet in pricing.bets
    ]


def cmd_price(args: argparse.Namespace) -> int:
    fetched = _fetch(args.days)
    if not fetched.games:
        return 0
    pricings = price_slate(fetched.games, fetched.board, sim=DriveSim(n_sims=args.sims))
    entries = _ledger_rows(pricings, fetched.captured_at)
    added = merge_ledger(ledger_path(), entries) if args.write else []
    buys = slate_buys(pricings)
    print(f"{len(entries)} selections, {len(buys)} survive the screens [{PAPER_BANNER}]")
    if args.write:
        print(f"  {len(added)} new ledger rows ({len(entries) - len(added)} already held)")
    for bet in buys[: args.top]:
        fair_ev = bet.ev_fair or 0.0
        print(
            f"  {bet.matchup:12s} {bet.label():14s} {bet.american:+6.0f} {bet.book:14s}"
            f" model {bet.model_prob:.3f} fair {bet.fair_prob or 0.0:.3f}"
            f" exec EV {fair_ev:+.3f} [{tier_of(bet).value}]"
        )
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    """Re-fetch the board and stamp the closing number on ungraded rows."""
    path = ledger_path()
    entries = load_ledger(path)
    if not entries:
        print("empty ledger")
        return 0
    board = _fetch(args.days, kind=capture.CLOSE_KIND).board
    stamped = 0
    for entry in entries:
        if entry.close_odds is not None or entry.result:
            continue
        quote = _closing_quote(board, entry)
        if quote is None:
            continue
        apply_close(entry, quote[0], quote[1])
        stamped += 1
    if args.write:
        update_ledger(path, entries)
    print(f"closing prices stamped on {stamped} rows")
    return 0


def _closing_quote(board: dict, entry: LedgerEntry) -> tuple[float, float | None] | None:
    odds = board.get(entry.matchup)
    if odds is None:
        return None
    if entry.market == "moneyline":
        quotes = odds.ml.get(entry.side, [])
    elif entry.market == "spread" and entry.line is not None:
        home_point = entry.line if entry.side == entry.matchup.split(" @ ")[-1] else -entry.line
        quotes = odds.spreads.get(home_point, {}).get(entry.side, [])
    elif entry.market == "total" and entry.line is not None:
        quotes = odds.totals.get(entry.line, {}).get(entry.side, [])
    else:
        return None
    same_book = [q for q in quotes if q.book == entry.book] or quotes
    if not same_book:
        return None
    quote = same_book[0]
    return (quote.american, quote.opposite_american)


def cmd_grade(args: argparse.Namespace) -> int:
    path = ledger_path()
    entries = load_ledger(path)
    if not entries:
        print("empty ledger")
        return 0
    scores = _final_scores(args.season)
    graded = 0
    for entry in entries:
        if entry.result:
            continue
        key = (entry.matchup, entry.date)
        final = scores.get(key)
        if final is None:
            continue
        home = entry.matchup.split(" @ ")[-1]
        grade(entry, final[0], final[1], home=home)
        graded += 1
    if args.write:
        update_ledger(path, entries)
    print(f"graded {graded} rows")
    return 0


def _final_scores(season: int | None) -> dict[tuple[str, str], tuple[int, int]]:
    games = nflverse.games()
    games = games[games.home_score.notna() & games.away_score.notna()]
    if season:
        games = games[games.season == season]
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for row in games.itertuples():
        matchup = f"{row.away_team} @ {row.home_team}"
        out[(matchup, str(row.gameday))] = (int(row.home_score), int(row.away_score))
    return out


def cmd_replay(args: argparse.Namespace) -> int:
    """Run played weeks at their closing prices through the live functions.

    The board has one book, so the de-vigged consensus is the price taken and the
    execution edge is zero: every replayed row is a Pass, and that is the right
    answer rather than a bug. What is being tested is the machinery -- append-once
    ledger writes, grading on the side taken, CLV against the price struck, and the
    PPV/NPV report over real outcomes.
    """
    weeks = replay_mod.played_weeks(args.season, args.weeks)
    if not weeks:
        print(f"no played games for {args.season}")
        return 0
    path = ledger_path()
    sim = DriveSim(n_sims=args.sims)
    priced = added = graded = closed = 0
    for week in weeks:
        taken = capture.stamp()
        if args.archive:
            capture.write_snapshot(
                week.quote_rows(taken),
                season=week.season,
                week=week.week,
                kind=capture.CLOSE_KIND,
            )
        pricings = price_slate(week.games, week.board, sim=sim)
        entries = _ledger_rows(pricings, taken)
        priced += len(entries)
        for entry in entries:
            quote = _closing_quote(week.board, entry)
            if quote is not None:
                apply_close(entry, quote[0], quote[1])
                closed += 1
            final = week.finals.get(entry.matchup)
            if final is not None:
                grade(entry, final[0], final[1], home=entry.matchup.split(" @ ")[-1])
                graded += 1
        if args.write:
            added += len(merge_ledger(path, entries))
    print(
        f"{args.season}: {len(weeks)} weeks, {priced} selections priced, "
        f"{closed} closed, {graded} graded [{PAPER_BANNER}]"
    )
    if args.write:
        print(f"  {added} new ledger rows ({priced - added} already held)")
    return 0


def cmd_job(args: argparse.Namespace) -> int:
    """The unattended weekly run: capture, price, close, grade, report.

    One entry point for cron or a double-click, so the archive accrues and the
    ledger settles without anyone remembering the order. Each step is the same
    function the individual command calls, and a step with nothing to do is not an
    error -- in the off-season the whole job is a no-op that still exits 0.
    """
    steps = (
        ("capture", cmd_capture),
        ("price", cmd_price),
        ("close", cmd_close),
        ("grade", cmd_grade),
        ("report", cmd_report),
    )
    for name, func in steps:
        print(f"== {name}")
        func(args)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    entries = load_ledger(ledger_path())
    if not entries:
        print("empty ledger")
        return 0
    rows = [
        *tier_metrics(entries),
        *market_metrics(entries),
        *screen_metrics(entries),
    ]
    if args.all:
        rows.append(metrics(entries, lambda e: True, "ALL"))
    header = (
        f"{'':26s} {'n':>5s} {'win%':>7s} {'need':>7s} {'PPV+':>7s} {'NPV+':>7s}"
        f" {'ROI':>7s} {'units':>8s} {'CLV':>8s}"
    )
    print(header)
    for row in rows:
        print(
            f"{row.label:26s} {row.n:5d} {row.win_pct:7.4f} {row.required_win_pct:7.4f}"
            f" {row.ppv_lift:+7.4f} {row.npv_lift:+7.4f} {row.roi:+7.4f}"
            f" {row.units:+8.2f} {row.mean_clv:+8.4f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nfl-engine", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    price = sub.add_parser("price", help="price the board and write the ledger")
    price.add_argument("--days", type=int, default=8)
    price.add_argument("--sims", type=int, default=40000)
    price.add_argument("--top", type=int, default=25)
    price.add_argument("--write", action="store_true", default=True)
    price.add_argument("--no-write", dest="write", action="store_false")
    price.set_defaults(func=cmd_price)

    close = sub.add_parser("close", help="stamp the closing number for CLV")
    close.add_argument("--days", type=int, default=2)
    close.add_argument("--write", action="store_true", default=True)
    close.add_argument("--no-write", dest="write", action="store_false")
    close.set_defaults(func=cmd_close)

    grade_cmd = sub.add_parser("grade", help="settle rows against final scores")
    grade_cmd.add_argument("--season", type=int, default=None)
    grade_cmd.add_argument("--write", action="store_true", default=True)
    grade_cmd.add_argument("--no-write", dest="write", action="store_false")
    grade_cmd.set_defaults(func=cmd_grade)

    report = sub.add_parser("report", help="tier, market and screen records")
    report.add_argument("--all", action="store_true")
    report.set_defaults(func=cmd_report)

    capture_cmd = sub.add_parser("capture", help="archive prices, price nothing")
    capture_cmd.add_argument("--days", type=int, default=8)
    capture_cmd.add_argument(
        "--props", action="store_true", help="also archive player-prop prices"
    )
    capture_cmd.add_argument("--max-events", type=int, default=32)
    capture_cmd.set_defaults(func=cmd_capture)

    replay_cmd = sub.add_parser("replay", help="run played weeks at their closing prices")
    replay_cmd.add_argument("--season", type=int, required=True)
    replay_cmd.add_argument("--weeks", type=int, nargs="*", default=None)
    replay_cmd.add_argument("--sims", type=int, default=20000)
    replay_cmd.add_argument("--archive", action="store_true", default=False)
    replay_cmd.add_argument("--write", action="store_true", default=True)
    replay_cmd.add_argument("--no-write", dest="write", action="store_false")
    replay_cmd.set_defaults(func=cmd_replay)

    job = sub.add_parser("job", help="capture, price, close, grade and report in order")
    job.add_argument("--days", type=int, default=8)
    job.add_argument("--sims", type=int, default=40000)
    job.add_argument("--top", type=int, default=25)
    job.add_argument("--props", action="store_true")
    job.add_argument("--max-events", type=int, default=32)
    job.add_argument("--season", type=int, default=None)
    job.add_argument("--all", action="store_true")
    job.add_argument("--write", action="store_true", default=True)
    job.add_argument("--no-write", dest="write", action="store_false")
    job.set_defaults(func=cmd_job)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
