"""`nfl-engine price`, `close`, `grade` and `report`.

Four verbs in the order a week actually happens: price the board and write every
selection *and rejection* to the ledger; capture the closing number so CLV can be
scored; settle the rows against final scores; read the record back with PPV/NPV
against the base rate. Nothing here stakes money -- the paper dry run is phase 6.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from pathlib import Path

from nfl_engine.audit.ledger import (
    LedgerEntry,
    apply_close,
    entry_from_bet,
    grade,
    load_ledger,
    market_metrics,
    metrics,
    screen_metrics,
    tier_metrics,
    update_ledger,
)
from nfl_engine.config import data_dir, load_config
from nfl_engine.data import nflverse
from nfl_engine.data.oddsapi import OddsAPIClient
from nfl_engine.market.screens import tier_of
from nfl_engine.models.drives import DriveSim
from nfl_engine.pipeline import price_slate, slate_buys

log = logging.getLogger(__name__)

LEDGER_NAME = "nfl_ledger.csv"


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


def _board_and_slate(days: int) -> tuple[list, dict]:
    client = _client()
    if not client.available():
        log.warning("no Odds API key: nothing to price")
        return [], {}
    season, week, first_day = current_week()
    slate, board = client.fetch_board(
        season=season, week=week, first_day=first_day, days=days
    )
    return list(slate.games), board


def cmd_price(args: argparse.Namespace) -> int:
    games, board = _board_and_slate(args.days)
    if not games:
        return 0
    pricings = price_slate(games, board, sim=DriveSim(n_sims=args.sims))
    entries: list[LedgerEntry] = []
    for pricing in pricings:
        for bet in pricing.bets:
            entries.append(
                entry_from_bet(
                    bet,
                    season=pricing.game.season,
                    week=pricing.game.week,
                    date=pricing.game.game_date.isoformat(),
                )
            )
    path = ledger_path()
    if args.write:
        update_ledger(path, entries)
    buys = slate_buys(pricings)
    print(f"{len(entries)} selections, {len(buys)} survive the screens")
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
    _, board = _board_and_slate(args.days)
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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
