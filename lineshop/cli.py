"""``lineshop`` -- read the CFB/NFL board and print what is shoppable.

    lineshop scan --sport cfb            # crossings and middles worth the vig
    lineshop board --sport nfl           # best number and price per side
    lineshop books --sport cfb --have DraftKings --have BetMGM

By default the scan only counts numbers reachable at ``DEFAULT_BOOKS`` -- a
number nobody can bet is not an edge. ``--book`` replaces that list and
``--every-book`` drops it, which is how the whole screen gets surveyed before
deciding which accounts are worth opening.

Nothing here places, sizes or recommends a bet against a model: every number
printed is either something a book is offering or a frequency counted from
history.
"""

from __future__ import annotations

import argparse
import sys

from cfb_engine.config import Credentials
from lineshop import books as books_mod
from lineshop import feed
from lineshop.scan import SPREADS, TOTALS, GameScan, scan

# The accounts the operator actually holds. Measured on the live board, these
# four hold the best available number on ~57% of CFB sides against 21% for
# DraftKings and BetMGM alone; ``lineshop books`` re-measures it.
DEFAULT_BOOKS = ("DraftKings", "FanDuel", "BetRivers", "BetMGM")


def _load(sport: str) -> list[feed.Game]:
    key = Credentials().odds_api_key
    if not key:
        print("no odds API key (set THE_ODDS_API_KEY); nothing to shop", file=sys.stderr)
        return []
    return feed.fetch(sport, key)


def _print_scan(scans: list[GameScan]) -> None:
    crossings = [c for s in scans for c in s.crossings]
    middles = [m for s in scans for m in s.middles]
    print(f"{len(scans)} games scanned")

    print(f"\n-- crossings ({len(crossings)}): better number than the price costs")
    for c in sorted(crossings, key=lambda c: c.edge, reverse=True):
        keys = f" crosses {','.join(str(k) for k in c.keys)}" if c.keys else ""
        base = f"{c.consensus_point:g}" if c.market == TOTALS else f"{c.consensus_point:+g}"
        print(
            f"  {c.matchup:<44} {c.best.label():<26} vs consensus "
            f"{base} ({c.consensus_american:+d})  "
            f"number {c.prob_gain * 100:+.1f}% / price {-c.price_cost * 100:+.1f}% "
            f"= {c.edge * 100:+.1f}%{keys}  [{', '.join(c.best.books)}; n={c.sample}]"
        )

    print(f"\n-- middles ({len(middles)}): priced against the empirical distribution")
    for m in sorted(middles, key=lambda m: m.ev, reverse=True):
        flag = " THIN" if m.thin else ""
        print(
            f"  {m.matchup:<44} {m.low.label()} + {m.high.label()}  "
            f"hits {m.p_middle * 100:.1f}%, pushes {m.p_push * 100:.1f}% "
            f"(n={m.sample}){flag}  EV {m.ev * 100:+.1f}%  "
            f"[{'/'.join(m.low.books[:1] + m.high.books[:1])}]"
        )


def _print_board(scans: list[GameScan]) -> None:
    for s in sorted(scans, key=lambda s: s.commence):
        spread = f"{s.consensus_spread:+g}" if s.consensus_spread is not None else "n/a"
        total = f"{s.consensus_total:g}" if s.consensus_total is not None else "n/a"
        print(f"\n{s.matchup}  ({s.books} books, consensus {spread} / {total})")
        for (market, _side), offer in sorted(s.best.items()):
            if market in (SPREADS, TOTALS) or market == "h2h":
                print(f"    {offer.label():<28} {', '.join(offer.books)}")


def _print_books(report: books_mod.BookReport, have: tuple[str, ...]) -> None:
    print(f"{report.sport.upper()}: {report.games} games, {report.sides} priced sides")
    print(f"{'book':<18}{'sides':>6}{'best%':>8}{'avg cost':>10}{'hold':>8}")
    for s in report.scores:
        print(
            f"{s.book:<18}{s.sides:>6}{s.best_share * 100:>7.1f}%"
            f"{s.avg_cost * 100:>9.2f}%{s.hold * 100:>7.2f}%"
        )
    ranked = sorted(report.coverage.items(), key=lambda kv: kv[1], reverse=True)
    print("\nbest sets by union coverage (share of sides the set holds the best offer on)")
    for combo, share in ranked[:6]:
        print(f"  {share * 100:5.1f}%  {', '.join(combo)}")
    if have:
        with_have = [(c, v) for c, v in ranked if set(have) <= set(c)]
        print(f"\nkeeping {' + '.join(have)}:")
        for combo, share in with_have[:6]:
            print(f"  {share * 100:5.1f}%  {', '.join(combo)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lineshop", description=__doc__)
    parser.add_argument("command", choices=("scan", "board", "books"))
    parser.add_argument("--sport", choices=("cfb", "nfl"), default="cfb")
    parser.add_argument("--have", action="append", default=[], help="a book you already hold")
    parser.add_argument("--book", action="append", default=[], help="restrict to these books")
    parser.add_argument(
        "--every-book", action="store_true", help="scan the whole screen, bettable or not"
    )
    parser.add_argument("--set-size", type=int, default=4)
    args = parser.parse_args(argv)

    games = _load(args.sport)
    if not games:
        return 1
    if args.command == "books":
        have = tuple(args.have) or DEFAULT_BOOKS[:2]
        _print_books(books_mod.rank(args.sport, games, sets_of=args.set_size, fixed=have), have)
        return 0
    books = () if args.every_book else tuple(args.book) or DEFAULT_BOOKS
    if books:
        print(f"shopping {', '.join(books)}")
    scans = scan(args.sport, games, books)
    if args.command == "board":
        _print_board(scans)
    else:
        _print_scan(scans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
