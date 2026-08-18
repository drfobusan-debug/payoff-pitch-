"""Archive an EV Analytics player propsheet as real, paired prop prices.

The one thing 27 seasons of game lines cannot give us is a *prop* price
history: no free archive exists, and the day we grade a total-bases bet
against an assumed -110 is the day the grade becomes fiction. A saved
propsheet is the cheapest fix -- one book, both sides, every line, on the day.

What this reads, and what it deliberately does not:

* both American prices per line, taken as printed. A side the book did not
  print is left empty, never inferred from the other one -- an unpaired
  price is still a real price, and the pair is where the vig lives, so
  filling one in from the other would invent the number we most need;
* the book's line, the player, the team and the game;
* THE BAT X projection and the book-implied projection, kept side by side
  so a later study can ask which one moved;
* the projected pitch count, which is an outside read on the hook -- the
  input our own removal hazard is fitted on.

The sheet is *not* an independent forecast. Its projections are THE BAT X,
which the ledger already grades through ``batx_study.py``; treating it as a
second opinion would be counting the same source twice. Its value here is
the price column.

Nor is its SUGGESTED BET column imported. It is the implied-vs-projection
percentage difference, which is the ordering our own best-bets list was
just taken off: on this sheet it points at triples at +4100 and stolen
bases at +925, the price band where our ledger loses most.

The sheet paginates -- a full slate is two or three saved pages, ~600 prices --
so every page saved for the slate is archived together, deduplicated on
(date, book, player, sheet market, line) because the pages overlap at their
seams.

Usage -- the saves in ~/Downloads are found, dated and filed by themselves,
so the daily step is the command and nothing else::

    python scripts/propsheet_import.py

``--date`` and ``--out`` remain for backfilling old saves, where the year
cannot be inferred from today::

    python scripts/propsheet_import.py --html page1.html page2.html \\
        --date 2026-08-15 --out ~/.mlb_engine/props/2026-08-15.csv
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from datetime import date as Date
from pathlib import Path

import pandas as pd

DEFAULT_OUT_DIR = "~/.mlb_engine/props"
DEFAULT_SAVE_DIR = "~/Downloads"
# The browser names the save after the page title, so "prop" is the stable part
# across "MLB_Betting_Model_-_Player_Prop_Odd_Predictions_-_EVAnalytics.com.html"
# and whatever the propsheet is called next season. read_sheet still has to find
# the table, so a wrong guess fails loudly rather than importing something else.
_SAVED_SHEET = re.compile(r"prop", re.IGNORECASE)

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Sheet market -> our ledger market. Anything absent is kept with an empty
# ledger market rather than guessed at: a mis-mapped market grades the wrong
# outcome, which is worse than an unjoined row.
MARKET_MAP: dict[str, str] = {
    "Hits": "batter_h",
    "Singles": "batter_1b",
    "Doubles": "batter_2b",
    "Triples": "batter_3b",
    "Home Runs": "batter_hr",
    "Total Bases": "batter_tb",
    "Runs": "batter_r",
    "RBIs": "batter_rbi",
    "Hits Runs and RBIs": "batter_hrr",
    "Walks": "batter_bb",
    "Hitter Strikeouts": "batter_k",
    "Strikeouts": "pitcher_k",
    "Pitching Outs": "pitcher_outs",
    "Hits Allowed": "pitcher_h",
    "Walks Allowed": "pitcher_bb",
    "Earned Runs": "pitcher_er",
}

# "0.5 (+280)" -> line 0.5, price +280. A line printed without a price is a
# line we cannot bet, so both halves of the cell are required.
_PRICED = re.compile(r"^\s*([\d.]+)\s*\(([+-]\d+)\)\s*$")

# "Aug 15" -- the sheet prints no year.
_SHEET_DAY = re.compile(r"^\s*([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})\s*$")

COLUMNS = (
    "date",
    "book",
    "player",
    "team",
    "game",
    "sheet_market",
    "market",
    "line",
    "over_american",
    "under_american",
    "batx_projection",
    "implied_projection",
    "batting_order",
    "pitch_count",
)


def parse_priced(cell: object) -> tuple[float, float] | None:
    match = _PRICED.match(str(cell))
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _number(cell: object) -> float | None:
    try:
        value = float(str(cell))
    except ValueError:
        return None
    return None if pd.isna(value) else value


def parse_day(cell: object, today: Date) -> Date | None:
    """Read "Aug 15" as a date, choosing the year that sits nearest today.

    Nearest rather than the current one so that a sheet saved either side of
    New Year lands in the right season instead of eleven months away.
    """
    match = _SHEET_DAY.match(str(cell))
    if match is None:
        return None
    month = MONTHS.get(match.group(1).lower())
    if month is None:
        return None
    day = int(match.group(2))
    best: Date | None = None
    for year in (today.year - 1, today.year, today.year + 1):
        try:
            candidate = Date(year, month, day)
        except ValueError:  # Feb 29 in a common year
            continue
        if best is None or abs((candidate - today).days) < abs((best - today).days):
            best = candidate
    return best


def slate_day(sheet: pd.DataFrame, today: Date) -> Date | None:
    """The date most of the sheet's games fall on, which names the archive file.

    Every row keeps its own date, so a sheet spanning midnight is still
    archived row-accurately; this only decides what the file is called.
    """
    if "DATE" not in sheet.columns:
        return None
    days = [d for d in (parse_day(cell, today) for cell in sheet["DATE"]) if d is not None]
    if not days:
        return None
    return Counter(days).most_common(1)[0][0]


def find_saved_sheets(directory: str, within_hours: float = 12.0) -> list[Path]:
    """Every propsheet page in ``directory`` saved alongside the newest one.

    Taken by save time rather than by name: the browser numbers a second save
    of the same page " (1)", and last night's pages are still in the folder.
    Pages saved more than ``within_hours`` before the newest one are a previous
    slate and left alone, so a stale straggler cannot smuggle yesterday's
    prices into tonight's archive.
    """
    folder = Path(os.path.expanduser(directory))
    if not folder.is_dir():
        return []
    saves = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".html", ".htm") and _SAVED_SHEET.search(p.name)
    ]
    if not saves:
        return []
    newest = max(p.stat().st_mtime for p in saves)
    return sorted(
        (p for p in saves if newest - p.stat().st_mtime <= within_hours * 3600),
        key=lambda p: p.stat().st_mtime,
    )


def read_sheet(path: str) -> pd.DataFrame:
    tables = pd.read_html(path)
    for table in tables:
        if {"MARKET", "PLAYER", "OVER", "UNDER"}.issubset(table.columns):
            return table
    raise SystemExit(f"no propsheet table found in {path}")


def to_rows(sheet: pd.DataFrame, day: Date) -> list[dict[str, object]]:
    """``day`` dates any row whose own DATE cell cannot be read."""
    rows: list[dict[str, object]] = []
    for _, row in sheet.iterrows():
        row_day = parse_day(row.get("DATE"), day) or day
        over = parse_priced(row["OVER"])
        under = parse_priced(row["UNDER"])
        side = over if over is not None else under
        if side is None:
            continue
        if over is not None and under is not None and over[0] != under[0]:
            # Both sides must sit on the same number, or the pair prices
            # nothing -- an over 1.5 against an under 2.5 is not a market.
            continue
        line = side[0]
        sheet_market = str(row["MARKET"]).strip()
        rows.append(
            {
                "date": row_day.isoformat(),
                "book": str(row.get("SITE", "")).strip(),
                "player": str(row["PLAYER"]).strip(),
                "team": str(row.get("TM", "")).strip(),
                "game": str(row.get("GAME", "")).strip(),
                "sheet_market": sheet_market,
                "market": MARKET_MAP.get(sheet_market, ""),
                "line": line,
                "over_american": over[1] if over is not None else None,
                "under_american": under[1] if under is not None else None,
                "batx_projection": _number(row.get("THE BAT X PROJECTION")),
                "implied_projection": _number(row.get("IMPLIED PROJECTION")),
                "batting_order": _number(row.get("BATTING ORDER")),
                "pitch_count": _number(row.get("THE BAT X PITCH COUNT")),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--html",
        nargs="+",
        help=f"saved propsheet pages; defaults to the newest slate's in {DEFAULT_SAVE_DIR}",
    )
    ap.add_argument("--date", help="slate date, YYYY-MM-DD; read off the sheet when omitted")
    ap.add_argument("--out", help=f"defaults to {DEFAULT_OUT_DIR}/<date>.csv")
    ap.add_argument("--downloads", default=DEFAULT_SAVE_DIR, help=argparse.SUPPRESS)
    args = ap.parse_args()

    pages = [str(p) for p in (args.html or find_saved_sheets(args.downloads))]
    if not pages:
        raise SystemExit(f"no saved propsheet in {args.downloads} -- pass --html")

    sheets = [read_sheet(p) for p in pages]
    if args.html is None:
        for page, sheet in zip(pages, sheets, strict=True):
            print(f"reading {page} ({len(sheet)} rows)")

    both = pd.concat(sheets, ignore_index=True)
    day = Date.fromisoformat(args.date) if args.date else slate_day(both, Date.today())
    if day is None:
        raise SystemExit("could not read a date off the sheet -- pass --date YYYY-MM-DD")
    if args.date is None and day != Date.today():
        # A save that was never refreshed imports yesterday's prices under
        # yesterday's name, which is silent and wrong rather than loud and wrong.
        print(f"warning: this sheet is for {day.isoformat()}, not today")
    rows = to_rows(both, day)
    if not rows:
        raise SystemExit("no priced rows parsed -- check the saved page is the propsheet itself")

    out = pd.DataFrame(rows, columns=list(COLUMNS))
    # The pages overlap where one ends and the next begins, and a page saved
    # twice is saved whole; the last copy of a row is the later save of it.
    out = out.drop_duplicates(
        subset=["date", "book", "player", "sheet_market", "line"], keep="last"
    ).reset_index(drop=True)
    dest = os.path.expanduser(args.out or f"{DEFAULT_OUT_DIR}/{day.isoformat()}.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)

    mapped = out[out.market != ""]
    paired = out[out.over_american.notna() & out.under_american.notna()]
    print(f"{len(both)} sheet rows over {len(pages)} page(s) -> {len(out)} prices -> {dest}")
    print(f"  both sides printed: {len(paired)}; one side only: {len(out) - len(paired)}")
    print(f"  joinable to a ledger market: {len(mapped)} over {mapped.market.nunique()} markets")
    unmapped = sorted(set(out.loc[out.market == "", "sheet_market"]))
    if unmapped:
        print(f"  kept unmapped (no ledger market): {', '.join(unmapped)}")
    counts = out.pitch_count.notna().sum()
    if counts:
        print(f"  projected pitch counts on {counts} starters")


if __name__ == "__main__":
    main()
