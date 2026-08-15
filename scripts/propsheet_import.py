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

Usage::

    python scripts/propsheet_import.py --html propsheet.html --date 2026-08-15 \\
        --out ~/.mlb_engine/props/2026-08-15.csv
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import date as Date

import pandas as pd

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


def read_sheet(path: str) -> pd.DataFrame:
    tables = pd.read_html(path)
    for table in tables:
        if {"MARKET", "PLAYER", "OVER", "UNDER"}.issubset(table.columns):
            return table
    raise SystemExit(f"no propsheet table found in {path}")


def to_rows(sheet: pd.DataFrame, day: Date) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, row in sheet.iterrows():
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
                "date": day.isoformat(),
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
    ap.add_argument("--html", required=True, help="saved EV Analytics propsheet")
    ap.add_argument("--date", required=True, help="slate date, YYYY-MM-DD (the sheet prints no year)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    day = Date.fromisoformat(args.date)
    sheet = read_sheet(args.html)
    rows = to_rows(sheet, day)
    if not rows:
        raise SystemExit("no priced rows parsed -- check the saved page is the propsheet itself")

    out = pd.DataFrame(rows, columns=list(COLUMNS))
    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)

    mapped = out[out.market != ""]
    paired = out[out.over_american.notna() & out.under_american.notna()]
    print(f"{len(sheet)} sheet rows -> {len(out)} prices -> {dest}")
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
