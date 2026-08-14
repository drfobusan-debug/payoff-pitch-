"""Unprice the closes that are in-play prices, so they stop reading as closing value.

The 2026-08-08 closing capture ran after first pitch on some games -- the automation
had exhausted its quota at the scheduled time -- and #97's ``pregame_only`` guard did
not exist yet. What it recorded on that slate looks like enormous closing line value
and is in fact the scoreboard:

    ATH ML  bet +242, "close" -2000, won      clv +0.600
    BOS ML  bet -219, "close" +1600, lost     clv -0.604

That is not a line that came to our side. It is a team leading late.

The important part is how few rows this is. The whole slate carries 296 closes and it
is tempting to drop all of them, which is what I first proposed and it was wrong: only
3 of the 296 are in-play. Removing those 3 takes the slate's correlation between CLV
and *winning* from +0.111, whose 95% interval excludes zero, to +0.054, whose interval
includes it and overlaps the rest of the season's +0.025. A pre-game close cannot know
the result, so that correlation is the tell, and after these three rows the tell is
gone. The remaining 283 were captured late -- mean |clv| 0.0195 against 0.0069 on a
normal day, a wider close, not a corrupted one -- and they are the only CLV evidence
that slate has. Deleting them to tidy up the three would cost real information.

Rows are unpriced rather than deleted: the bet, its price and its grade all still
happened. Only the claim about the close is withdrawn.

The bound this uses is asymmetric, and a symmetric one is a trap worth naming: a
``batter_hr o0.5`` on a weak hitter honestly closes at +2600, and 46 real closes sit
past +1000 with a mean |clv| of 0.003. A flat |1000| rule -- which is what I wrote
first -- refuses 34 of them. Only a team market has no longshot side.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from mlb_engine.audit.clv import (
    IMPLAUSIBLE_FAVOURITE,
    IMPLAUSIBLE_TEAM_DOG,
    is_plausible_close,
)

CLV_COLUMNS = ("close_odds", "close_prob", "clv", "clv_ev")

# Judged by hand, and only ever by hand. Its price of -185 is an ordinary number and
# its move is 0.2796 against a legitimate season maximum of 0.2241 -- 25% beyond, where
# a scratched starter can move a hitter's line honestly by nearly as much. No threshold
# separates the two, so this row is named rather than detected, and no rule here claims
# it could have been caught. It is on the slate whose capture is known to have run late.
NAMED = (("2026-08-08", "TOR @ PHI", "batter_h", "Trea Turner H o1.5"),)


def scrub(led: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ledger with in-play closes unpriced, and the rows that were."""
    priced = led["close_odds"].notna()
    odds = pd.to_numeric(led["close_odds"], errors="coerce")
    implausible = priced & ~pd.Series(
        [
            is_plausible_close(o, m)
            for o, m in zip(odds.fillna(0.0), led["market"], strict=True)
        ],
        index=led.index,
    )

    named = pd.Series(False, index=led.index)
    for date, matchup, market, selection in NAMED:
        named |= (
            (led["date"] == date)
            & (led["matchup"] == matchup)
            & (led["market"] == market)
            & (led["selection"] == selection)
        )

    hit = implausible | (named & priced)
    out = led.copy()
    out.loc[hit, list(CLV_COLUMNS)] = None
    return out, led.loc[hit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(Path.home() / ".mlb_engine/audit/ledger.csv"))
    ap.add_argument(
        "--apply",
        action="store_true",
        help="write the change; without it the affected rows are only printed",
    )
    args = ap.parse_args()

    path = Path(args.ledger)
    led = pd.read_csv(path, dtype=str)
    out, hit = scrub(led)

    print(f"{len(led):,} ledger rows, {led['close_odds'].notna().sum():,} with a close")
    print(
        f"a close at or past {IMPLAUSIBLE_FAVOURITE:+.0f}, or {IMPLAUSIBLE_TEAM_DOG:+.0f} "
        "on a team market, cannot be pre-game; named rows are listed in NAMED\n"
    )
    if hit.empty:
        print("nothing to unprice")
        return
    cols = ["date", "matchup", "market", "selection", "odds", "close_odds", "clv", "result"]
    print(hit[cols].to_string(index=False))
    print(f"\n{len(hit)} row(s) to unprice, of {led['close_odds'].notna().sum():,} priced")

    if not args.apply:
        print("\ndry run -- pass --apply to write")
        return
    # Written back with the line ending it arrived with. The ledger is CRLF, pandas
    # defaults to the platform's, and the difference turns a four-row correction into
    # an 86,444-line diff that nobody can review -- and that no reviewer could
    # distinguish from a rewrite of every value in the file.
    terminator = "\r\n" if b"\r\n" in path.read_bytes()[:4096] else "\n"
    backup = path.with_suffix(".csv.pre-scrub")
    shutil.copy2(path, backup)
    out.to_csv(path, index=False, lineterminator=terminator)
    print(f"\nwritten; the ledger as it was is at {backup}")


if __name__ == "__main__":
    main()
