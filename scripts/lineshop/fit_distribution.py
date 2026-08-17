"""Regenerate the empirical ``(line, result)`` tables the line-shop scanner
prices middles and key-number crossings from.

A middle is only worth taking if the window between the two numbers is hit often
enough to pay for losing one leg, and that frequency is not the unconditional
frequency of a margin: games land on 3 far more often than on 4, but *given a
line of -10* they land on 3 rarely. So what gets tabulated here is the joint
distribution of the closing number and the result, and the scanner conditions on
the number in front of it.

NFL comes from the nflverse game file (closing ``spread_line`` / ``total_line``,
1999 on). CFB comes from CFBD ``/lines``, using the median of the real
sportsbooks on the game -- TeamRankings publishes into the same feed and is a
projection, not a price, so it is dropped.

Usage::

    python -m scripts.lineshop.fit_distribution [--out lineshop/data]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from cfb_engine.config import Credentials
from cfb_engine.data.cfbd import CFBDClient
from nfl_engine.data import nflverse

# CFBD carries projection providers in the same payload as sportsbooks.
NOT_A_BOOK = {"teamrankings"}
CFB_FIRST_SEASON = 2014
NFL_FIRST_SEASON = 1999

Table = dict[str, dict[str, dict[str, int]]]  # market -> line -> result -> count


def _add(table: Table, market: str, line: float, result: float) -> None:
    table[market][f"{line:g}"][f"{result:g}"] += 1


def _empty() -> Table:
    return {
        "margin": defaultdict(lambda: defaultdict(int)),
        "total": defaultdict(lambda: defaultdict(int)),
    }


def nfl_table(first_season: int = NFL_FIRST_SEASON) -> Table:
    frame = nflverse.graded_games(first_season)
    table = _empty()
    for row in frame.itertuples():
        # nflverse ``spread_line`` is the home handicap as a favourite margin
        # (positive = home favoured); ``result`` is home score - away score.
        _add(table, "margin", float(row.spread_line), float(row.result))
        _add(table, "total", float(row.total_line), float(row.total))
    return table


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def cfb_table(first_season: int = CFB_FIRST_SEASON, last_season: int = 2025) -> Table:
    client = CFBDClient(Credentials().cfbd_api_key)
    table = _empty()
    for season in range(first_season, last_season + 1):
        for game in client.fetch_lines(season):
            home, away = _number(game.get("homeScore")), _number(game.get("awayScore"))
            if home is None or away is None:
                continue
            lines = game.get("lines")
            books = [
                line
                for line in (lines if isinstance(lines, list) else [])
                if isinstance(line, dict)
                and str(line.get("provider", "")).lower() not in NOT_A_BOOK
            ]
            spreads = [v for v in (_number(b.get("spread")) for b in books) if v is not None]
            totals = [v for v in (_number(b.get("overUnder")) for b in books) if v is not None]
            # CFBD quotes the spread from the home side with the sign a bettor
            # sees (+12.5 = home getting points); the scanner's margin axis is
            # home score - away score, so the handicap flips.
            if spreads:
                _add(table, "margin", -statistics.median(spreads), home - away)
            if totals:
                _add(table, "total", statistics.median(totals), home + away)
    return table


# How far either side of a number to look when asking whether it is lumpier than
# its neighbourhood. Wide enough to average out the local slope, narrow enough
# that 3 and 7 do not smooth each other.
NEIGHBOURHOOD = 3
# A number cannot be a tenth as likely as its neighbours, and 3 is only about
# twice as likely -- anything outside this is a small-sample artefact.
FACTOR_BOUNDS = (0.5, 4.0)


def key_factors(table: Table, market: str) -> dict[str, float]:
    """How much mass sits on each number relative to its neighbours.

    Football scores are lumpy in a way no smooth curve reproduces -- margins of
    3 and 7 are roughly twice as common as the numbers either side of them --
    and that lumpiness is a property of the *number*, not of the line the game
    was priced at. Measuring it over the whole sample gives the conditional
    estimator a shape to fall back on when only a few hundred games were priced
    near the number in question.

    Margins are folded to absolute value: a 3-point win and a 3-point loss are
    the same field-goal lump.
    """
    pooled: dict[float, int] = defaultdict(int)
    for results in table[market].values():
        for result, n in results.items():
            value = float(result)
            pooled[abs(value) if market == "margin" else value] += n
    out: dict[str, float] = {}
    for value, count in pooled.items():
        around = [
            pooled.get(value + offset, 0)
            for offset in range(-NEIGHBOURHOOD, NEIGHBOURHOOD + 1)
            if offset
        ]
        local = statistics.mean(around) if around else 0.0
        if local <= 0 or count < 10:
            continue
        lo, hi = FACTOR_BOUNDS
        out[f"{value:g}"] = min(max(count / local, lo), hi)
    return out


def _write(table: Table, path: Path, *, sport: str, source: str) -> None:
    payload = {
        "sport": sport,
        "source": source,
        "games": {m: sum(sum(r.values()) for r in lines.values()) for m, lines in table.items()},
        "key_factors": {m: key_factors(table, m) for m in table},
        "counts": {m: {ln: dict(res) for ln, res in lines.items()} for m, lines in table.items()},
    }
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    print(f"{sport}: {payload['games']} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("lineshop/data"))
    ap.add_argument("--sport", choices=("cfb", "nfl", "both"), default="both")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.sport in ("nfl", "both"):
        _write(
            nfl_table(),
            args.out / "nfl_history.json",
            sport="nfl",
            source=f"nflverse game file, {NFL_FIRST_SEASON}+",
        )
    if args.sport in ("cfb", "both"):
        _write(
            cfb_table(),
            args.out / "cfb_history.json",
            sport="cfb",
            source=f"CFBD /lines median of books, {CFB_FIRST_SEASON}+",
        )


if __name__ == "__main__":
    main()
