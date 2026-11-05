"""Profile of the "line moved >= 1 toward the side" ATS/total rows (graded-only, analysis).

Splits the band's rows by venue, favourite/underdog, AP rank, conference tier and net
rank, W-L and flat units per cell with the market's expectation, so a losing band can
be read for where the loss sits. Reads the production ledger and archives; writes
nothing; touches no pricing. Cells are tiny this early -- read n before the record.

    .venv/bin/python -m scripts.cbb.steam_profile [--through YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import date as Date

from cbb_engine import bands
from cbb_engine.cli import _season_ctx
from cbb_engine.config import SPORT, runtime_root
from engine_runtime import ledger

Cell = Callable[[ledger.Row, bands.GameCtx], str]


def _tier(t: bands.Team) -> str:
    if t.power is True:
        return "power"
    if t.power is False:
        return "mid-major"
    return "unknown conf"


def _rank(t: bands.Team) -> str:
    if t.ap is not None:
        return "AP top 25"
    if t.net_rank is None:
        return "net rank n/a"
    return "net 1-100" if t.net_rank <= 100 else "net 101-200" if t.net_rank <= 200 else "net 200+"


SPLITS: dict[str, Cell] = {
    "venue": lambda r, c: "neutral" if c.neutral else r.side if r.market == "game_ats" else "n/a",
    "role": lambda r, c: (
        ("favourite" if bands._fav(r) else "underdog") if r.market == "game_ats" else r.side
    ),
    "conference tier (the side)": lambda r, c: _tier(bands._me(r, c)),
    "tier matchup": lambda r, c: f"{_tier(bands._me(r, c))} vs {_tier(bands._them(r, c))}",
    "rank (the side)": lambda r, c: _rank(bands._me(r, c)),
    "opponent rank": lambda r, c: _rank(bands._them(r, c)),
    "conf / non-conf": lambda r, c: "conference" if c.conference else "non-conference",
    "move size": lambda r, c: _move_bucket(r, c),
}


def _move_bucket(r: ledger.Row, c: bands.GameCtx) -> str:
    first = c.opener.get((r.market, r.side))
    if first is None or r.line is None:
        return "n/a"
    toward = r.line - first if r.side == "over" else first - r.line
    return "1-1.5" if toward < 2 else "2-2.5" if toward < 3 else "3+"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", default=Date.today().isoformat())
    args = ap.parse_args()
    root = runtime_root()
    days = sorted(p.name for p in (root / "ledger" / SPORT).glob("*") if p.is_dir())
    season = [r for d in days if d <= args.through for r in ledger.record(root, SPORT, d).values()]
    ctx, _ = _season_ctx(root, season, Date.fromisoformat(args.through))
    main = [r for r in bands.main_rows(season) if r.event_id in ctx]
    for market in ("game_ats", "game_total"):
        rows = [r for r in main if r.market == market and bands._moved(r, ctx[r.event_id])]
        w = sum(r.outcome == "win" for r in rows)
        lo = sum(r.outcome == "loss" for r in rows)
        units = sum(r.pnl or 0.0 for r in rows)
        exp = sum(bands.market_expected(r) for r in rows)
        print(
            f"\n== {market}: line moved >= 1 toward the side  n={len(rows)} {w}-{lo} {units:+.1f}u (market expected {exp:+.1f})"
        )
        for name, cell in SPLITS.items():
            groups: dict[str, list[ledger.Row]] = defaultdict(list)
            for r in rows:
                groups[cell(r, ctx[r.event_id])].append(r)
            print(f"  {name}")
            for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                ex = [(r.pnl or 0.0) - bands.market_expected(r) for r in rs]
                m = sum(ex) / len(ex)
                se = (
                    math.sqrt(sum((x - m) ** 2 for x in ex) / (len(ex) - 1) / len(ex))
                    if len(ex) > 1
                    else math.nan
                )
                print(
                    f"    {key:<26} n={len(rs):<4} {sum(r.outcome == 'win' for r in rs)}-{sum(r.outcome == 'loss' for r in rs):<4}"
                    f" {sum(r.pnl or 0.0 for r in rs):+6.1f}u  excess {sum(ex):+6.1f} ± {se * len(ex):4.1f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
