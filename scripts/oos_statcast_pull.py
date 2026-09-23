"""Pull and cache Statcast pitch data for whole seasons, one month per cache file.

Uses the engine's ``StatcastRepository`` so the frames carry exactly the columns
production reads; cached months are skipped.

    .venv/bin/python scripts/oos_statcast_pull.py --seasons 2024 2025 2026
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

from mlb_engine.data.statcast import StatcastRepository, dedupe_pitches

DEFAULT_CACHE = Path.home() / ".mlb_engine" / "cache"
SEASON_LAST = {2024: Date(2024, 9, 30), 2025: Date(2025, 9, 30), 2026: Date(2026, 9, 22)}


def month_ranges(season: int, last: Date) -> list[tuple[Date, Date]]:
    out = []
    cur = Date(season, 3, 1)
    while cur <= last:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur, min(nxt - timedelta(days=1), last)))
        cur = nxt
    return out


def season_frame(season: int, cache_dir: Path = DEFAULT_CACHE, last: Date | None = None) -> pd.DataFrame:
    repo = StatcastRepository(cache_dir)
    frames = [repo.load_range(a, b) for a, b in month_ranges(season, last or SEASON_LAST[season])]
    df = dedupe_pitches(pd.concat(frames, ignore_index=True))
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=[2024, 2025, 2026])
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for s in args.seasons:
        df = season_frame(s, args.cache_dir)
        print(f"{s}: {len(df):,} pitches, {df['game_date'].nunique()} game days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
