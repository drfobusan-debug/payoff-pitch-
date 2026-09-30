"""Warm the NHL API cache with play-by-play + shift charts for whole seasons.

    python scripts/nhl/fetch_shifts.py --from 2023 --to 2024

Idempotent: finished games are cached forever, so re-running only fetches
what is missing. Feeds ``scripts/nhl/rapm_fit.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from nhl_engine.config import cache_dir
from nhl_engine.data.nhlapi import NHLAPIClient, regular_season_game_ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2023)
    ap.add_argument("--to", dest="end", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    client = NHLAPIClient(cache_dir=cache_dir() / "nhlapi")

    def one(gid: int) -> tuple[int, bool, int]:
        pbp = client.play_by_play(gid)
        shifts = client.shifts(gid)
        return gid, pbp is not None, len(shifts)

    for season in range(args.start, args.end + 1):
        t0 = time.time()
        ok = missing = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for _gid, has_pbp, n_shifts in pool.map(one, regular_season_game_ids(season)):
                if has_pbp and n_shifts:
                    ok += 1
                else:
                    missing += 1
        print(f"{season}: {ok} games cached, {missing} missing ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
