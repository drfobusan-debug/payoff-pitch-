"""Statcast ingestion via pybaseball / Baseball Savant (free, public).

Pulls pitch-level Statcast data for a date range and caches it locally so
repeated runs on the same day are cheap. Downstream feature code slices this
frame by batter/pitcher and by the required rolling windows.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Columns we actually use downstream. Keeping the frame narrow saves memory/cache.
USE_COLS = [
    "game_date",
    "batter",
    "pitcher",
    "pitch_type",
    "release_speed",
    "events",
    "description",
    "type",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
    "balls",
    "strikes",
    "pfx_z",
    "launch_speed",
    "launch_angle",
    "launch_speed_angle",  # 6 == barrel
    "bb_type",
    "hc_x",
    "hc_y",
    "hit_distance_sc",  # projected landing distance, for expected HR
    "zone",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "woba_value",
    "woba_denom",
    # bat tracking (2024+)
    "bat_speed",
    "swing_length",
    # pitch release / movement
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "release_spin_rate",
]


def batted_balls(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are genuinely balls in play.

    Statcast records exit velocity on **foul balls** as well as balls in play, so
    filtering on ``launch_speed.notna()`` builds a pool that is ~47% fouls --
    weak contact by construction, which drags every contact-quality rate away
    from its true value (league hard-hit reads 24% instead of 39%) and roughly
    doubles any batted-ball count. Balls in play are ``type == "X"``
    (equivalently ``description == "hit_into_play"``).

    Falls back to the exit-velocity filter only when neither column is present,
    so callers working with a reduced frame still get a usable slice.
    """
    if "type" in df:
        bip = df[df["type"].eq("X")]
    elif "description" in df:
        bip = df[df["description"].eq("hit_into_play")]
    else:
        return df[df["launch_speed"].notna()] if "launch_speed" in df else df.iloc[0:0]
    return bip[bip["launch_speed"].notna()] if "launch_speed" in bip else bip


class StatcastRepository:
    """Loads and caches Statcast pitch-level data."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, start: Date, end: Date) -> Path:
        return self.cache_dir / f"statcast_{start.isoformat()}_{end.isoformat()}.pkl"

    def load_range(self, start: Date, end: Date, refresh: bool = False) -> pd.DataFrame:
        """Return pitch-level Statcast data for [start, end] (inclusive)."""
        cache = self._cache_path(start, end)
        if cache.exists() and not refresh:
            log.info("Loading cached Statcast %s..%s", start, end)
            return pd.read_pickle(cache)

        from pybaseball import statcast  # imported lazily; heavy dependency

        log.info("Downloading Statcast %s..%s (this can take a while)", start, end)
        df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
        if df is None or df.empty:
            df = pd.DataFrame(columns=USE_COLS)
        else:
            keep = [c for c in USE_COLS if c in df.columns]
            df = df[keep].copy()
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df.to_pickle(cache)
        return df

    def load_trailing(self, as_of: Date, days: int, refresh: bool = False) -> pd.DataFrame:
        """Statcast for the ``days`` ending the day before ``as_of``."""
        end = as_of - timedelta(days=1)
        start = end - timedelta(days=days - 1)
        return self.load_range(start, end, refresh=refresh)

    def max_window(self, as_of: Date, windows: list[int], refresh: bool = False) -> pd.DataFrame:
        """Fetch one frame covering the largest window; callers slice by date."""
        longest = max(windows)
        return self.load_trailing(as_of, longest, refresh=refresh)
