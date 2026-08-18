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
    # change in run expectancy on the pitch, signed for the pitching side. The
    # only field that scores a called strike and a foul, so it is what a
    # pitch-type matchup read has to be built on. Frames cached before it was
    # requested do not carry it; callers must treat it as optional.
    "delta_run_exp",
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


#: A pitch has no id in the Statcast feed, so it is identified by the game, the
#: two players, the count and the delivery. Two pitches matching on all of these
#: are the same pitch scraped twice.
PITCH_KEY = [
    "game_date",
    "home_team",
    "away_team",
    "batter",
    "pitcher",
    "inning",
    "inning_topbot",
    "balls",
    "strikes",
    "pitch_type",
    "release_speed",
    "description",
    "events",
]


def dedupe_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pitches that appear more than once, for frames stitched from caches.

    Cached ranges overlap, and a pitch carries no id, so concatenating two of
    them counts the shared days twice. ``drop_duplicates()`` over every column
    does not catch it -- the frames can differ in which columns they carry, or
    in a value Savant revised between scrapes -- and the result silently
    corrupts every rate built on plate appearances: a starter's walk rate read
    .275 against a true .100, which pushed median SIERA from 4.06 to 7.0 and
    would have moved the ace gate.

    Production reads one cached range at a time and does not need this; it is
    for research code assembling a season from several caches.
    """
    key = [c for c in PITCH_KEY if c in df.columns]
    return _unique_index(df.drop_duplicates(subset=key) if key else df)


def _unique_index(df: pd.DataFrame) -> pd.DataFrame:
    """Renumber the rows.

    pybaseball stitches the range together one day at a time and keeps each
    day's row numbers, so labels repeat thousands of times. Any ``series[idx]``
    downstream then returns every row sharing that label instead of a value --
    it reads as a hitter's whole slice where one batted ball was meant.
    """
    return df if df.index.is_unique else df.reset_index(drop=True)


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
            return _unique_index(pd.read_pickle(cache))

        from pybaseball import statcast  # imported lazily; heavy dependency

        log.info("Downloading Statcast %s..%s (this can take a while)", start, end)
        df = statcast(start_dt=start.isoformat(), end_dt=end.isoformat())
        if df is None or df.empty:
            df = pd.DataFrame(columns=USE_COLS)
        else:
            keep = [c for c in USE_COLS if c in df.columns]
            df = df[keep].copy()
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = _unique_index(df)
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
