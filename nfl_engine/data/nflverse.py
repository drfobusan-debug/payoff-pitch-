"""nflverse and nfldata loaders, cached on disk.

Everything the engine needs about what happened on a football field is free and
public here: play-by-play back to 1999, weekly player lines, snap counts,
per-play participation (who was on the field, and who ran a route), Next Gen
Stats tracking aggregates, injury reports and depth charts. The game file adds
the thing no vendor sells us -- 27 seasons of *closing* spreads, totals and
moneylines, which is both the backtest spine and the closing-line-value
benchmark.

Two conventions worth knowing:

* **Completed seasons are immutable, the current one is not.** A 2019 parquet
  will never change, so it is cached forever; the live season is refetched after
  ``LIVE_TTL``. Without that split either the cache serves a stale week or every
  run re-downloads 25 years of play-by-play.
* **A failed fetch returns an empty frame, it does not raise.** A Sunday morning
  run that loses one optional feed should price the slate without it and say so,
  the same way the MLB engine survives a Savant hiccup. Callers check
  ``frame.empty``; the loss is logged.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from mlb_engine.data import http
from nfl_engine.config import cache_dir

log = logging.getLogger(__name__)

_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

# Seconds before a current-season file is refetched.
LIVE_TTL = 6 * 3600
# Play-by-play starts in 1999; per-play participation and snap counts in 2016.
PBP_FIRST_SEASON = 1999
PARTICIPATION_FIRST_SEASON = 2016
SNAPS_FIRST_SEASON = 2012
NGS_FIRST_SEASON = 2016


def _current_season(today: date | None = None) -> int:
    """The season a date belongs to. A season is named for its September."""
    day = today or date.today()
    return day.year if day.month >= 3 else day.year - 1


def _cache_path(name: str) -> Path:
    return cache_dir() / "nflverse" / name


def _read_cached(path: Path, ttl: int | None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if ttl is not None and time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:  # corrupt or truncated download
        log.warning("nflverse cache unreadable (%s): %s", path.name, exc)
        return None


def _fetch(url: str, name: str, ttl: int | None, *, csv: bool = False) -> pd.DataFrame:
    """Load ``url`` through the on-disk cache, returning an empty frame on failure."""
    path = _cache_path(name)
    cached = _read_cached(path, ttl)
    if cached is not None:
        return cached
    try:
        resp = http.get(url, timeout=90, user_agent="nfl-prediction-engine/0.1")
        resp.raise_for_status()
        buf = io.BytesIO(resp.content)
        frame = pd.read_csv(buf) if csv else pd.read_parquet(buf)
    except (requests.RequestException, ValueError, OSError) as exc:
        log.warning("nflverse fetch failed (%s): %s", name, exc)
        stale = _read_cached(path, None)
        return stale if stale is not None else pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path)
    except (OSError, ValueError) as exc:
        log.warning("could not cache %s: %s", name, exc)
    return frame


def _season_ttl(season: int) -> int | None:
    """No expiry for a finished season; ``LIVE_TTL`` for the one in progress."""
    return None if season < _current_season() else LIVE_TTL


def _seasonal(asset: str, season: int, subdir: str | None = None) -> pd.DataFrame:
    folder = subdir or asset
    url = f"{_RELEASES}/{folder}/{asset}_{season}.parquet"
    return _fetch(url, f"{asset}_{season}.parquet", _season_ttl(season))


# -- the spine --------------------------------------------------------------
def games() -> pd.DataFrame:
    """Every scheduled and completed game, 1999 to date.

    Carries the closing ``spread_line``, ``total_line`` and both moneylines, plus
    rest days, roof, surface, temperature, wind, starting quarterbacks and
    coaches. Future games appear with null scores, which is how a live slate is
    matched to its environment.
    """
    return _fetch(GAMES_URL, "games.csv.parquet", LIVE_TTL, csv=True)


def graded_games(first_season: int = PBP_FIRST_SEASON) -> pd.DataFrame:
    """Completed games with a closing spread and total -- the backtest sample."""
    frame = games()
    if frame.empty:
        return frame
    need = ["result", "total", "spread_line", "total_line", "home_score", "away_score"]
    missing = [col for col in need if col not in frame.columns]
    if missing:
        log.warning("game file missing columns: %s", ",".join(missing))
        return pd.DataFrame()
    out = frame[frame.season >= first_season].dropna(subset=need)
    return out.reset_index(drop=True)


# -- play level -------------------------------------------------------------
def play_by_play(season: int) -> pd.DataFrame:
    """Every play of a season, with ``epa``, ``success``, ``xpass`` and the drive
    fields the score simulator's per-drive rates are measured from."""
    return _seasonal("play_by_play", season, subdir="pbp")


def participation(season: int) -> pd.DataFrame:
    """Per-play personnel: who was on the field, and whether a player ran a route.

    2016 onward. This is the only public source for routes run, which is what
    makes targets-per-route-run computable -- a receiving prop's most repeatable
    input (r=0.78 for wide receivers, split-half within season).
    """
    if season < PARTICIPATION_FIRST_SEASON:
        return pd.DataFrame()
    return _seasonal("pbp_participation", season, subdir="pbp_participation")


def snap_counts(season: int) -> pd.DataFrame:
    """Offensive, defensive and special-teams snaps per player-game (2012 on)."""
    if season < SNAPS_FIRST_SEASON:
        return pd.DataFrame()
    return _seasonal("snap_counts", season, subdir="snap_counts")


# -- player and team weeks --------------------------------------------------
def player_week(season: int) -> pd.DataFrame:
    """Weekly player lines: targets, target share, air yards, carries, attempts."""
    return _seasonal("stats_player_week", season, subdir="stats_player")


def team_week(season: int) -> pd.DataFrame:
    """Weekly team lines, the box-score view of the same games."""
    return _seasonal("stats_team_week", season, subdir="stats_team")


def rosters(season: int) -> pd.DataFrame:
    """Season roster, used to attach a position to a player id.

    Not cosmetic: measuring targets-per-route-run over *everyone* on the field
    reads r=0.95, because offensive linemen run pass-protection "routes" and are
    never targeted, so between-position variance masquerades as reliability.
    Filtered to receivers, tight ends and backs it is 0.72.
    """
    return _seasonal("roster", season, subdir="rosters")


def depth_charts(season: int) -> pd.DataFrame:
    return _seasonal("depth_charts", season, subdir="depth_charts")


def injuries(season: int) -> pd.DataFrame:
    """Official weekly injury reports: practice status and game designation."""
    return _seasonal("injuries", season, subdir="injuries")


def pfr_advstats(season: int, kind: str = "pass") -> pd.DataFrame:
    """Pro-Football-Reference advanced stats: pressures, blitzes, broken tackles.

    The free stand-in for ESPN's pass-rush / pass-block win rates, which are
    published weekly with no bulk archive and therefore cannot be fitted on.
    ``kind`` is one of ``pass``, ``rush``, ``rec``, ``def``.
    """
    asset = f"advstats_season_{kind}"
    url = f"{_RELEASES}/pfr_advstats/{asset}.parquet"
    frame = _fetch(url, f"{asset}.parquet", LIVE_TTL)
    if frame.empty or "season" not in frame.columns:
        return frame
    return frame[frame.season == season].reset_index(drop=True)


def next_gen_stats(season: int, kind: str = "receiving") -> pd.DataFrame:
    """Next Gen Stats weekly tracking aggregates (2016 on).

    ``kind`` is ``passing`` (CPOE, time to throw, ADOT), ``receiving``
    (separation, air-yard share, YAC over expected) or ``rushing`` (expected
    rush yards, rush yards over expected).
    """
    if season < NGS_FIRST_SEASON:
        return pd.DataFrame()
    asset = f"ngs_{kind}"
    url = f"{_RELEASES}/nextgen_stats/{asset}.parquet"
    frame = _fetch(url, f"{asset}.parquet", LIVE_TTL)
    if frame.empty or "season" not in frame.columns:
        return frame
    return frame[frame.season == season].reset_index(drop=True)


def load_seasons(loader: str, seasons: list[int]) -> pd.DataFrame:
    """Concatenate one seasonal loader over several seasons.

    ``loader`` names a module-level function taking a season. Seasons that come
    back empty are skipped rather than fatal, so a partial history still fits.
    """
    fn = _LOADERS.get(loader)
    if fn is None:
        raise KeyError(f"unknown loader: {loader}")
    frames = [frame for frame in (fn(season) for season in seasons) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_LOADERS = {
    "play_by_play": play_by_play,
    "participation": participation,
    "snap_counts": snap_counts,
    "player_week": player_week,
    "team_week": team_week,
    "rosters": rosters,
    "depth_charts": depth_charts,
    "injuries": injuries,
}
