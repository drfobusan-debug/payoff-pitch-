"""Team-game panel: one row per team's offence in a game.

The unit the ratings are fitted on. Play-by-play is 372 columns and 50,000 plays
a season; everything the rating layer needs collapses to seven numbers per team
per game, so the collapse is cached and the parquet is only read once.

What is measured, and why these and not the others (split-half within season,
2006-2025, 640 team-seasons, opponent-unadjusted):

    off EPA/play      r=0.578      def EPA/play allowed      r=0.349
    off success rate  r=0.588      def success allowed       r=0.404
    off PROE          r=0.630      def PROE faced            r=0.278
    drives            r=0.522      neutral sec/play          r=0.475

Two things fall out. **Defence is measured on success rate, not EPA** -- EPA
allowed repeats at 0.35 against 0.40 for success allowed, because EPA is
dominated by the explosive plays and turnovers that do not recur, and a defensive
rating built on it is substantially a rating of last month's luck. And **PROE is
the most stable number on the list**, which makes sense: pass rate over expected
is a coaching identity rather than a performance, and it is what drives clock
stoppages and therefore possessions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_engine.config import cache_dir
from nfl_engine.data import nflverse

log = logging.getLogger(__name__)

# The metrics carried per team-game. Anything not here was tested and dropped.
METRICS = ("epa", "success", "proe", "sec_per_play", "drives")

# Neutral game state for pace: within a score, before the fourth quarter. Pace
# measured over the whole game is mostly the scoreboard -- a team behind by 20
# plays fast because it is losing, which is not a tempo the next game inherits.
NEUTRAL_MAX_LEAD = 7
NEUTRAL_MAX_QTR = 3
# Plausible seconds between snaps of the same drive; outside this is a clock
# stoppage, a replay or a change of possession the drive field missed.
SNAP_GAP = (3.0, 60.0)


def _panel_path(season: int) -> Path:
    return cache_dir() / "panel" / f"team_game_{season}.parquet"


def season_panel(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """One row per (game, offence, defence) for ``season``, cached on disk."""
    path = _panel_path(season)
    if path.exists() and not refresh:
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            log.warning("panel cache unreadable (%s): %s", path.name, exc)
    frame = build_season_panel(season)
    if frame.empty:
        return frame
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path)
    except (OSError, ValueError) as exc:
        log.warning("could not cache panel %s: %s", season, exc)
    return frame


def build_season_panel(season: int) -> pd.DataFrame:
    """Collapse a season's play-by-play into the rating layer's inputs."""
    pbp = nflverse.play_by_play(season)
    if pbp.empty:
        return pd.DataFrame()
    return collapse(pbp)


def collapse(pbp: pd.DataFrame) -> pd.DataFrame:
    """The panel for whatever play-by-play is handed in.

    Kept separate from the loader so a test can drive it with a handful of rows.
    """
    need = {"play_type", "posteam", "defteam", "epa", "game_id", "season", "week"}
    missing = need - set(pbp.columns)
    if missing:
        log.warning("play-by-play missing columns: %s", ",".join(sorted(missing)))
        return pd.DataFrame()
    plays = pbp[
        pbp.play_type.isin(["pass", "run"])
        & pbp.posteam.notna()
        & pbp.defteam.notna()
        & pbp.epa.notna()
    ].copy()
    if plays.empty:
        return pd.DataFrame()
    keys = ["season", "week", "game_id", "posteam", "defteam"]
    out = (
        plays.groupby(keys, observed=True)
        .agg(
            plays=("epa", "size"),
            epa=("epa", "mean"),
            success=("success", "mean"),
            proe=("pass_oe", "mean"),
        )
        .reset_index()
    )
    out["proe"] = out["proe"] / 100.0 if out["proe"].abs().max() > 1.5 else out["proe"]
    out = out.merge(_pace(plays), on=["game_id", "posteam"], how="left")
    out = out.merge(_drives(pbp), on=["game_id", "posteam"], how="left")
    return out


def _pace(plays: pd.DataFrame) -> pd.DataFrame:
    """Seconds per snap in a neutral game state."""
    blank = pd.DataFrame(columns=["game_id", "posteam", "sec_per_play"])
    if not {"game_seconds_remaining", "fixed_drive", "play_id", "qtr"} <= set(plays.columns):
        return blank
    ordered = plays.sort_values(["game_id", "play_id"])
    gap = ordered.groupby(["game_id", "fixed_drive"], observed=True)[
        "game_seconds_remaining"
    ].diff(-1)
    lead = ordered.get("score_differential", pd.Series(0.0, index=ordered.index)).abs()
    neutral = ordered[
        (ordered.qtr <= NEUTRAL_MAX_QTR)
        & (lead <= NEUTRAL_MAX_LEAD)
        & gap.between(*SNAP_GAP)
    ].assign(gap=gap)
    if neutral.empty:
        return blank
    return (
        neutral.groupby(["game_id", "posteam"], observed=True)
        .agg(sec_per_play=("gap", "mean"))
        .reset_index()
    )


def _drives(pbp: pd.DataFrame) -> pd.DataFrame:
    """Possessions per team, which is what sets the simulator's drive count."""
    blank = pd.DataFrame(columns=["game_id", "posteam", "drives"])
    if "fixed_drive" not in pbp.columns:
        return blank
    have = pbp[pbp.posteam.notna() & pbp.fixed_drive.notna()]
    if have.empty:
        return blank
    return (
        have.groupby(["game_id", "posteam"], observed=True)
        .agg(drives=("fixed_drive", "nunique"))
        .reset_index()
    )


def panel(seasons: list[int], *, refresh: bool = False) -> pd.DataFrame:
    """Several seasons of panel, concatenated; empty seasons are skipped."""
    frames = [
        frame
        for frame in (season_panel(season, refresh=refresh) for season in seasons)
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def with_results(panel_frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach each team-game's own result and the game's closing line."""
    if panel_frame.empty or games.empty:
        return pd.DataFrame()
    cols = [
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "total_line", "home_rest",
        "away_rest", "roof", "temp", "wind", "location", "div_game",
        "home_qb_id", "away_qb_id", "stadium_id",
    ]
    have = [c for c in cols if c in games.columns]
    merged = panel_frame.merge(games[have], on=["game_id", "season", "week"], how="inner")
    merged["is_home"] = merged.posteam == merged.home_team
    merged["points_for"] = np.where(merged.is_home, merged.home_score, merged.away_score)
    merged["points_against"] = np.where(merged.is_home, merged.away_score, merged.home_score)
    return merged.dropna(subset=["points_for", "points_against"]).reset_index(drop=True)
