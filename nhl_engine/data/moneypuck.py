"""MoneyPuck public CSVs: game-by-game team and goalie logs, season summaries.

Keyless, static files, refreshed by MoneyPuck daily during the season. Three
families are used:

* ``careers/gameByGame/regular/teams/<CODE>.csv`` -- one row per game per
  strength state (``all``, ``5on5``, ``5on4``, ``4on5``, ``other``) since 2008,
  with ``gameDate`` and ``iceTime`` (seconds), score/venue-adjusted xG, shot
  attempts, danger bands, penalties. This is the as-of source for every team
  rate in §5.1: filter on ``gameDate < slate`` and nothing from the future
  leaks in. Some franchises are split across two files by era (``L.A`` then
  ``LAK``); the client concatenates the variants.
* ``careers/gameByGame/regular/goalies/<playerId>.csv`` -- same shape for a
  goalie (``icetime``, ``xGoals``, ``goals``), the input to GSAx/60.
* ``seasonSummary/<season>/regular/{teams,goalies,skaters}.csv`` -- season
  totals; used only to enumerate ids and as a cross-check, never as an as-of
  input (a season file always contains the whole season).

MoneyPuck's shot model already applies a rink adjustment to coordinates
(``arenaAdjusted*`` columns in the shot files), so the xG columns here are
venue-corrected upstream; the §3 arena-bias check measures what survives.
Hits/giveaways/takeaways are present in the files and deliberately *not*
exposed as metrics (tracking-biased; card context only).

Everything downloaded is cached under ``cache_dir/moneypuck`` with a TTL, so
a study over ten seasons costs one download per file, and the tests run from
fixtures without touching the network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import date as Date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from mlb_engine.data import http
from nhl_engine.data.teamnames import canonical

log = logging.getLogger(__name__)

BASE = "https://moneypuck.com/moneypuck/playerData"
SITUATIONS = ("all", "5on5", "5on4", "4on5", "other")

# File names MoneyPuck uses for each franchise, oldest era last.
_FILE_VARIANTS: dict[str, tuple[str, ...]] = {
    "LAK": ("LAK", "L.A"),
    "NJD": ("NJD", "N.J"),
    "SJS": ("SJS", "S.J"),
    "TBL": ("TBL", "T.B"),
}
# Codes that appear inside the files (``playerTeam`` / ``opposingTeam``).
_IN_FILE_CODES: dict[str, str] = {"L.A": "LAK", "N.J": "NJD", "S.J": "SJS", "T.B": "TBL"}
# The Utah franchise's history is Arizona's; a preseason prior for UTA in 2024
# reads ARI's 2023. Nothing else is carried across a relocation.
FRANCHISE_PREDECESSOR: dict[str, str] = {"UTA": "ARI"}

TEAM_COLUMNS = (
    "team", "season", "gameId", "playerTeam", "opposingTeam", "home_or_away", "gameDate",
    "situation", "iceTime", "xGoalsFor", "xGoalsAgainst", "scoreVenueAdjustedxGoalsFor",
    "scoreVenueAdjustedxGoalsAgainst", "goalsFor", "goalsAgainst", "shotAttemptsFor",
    "shotAttemptsAgainst", "scoreAdjustedShotsAttemptsFor", "scoreAdjustedShotsAttemptsAgainst",
    "shotsOnGoalFor", "shotsOnGoalAgainst", "highDangerxGoalsFor", "highDangerxGoalsAgainst",
    "highDangerShotsFor", "highDangerShotsAgainst", "penaltiesFor", "penaltiesAgainst",
    "penalityMinutesFor", "penalityMinutesAgainst",
)  # fmt: skip
GOALIE_COLUMNS = (
    "playerId", "season", "name", "gameId", "playerTeam", "opposingTeam", "home_or_away",
    "gameDate", "situation", "icetime", "xGoals", "goals", "ongoal", "unblocked_shot_attempts",
)  # fmt: skip


def mp_code(code: str) -> str:
    """Canonical NHL code for a code as spelled inside a MoneyPuck file."""
    return canonical(_IN_FILE_CODES.get(code, code))


class MoneyPuckClient:
    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        cache_ttl: int = 6 * 3600,
        timeout: int = 60,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.timeout = timeout

    # -- game-by-game ----------------------------------------------------

    def team_games(self, code: str) -> pd.DataFrame:
        """Every regular-season game row for a franchise code, all eras, all situations."""
        code = canonical(code)
        frames = []
        for name in _FILE_VARIANTS.get(code, (code,)):
            raw = self._get(f"{BASE}/careers/gameByGame/regular/teams/{name}.csv")
            if raw is not None:
                frames.append(parse_team_games(raw))
        if not frames:
            return pd.DataFrame(columns=list(TEAM_COLUMNS))
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(["gameId", "situation"])
            .sort_values(["gameDate", "gameId"], kind="stable")
            .reset_index(drop=True)
        )

    def goalie_games(self, player_id: int) -> pd.DataFrame:
        raw = self._get(f"{BASE}/careers/gameByGame/regular/goalies/{int(player_id)}.csv")
        if raw is None:
            return pd.DataFrame(columns=list(GOALIE_COLUMNS))
        return parse_goalie_games(raw)

    # -- season summaries (id enumeration / cross-checks only) -------------

    def season_goalies(self, season: int) -> pd.DataFrame:
        raw = self._get(f"{BASE}/seasonSummary/{season}/regular/goalies.csv")
        return pd.read_csv(StringIO(raw)) if raw is not None else pd.DataFrame()

    def season_teams(self, season: int) -> pd.DataFrame:
        raw = self._get(f"{BASE}/seasonSummary/{season}/regular/teams.csv")
        return pd.read_csv(StringIO(raw)) if raw is not None else pd.DataFrame()

    def season_skaters(self, season: int) -> pd.DataFrame:
        raw = self._get(f"{BASE}/seasonSummary/{season}/regular/skaters.csv")
        return pd.read_csv(StringIO(raw)) if raw is not None else pd.DataFrame()

    def goalie_ids(self, seasons: Iterable[int]) -> list[int]:
        ids: set[int] = set()
        for season in seasons:
            df = self.season_goalies(season)
            if not df.empty and "playerId" in df:
                ids.update(int(x) for x in df["playerId"].dropna().unique())
        return sorted(ids)

    # -- transport -----------------------------------------------------------

    def _get(self, url: str) -> str | None:
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / "moneypuck" / url.replace(BASE + "/", "").replace("/", "_")
            if cache.exists() and time.time() - cache.stat().st_mtime < self.cache_ttl:
                return cache.read_text()
        try:
            resp = http.get(url, timeout=self.timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("MoneyPuck request failed (%s): %s", url, exc)
            if cache is not None and cache.exists():
                return cache.read_text()
            return None
        text = resp.text
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text)
        return text


# -- pure parsers ---------------------------------------------------------------


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gameDate"] = pd.to_datetime(df["gameDate"].astype(int).astype(str), format="%Y%m%d").dt.date
    df["gameId"] = df["gameId"].astype(int)
    df["season"] = df["season"].astype(int)
    return df


def parse_team_games(raw: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw), usecols=lambda c: c in TEAM_COLUMNS)
    df = _parse_dates(df)
    df["playerTeam"] = df["playerTeam"].map(mp_code)
    df["opposingTeam"] = df["opposingTeam"].map(mp_code)
    df["team"] = df["team"].map(mp_code)
    return df


def parse_goalie_games(raw: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(raw), usecols=lambda c: c in GOALIE_COLUMNS)
    df = _parse_dates(df)
    df["playerTeam"] = df["playerTeam"].map(mp_code)
    df["opposingTeam"] = df["opposingTeam"].map(mp_code)
    df["playerId"] = df["playerId"].astype(int)
    return df


def as_of(df: pd.DataFrame, slate: Date, *, season: int | None = None) -> pd.DataFrame:
    """Rows from games played strictly before ``slate`` (optionally one season only)."""
    mask = df["gameDate"] < slate
    if season is not None:
        mask &= df["season"] == season
    return df.loc[mask]


def season_of(day: Date) -> int:
    """MoneyPuck season label: the year the season started (Oct 2025 -> 2025)."""
    return day.year if day.month >= 9 else day.year - 1


__all__ = [
    "FRANCHISE_PREDECESSOR",
    "GOALIE_COLUMNS",
    "SITUATIONS",
    "TEAM_COLUMNS",
    "MoneyPuckClient",
    "as_of",
    "mp_code",
    "parse_goalie_games",
    "parse_team_games",
    "season_of",
]
