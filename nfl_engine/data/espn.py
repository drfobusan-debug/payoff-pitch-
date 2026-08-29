"""ESPN's FPI game projections: an outside forecast to be judged against.

ESPN publishes, per team per game, the Football Power Index's win probability and
predicted point differential. Both persist after the game is played, at

    sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event}
        /competitions/{event}/powerindex/{team}

which needs no key, and -- unlike every prop benchmark on the MLB side -- can be
read for seasons already finished. That is worth saying plainly: FPI is the first
NFL benchmark with an *archive*, so it can be scored over hundreds of past games
instead of only from the day capture starts.

It is a benchmark and never an input. Nothing here touches a probability, a
price, a screen or a tier: the numbers are stored on their own ledger rows,
graded against the same final score, and excluded from every measurement of the
engine. The reason to keep it is the reason CLV exists -- grading ourselves
against ourselves is no test at all.

Two things this reader is careful about:

* **A benchmark going dark must not take a slate down.** Every fetch is
  best-effort; a game whose projection cannot be read is dropped, not guessed at.
* **The two sides need not sum to one.** ESPN publishes each team's number from
  its own BPI run, and 58.2% against 41.5% happens. The published numbers are
  kept as published and never renormalised, because a benchmark rescaled by us is
  partly ours.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from mlb_engine.data import http
from nfl_engine.config import cache_dir
from nfl_engine.data import nflverse

log = logging.getLogger(__name__)

FPI = "fpi"
_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
# Seconds before a week that is not finished yet is refetched. FPI moves during
# the week -- an injury, a line move -- so a captured projection is only the read
# at the moment it was taken, and the week in progress is not cached for long.
LIVE_TTL = 3 * 3600
_TIMEOUT = 30

# ESPN's numeric team ids, which are stable, and the two abbreviations where ESPN
# and nflverse disagree: ESPN writes LAR and WSH where nflverse writes LA and WAS.
# Left unmapped, the Rams' and Commanders' games silently lose their benchmark.
# fmt: off
TEAM_IDS = {
    "ATL": 1, "BUF": 2, "CHI": 3, "CIN": 4, "CLE": 5, "DAL": 6, "DEN": 7,
    "DET": 8, "GB": 9, "TEN": 10, "IND": 11, "KC": 12, "LV": 13, "LA": 14,
    "MIA": 15, "MIN": 16, "NE": 17, "NO": 18, "NYG": 19, "NYJ": 20, "PHI": 21,
    "ARI": 22, "PIT": 23, "LAC": 24, "SF": 25, "SEA": 26, "TB": 27, "WAS": 28,
    "CAR": 29, "JAX": 30, "BAL": 33, "HOU": 34,
}
# fmt: on


@dataclass(frozen=True)
class FpiGame:
    """FPI's read on one game, on the home team's axis.

    ``home_prob`` is FPI's win probability for the home side and ``home_margin``
    its predicted point differential for that side, both exactly as published.
    """

    season: int
    week: int
    date: str
    matchup: str
    home: str
    away: str
    home_prob: float
    home_margin: float

    @property
    def pick(self) -> str:
        """The side FPI has ahead. Empty when it calls the game even."""
        if self.home_prob > 50.0:
            return self.home
        if self.home_prob < 50.0:
            return self.away
        return ""

    @property
    def pick_prob(self) -> float:
        """FPI's probability for the side it has ahead, as a fraction."""
        prob = self.home_prob if self.home_prob >= 50.0 else 100.0 - self.home_prob
        return round(prob / 100.0, 6)


def _cache_path(season: int, week: int) -> Path:
    return cache_dir() / "espn" / f"fpi_{season}_wk{week}.json"


def _cached(path: Path, ttl: int | None) -> list[FpiGame] | None:
    if not path.exists():
        return None
    if ttl is not None and time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [FpiGame(**row) for row in raw]
    except (OSError, ValueError, TypeError) as exc:  # truncated or older shape
        log.warning("FPI cache unreadable (%s): %s", path.name, exc)
        return None


def _store(path: Path, games: list[FpiGame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps([asdict(g) for g in games]), encoding="utf-8")
    except OSError as exc:
        log.warning("could not cache FPI (%s): %s", path.name, exc)


def _get(url: str) -> dict[str, Any] | None:
    try:
        resp = http.get(url, timeout=_TIMEOUT, user_agent="nfl-prediction-engine/0.1")
        resp.raise_for_status()
        body = resp.json()
    except (requests.RequestException, ValueError) as exc:
        log.debug("FPI fetch failed (%s): %s", url, exc)
        return None
    return body if isinstance(body, dict) else None


def _stat(body: dict[str, Any], name: str) -> float | None:
    for stat in body.get("stats") or ():
        if isinstance(stat, dict) and stat.get("name") == name:
            value = stat.get("value")
            if isinstance(value, int | float):
                return float(value)
    return None


def game_projection(event_id: str, home: str) -> tuple[float, float] | None:
    """FPI's win probability and predicted margin for the home side of one event.

    ``None`` when ESPN has no projection for it -- which is normal for a game that
    was never given one, and must cost the caller this game only.
    """
    team = TEAM_IDS.get(home)
    if team is None or not event_id:
        return None
    body = _get(f"{_CORE}/events/{event_id}/competitions/{event_id}/powerindex/{team}")
    if body is None:
        return None
    prob = _stat(body, "gameprojection")
    margin = _stat(body, "teampredptdiff")
    if prob is None or margin is None:
        return None
    return (prob, margin)


def projections(season: int, week: int) -> list[FpiGame]:
    """Every game of one week that FPI has a projection for, cached on disk.

    One request per game, keyed off the ESPN event id nflverse already carries, so
    no name matching is involved. A finished week is cached forever; the week in
    progress expires after :data:`LIVE_TTL`, because FPI is revised as the week
    runs and the point of capturing it is the read at the time.
    """
    schedule = nflverse.games()
    if schedule.empty:
        return []
    rows = schedule[(schedule.season == season) & (schedule.week == week)]
    if rows.empty:
        return []
    played = bool(rows.home_score.notna().all())
    path = _cache_path(season, week)
    cached = _cached(path, None if played else LIVE_TTL)
    if cached is not None:
        return cached
    out: list[FpiGame] = []
    for row in rows.itertuples():
        # nflverse leaves the ESPN id null for a game it has not keyed yet, and
        # ``str(nan)`` would send "nan" to the API as an event.
        if pd.isna(row.espn):
            continue
        read = game_projection(str(row.espn).split(".")[0], str(row.home_team))
        if read is None:
            continue
        out.append(
            FpiGame(
                season=season,
                week=week,
                date=str(row.gameday),
                matchup=f"{row.away_team} @ {row.home_team}",
                home=str(row.home_team),
                away=str(row.away_team),
                home_prob=read[0],
                home_margin=read[1],
            )
        )
    if out:
        _store(path, out)
    else:
        log.warning("no FPI projections read for %d week %d", season, week)
    return out
