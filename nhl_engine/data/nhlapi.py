"""Official NHL API (``api-web.nhle.com``): schedule, finals, starters.

Keyless. Three endpoints carry everything Phase 0 needs:

* ``/v1/schedule/<date>`` -- the week's games with ``gameState`` and start times.
* ``/v1/gamecenter/<id>/right-rail`` -- ``linescore.byPeriod`` (per-period
  scores incl. OT and SO rows) and the shootout summary.
* ``/v1/gamecenter/<id>/boxscore`` -- ``playerByGameStats`` with a ``starter``
  flag on goalies; the roster of record for grading and the goalie audit.

Phase 1 lineup work adds the shift-level inputs for the isolated-impact fit
(master plan section 5.7):

* ``/v1/gamecenter/<id>/play-by-play`` -- ``rosterSpots`` (who dressed, with
  position, so goalies can be told from skaters) and the event stream.
* ``https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id>`` --
  every shift with period, start/end clock and player id (typeCode 517 = shift;
  goal rows carry other codes and are dropped).

Finished games are immutable, so those two are cached without expiry.

Field shapes were confirmed against games 2025021301 (BUF-DAL, SO) on 2026-09-28
and 2024020500 (CAR home) on 2026-09-29. Parsers are pure functions over the
JSON so fixtures can drive the tests.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import requests

from mlb_engine.data import http
from nhl_engine.data.teamnames import canonical
from nhl_engine.schemas import Game, GameResult, PeriodScore

log = logging.getLogger(__name__)

BASE = "https://api-web.nhle.com/v1"
STATS_BASE = "https://api.nhle.com/stats/rest/en"
REGULAR_SEASON = 2
PLAYOFFS = 3
SHIFT_TYPE = 517
FOREVER = 10 * 365 * 24 * 3600


@dataclass(frozen=True)
class RosterSpot:
    player_id: int
    team: str
    position: str  # C/L/R/D/G
    name: str


@dataclass(frozen=True)
class Shift:
    player_id: int
    team: str
    period: int
    start: int  # seconds into the period
    end: int


def regular_season_game_ids(season: int, games: int = 1312) -> list[int]:
    """NHL regular-season ids are ``<season>02<0001..N>``; 1312 games for 32 teams."""
    return [int(f"{season}02{n:04d}") for n in range(1, games + 1)]


class NHLAPIClient:
    def __init__(self, timeout: int = 20, *, cache_dir: Path | None = None, cache_ttl: int = 300):
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

    def schedule(self, day: Date) -> list[Game]:
        data = self._get_json(f"{BASE}/schedule/{day.isoformat()}")
        return parse_schedule(data, day) if isinstance(data, dict) else []

    def result(self, game: Game) -> GameResult | None:
        if game.nhl_game_id is None:
            return None
        rail = self._get_json(f"{BASE}/gamecenter/{game.nhl_game_id}/right-rail")
        box = self._get_json(f"{BASE}/gamecenter/{game.nhl_game_id}/boxscore")
        if not isinstance(box, dict):
            return None
        return parse_result(box, rail if isinstance(rail, dict) else {}, game)

    def play_by_play(self, game_id: int, *, final: bool = True) -> dict | None:
        data = self._get_json(
            f"{BASE}/gamecenter/{game_id}/play-by-play", ttl=FOREVER if final else None
        )
        return data if isinstance(data, dict) else None

    def shifts(self, game_id: int, *, final: bool = True) -> list[Shift]:
        data = self._get_json(
            f"{STATS_BASE}/shiftcharts?cayenneExp=gameId={game_id}",
            ttl=FOREVER if final else None,
        )
        return parse_shifts(data) if isinstance(data, dict) else []

    def roster(self, game_id: int, *, final: bool = True) -> list[RosterSpot]:
        pbp = self.play_by_play(game_id, final=final)
        return parse_roster_spots(pbp) if pbp else []

    def _get_json(self, url: str, *, ttl: int | None = None) -> object:
        cache = None
        ttl = self.cache_ttl if ttl is None else ttl
        if self.cache_dir is not None:
            name = url.replace(BASE + "/", "").replace(STATS_BASE + "/", "stats_")
            for ch in "/?=":
                name = name.replace(ch, "_")
            cache = self.cache_dir / (name + ".json")
            if cache.exists() and time.time() - cache.stat().st_mtime < ttl:
                try:
                    return json.loads(cache.read_text())
                except ValueError:
                    pass
        try:
            resp = http.get(url, timeout=self.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("NHL API request failed (%s): %s", url, exc)
            return None
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
        return payload


def parse_schedule(
    data: dict, day: Date, *, game_types: tuple[int, ...] = (REGULAR_SEASON, PLAYOFFS)
) -> list[Game]:
    """Games on ``day`` only; the endpoint returns the surrounding week."""
    out: list[Game] = []
    for week_day in data.get("gameWeek", []):
        if not isinstance(week_day, dict) or week_day.get("date") != day.isoformat():
            continue
        for raw in week_day.get("games", []):
            if not isinstance(raw, dict) or raw.get("gameType") not in game_types:
                continue
            home = raw.get("homeTeam", {}).get("abbrev")
            away = raw.get("awayTeam", {}).get("abbrev")
            gid = raw.get("id")
            if not (home and away and isinstance(gid, int)):
                continue
            out.append(
                Game(
                    game_date=day,
                    home=canonical(str(home)),
                    away=canonical(str(away)),
                    start_utc=str(raw.get("startTimeUTC", "")),
                    nhl_game_id=gid,
                )
            )
    out.sort(key=lambda g: g.start_utc)
    return out


def parse_result(box: dict, rail: dict, game: Game) -> GameResult:
    periods: list[PeriodScore] = []
    for p in rail.get("linescore", {}).get("byPeriod", []):
        if not isinstance(p, dict):
            continue
        desc = p.get("periodDescriptor", {})
        try:
            periods.append(
                PeriodScore(
                    number=int(desc.get("number", 0)),
                    kind=str(desc.get("periodType", "REG")),
                    away=int(p.get("away", 0)),
                    home=int(p.get("home", 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    outcome = box.get("gameOutcome", {})
    decided = str(outcome.get("lastPeriodType", "")) if isinstance(outcome, dict) else ""
    return GameResult(
        nhl_game_id=int(box.get("id", game.nhl_game_id or 0)),
        game_date=game.game_date,
        away=game.away,
        home=game.home,
        state=str(box.get("gameState", "")),
        periods=tuple(periods),
        decided=decided,
        away_starter=_starter(box, "awayTeam"),
        home_starter=_starter(box, "homeTeam"),
    )


def _starter(box: dict, side: str) -> str:
    goalies = box.get("playerByGameStats", {}).get(side, {}).get("goalies", [])
    for g in goalies:
        if isinstance(g, dict) and g.get("starter"):
            name = g.get("name", {})
            return str(name.get("default", "")) if isinstance(name, dict) else str(name)
    return ""


def _clock(text: object) -> int | None:
    try:
        mm, ss = str(text).split(":")
        return int(mm) * 60 + int(ss)
    except (TypeError, ValueError):
        return None


def parse_shifts(data: dict) -> list[Shift]:
    """Shift rows only (typeCode 517), zero-length shifts dropped."""
    out: list[Shift] = []
    for raw in data.get("data", []):
        if not isinstance(raw, dict) or raw.get("typeCode") != SHIFT_TYPE:
            continue
        start, end = _clock(raw.get("startTime")), _clock(raw.get("endTime"))
        pid, team, period = raw.get("playerId"), raw.get("teamAbbrev"), raw.get("period")
        if start is None or end is None or end <= start:
            continue
        if not (isinstance(pid, int) and team and isinstance(period, int)):
            continue
        out.append(Shift(pid, canonical(str(team)), period, start, end))
    out.sort(key=lambda s: (s.period, s.start, s.player_id))
    return out


def parse_roster_spots(pbp: dict) -> list[RosterSpot]:
    teams = {}
    for side in ("homeTeam", "awayTeam"):
        t = pbp.get(side, {})
        if isinstance(t, dict) and "id" in t and "abbrev" in t:
            teams[t["id"]] = canonical(str(t["abbrev"]))
    out: list[RosterSpot] = []
    for raw in pbp.get("rosterSpots", []):
        if not isinstance(raw, dict):
            continue
        pid, tid = raw.get("playerId"), raw.get("teamId")
        if not isinstance(pid, int) or tid not in teams:
            continue
        first = raw.get("firstName", {})
        last = raw.get("lastName", {})
        name = " ".join(
            str(x.get("default", "")) if isinstance(x, dict) else str(x) for x in (first, last)
        ).strip()
        out.append(RosterSpot(pid, teams[tid], str(raw.get("positionCode", "")), name))
    return out


__all__ = [
    "BASE",
    "NHLAPIClient",
    "RosterSpot",
    "Shift",
    "parse_result",
    "parse_roster_spots",
    "parse_schedule",
    "parse_shifts",
    "regular_season_game_ids",
]
