"""Official NHL API (``api-web.nhle.com``): schedule, finals, starters.

Keyless. Three endpoints carry everything Phase 0 needs:

* ``/v1/schedule/<date>`` -- the week's games with ``gameState`` and start times.
* ``/v1/gamecenter/<id>/right-rail`` -- ``linescore.byPeriod`` (per-period
  scores incl. OT and SO rows) and the shootout summary.
* ``/v1/gamecenter/<id>/boxscore`` -- ``playerByGameStats`` with a ``starter``
  flag on goalies; the roster of record for grading and the goalie audit.

Field shapes were confirmed against game 2025021301 (BUF-DAL, SO) on 2026-09-28.
Parsers are pure functions over the JSON so fixtures can drive the tests.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date as Date
from pathlib import Path

import requests

from mlb_engine.data import http
from nhl_engine.data.teamnames import canonical
from nhl_engine.schemas import Game, GameResult, PeriodScore

log = logging.getLogger(__name__)

BASE = "https://api-web.nhle.com/v1"
REGULAR_SEASON = 2
PLAYOFFS = 3


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

    def _get_json(self, url: str) -> object:
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / (url.replace(BASE + "/", "").replace("/", "_") + ".json")
            if cache.exists() and time.time() - cache.stat().st_mtime < self.cache_ttl:
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


__all__ = ["BASE", "NHLAPIClient", "parse_result", "parse_schedule"]
