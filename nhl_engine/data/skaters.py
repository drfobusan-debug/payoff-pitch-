"""Player game logs and current rosters from the NHL API (master plan §5.5).

Two endpoints, both cached by ``NHLAPIClient``:

- ``/v1/player/{id}/game-log/{season}/2`` -- one row per regular-season game:
  TOI, SOG, G, A, PTS, PPP for skaters; shots against, goals against, TOI for
  goalies. Finished seasons are cached forever, the running season briefly.
- ``/v1/roster/{TEAM}/current`` -- who dresses for a team, used only to map a
  book's player name (``"Connor McDavid"``) to an NHL player id.

Nothing here projects anything; ``features.props`` reads these logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date

from nhl_engine.data.nhlapi import BASE, FOREVER, REGULAR_SEASON, NHLAPIClient, RosterSpot
from nhl_engine.data.rotowire import norm_name

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkaterGame:
    player_id: int
    game_id: int
    game_date: Date
    team: str
    opponent: str
    home: bool
    toi: float  # seconds
    sog: int
    goals: int
    assists: int
    points: int
    ppp: int


@dataclass(frozen=True)
class GoalieGame:
    player_id: int
    game_id: int
    game_date: Date
    team: str
    opponent: str
    home: bool
    toi: float
    shots_against: int
    goals_against: int
    started: bool

    @property
    def saves(self) -> int:
        return self.shots_against - self.goals_against


def _toi(text: object) -> float:
    try:
        m, s = str(text).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return 0.0


def _int(x: object) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    try:
        return int(str(x))
    except ValueError:
        return 0


def parse_skater_log(player_id: int, data: dict) -> list[SkaterGame]:
    out: list[SkaterGame] = []
    for g in data.get("gameLog", []):
        if not isinstance(g, dict) or "shots" not in g:
            continue
        try:
            day = Date.fromisoformat(str(g.get("gameDate", "")))
        except ValueError:
            continue
        out.append(
            SkaterGame(
                player_id=player_id,
                game_id=_int(g.get("gameId")),
                game_date=day,
                team=str(g.get("teamAbbrev", "")),
                opponent=str(g.get("opponentAbbrev", "")),
                home=g.get("homeRoadFlag") == "H",
                toi=_toi(g.get("toi")),
                sog=_int(g.get("shots")),
                goals=_int(g.get("goals")),
                assists=_int(g.get("assists")),
                points=_int(g.get("points")),
                ppp=_int(g.get("powerPlayPoints")),
            )
        )
    out.sort(key=lambda g: g.game_date)
    return out


def parse_goalie_log(player_id: int, data: dict) -> list[GoalieGame]:
    out: list[GoalieGame] = []
    for g in data.get("gameLog", []):
        if not isinstance(g, dict) or "shotsAgainst" not in g:
            continue
        try:
            day = Date.fromisoformat(str(g.get("gameDate", "")))
        except ValueError:
            continue
        out.append(
            GoalieGame(
                player_id=player_id,
                game_id=_int(g.get("gameId")),
                game_date=day,
                team=str(g.get("teamAbbrev", "")),
                opponent=str(g.get("opponentAbbrev", "")),
                home=g.get("homeRoadFlag") == "H",
                toi=_toi(g.get("toi")),
                shots_against=_int(g.get("shotsAgainst")),
                goals_against=_int(g.get("goalsAgainst")),
                started=_int(g.get("gamesStarted")) == 1,
            )
        )
    out.sort(key=lambda g: g.game_date)
    return out


def parse_current_roster(team: str, data: dict) -> list[RosterSpot]:
    out: list[RosterSpot] = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            if not isinstance(p, dict):
                continue
            first = p.get("firstName", {})
            last = p.get("lastName", {})
            name = " ".join(
                s
                for s in (
                    str(first.get("default", "")) if isinstance(first, dict) else "",
                    str(last.get("default", "")) if isinstance(last, dict) else "",
                )
                if s
            )
            out.append(RosterSpot(_int(p.get("id")), team, str(p.get("positionCode", "")), name))
    return out


class PlayerLogClient:
    """Game logs and rosters over the shared NHL API client/cache."""

    def __init__(self, api: NHLAPIClient):
        self.api = api

    def _log(self, player_id: int, season: int, *, final: bool) -> dict:
        url = f"{BASE}/player/{player_id}/game-log/{season}{season + 1}/{REGULAR_SEASON}"
        data = self.api._get_json(url, ttl=FOREVER if final else None)
        return data if isinstance(data, dict) else {}

    def skater_log(self, player_id: int, season: int, *, final: bool = False) -> list[SkaterGame]:
        return parse_skater_log(player_id, self._log(player_id, season, final=final))

    def goalie_log(self, player_id: int, season: int, *, final: bool = False) -> list[GoalieGame]:
        return parse_goalie_log(player_id, self._log(player_id, season, final=final))

    def current_roster(self, team: str) -> list[RosterSpot]:
        data = self.api._get_json(f"{BASE}/roster/{team}/current")
        return parse_current_roster(team, data) if isinstance(data, dict) else []


class NameIndex:
    """Book player name -> roster spot, per team, on normalised full names.

    A book spells "Alex Ovechkin"; the roster says "Alexander Ovechkin": a last
    name that is unique on the team is accepted as the fallback.
    """

    def __init__(self, spots: list[RosterSpot]):
        self.by_full: dict[tuple[str, str], RosterSpot] = {}
        self.by_last: dict[tuple[str, str], list[RosterSpot]] = {}
        for s in spots:
            full = norm_name(s.name)
            self.by_full[(s.team, full)] = s
            last = full.split(" ")[-1] if full else ""
            self.by_last.setdefault((s.team, last), []).append(s)

    def find(self, team: str, name: str) -> RosterSpot | None:
        full = norm_name(name)
        hit = self.by_full.get((team, full))
        if hit is not None:
            return hit
        last = full.split(" ")[-1] if full else ""
        cands = self.by_last.get((team, last), [])
        if len(cands) == 1 and full and cands[0].name and (norm_name(cands[0].name)[0] == full[0]):
            return cands[0]
        return None


__all__ = [
    "GoalieGame",
    "NameIndex",
    "PlayerLogClient",
    "SkaterGame",
    "parse_current_roster",
    "parse_goalie_log",
    "parse_skater_log",
]
