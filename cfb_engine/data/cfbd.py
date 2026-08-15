"""CollegeFootballData.com (CFBD) client: team power ratings and final scores.

CFBD is a free, open API (key from https://collegefootballdata.com/key). Two
things the engine needs come from here:

* **SP+ ratings** (``/ratings/sp``): each team's adjusted offense and defense,
  expressed in points scored / allowed against an average opponent. Their
  difference is the team's net rating, so they anchor both the expected margin
  (moneyline / ATS) and the expected total.
* **Final scores** (``/games``): to grade the nightly audit.

With no key the client is inert; the pipeline then falls back to
market-implied ratings so it still runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import requests

from cfb_engine.data.advanced import AdvancedBook, parse_advanced
from cfb_engine.data.portal import PortalBook, build_portal_book
from cfb_engine.data.teamnames import school_key

# Transport only, and nothing in it is baseball-specific: retries and a default
# deadline so one dropped connection does not cost a Saturday's card the way it
# cost an August one.
from mlb_engine.data import http

log = logging.getLogger(__name__)

BASE = "https://api.collegefootballdata.com"


@dataclass(frozen=True)
class TeamRating:
    """Adjusted points a team scores / allows vs an average opponent (SP+)."""

    team: str
    offense: float  # adjusted points scored
    defense: float  # adjusted points allowed (lower = stronger defense)

    @property
    def net(self) -> float:
        return self.offense - self.defense


@dataclass
class RatingBook:
    """The slate's ratings plus the league scoring average that scales them."""

    ratings: dict[str, TeamRating]  # keyed by school_key
    league_avg: float  # mean adjusted offense (== mean adjusted defense)

    def get(self, team_name: str) -> TeamRating | None:
        return self.ratings.get(school_key(team_name))


@dataclass(frozen=True)
class GameResult:
    home: str
    away: str
    home_points: int
    away_points: int


@dataclass(frozen=True)
class GameMeta:
    """Schedule metadata for one game (feeds the rest/travel/HFA filters)."""

    home: str
    away: str
    start_date: str  # ISO8601, UTC
    neutral_site: bool
    venue_id: int | None
    week: int | None
    season_type: str


@dataclass(frozen=True)
class Venue:
    venue_id: int
    latitude: float
    longitude: float
    dome: bool


@dataclass(frozen=True)
class TeamLocation:
    team: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class TeamGamePPA:
    """One team's offensive points-per-play in one game (CFBD ``/ppa/games``)."""

    game_id: int
    season: int
    week: int
    season_type: str
    team: str
    opponent: str
    offence_ppa: float
    home: bool


@dataclass(frozen=True)
class GameWeather:
    home: str
    away: str
    wind_mph: float | None
    precipitation: float | None  # inches (0 = dry)
    temperature_f: float | None
    dome: bool


class CFBDClient:
    def __init__(self, api_key: str | None, timeout: int = 25) -> None:
        self.api_key = api_key
        self.timeout = timeout
        # A full-season /games pull is reused across results + schedule filters.
        self._games_cache: dict[int, list[dict[str, object]]] = {}

    def available(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, **params: str | int) -> object:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        try:
            resp = http.get(
                f"{BASE}{path}", params=params, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("CFBD request failed (%s): %s", path, exc)
            return None

    def fetch_ratings(self, season: int) -> RatingBook | None:
        """SP+ adjusted offense/defense per team for ``season``.

        Falls back to the previous season if the current one has no ratings yet
        (early in the year SP+ is still preseason-only but keyed to the year).
        """
        if not self.available():
            return None
        data = self._get("/ratings/sp", year=season)
        if not isinstance(data, list) or not data:
            data = self._get("/ratings/sp", year=season - 1)
        if not isinstance(data, list) or not data:
            return None

        ratings: dict[str, TeamRating] = {}
        offs: list[float] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            team = row.get("team")
            off = _nested(row, "offense", "rating")
            deff = _nested(row, "defense", "rating")
            if not team or off is None or deff is None:
                continue
            ratings[school_key(str(team))] = TeamRating(str(team), float(off), float(deff))
            offs.append(float(off))
        if not ratings:
            return None
        league_avg = sum(offs) / len(offs)
        return RatingBook(ratings=ratings, league_avg=league_avg)

    def fetch_advanced(self, season: int) -> AdvancedBook:
        """Advanced efficiency stats + turnover components for ``season``.

        Garbage time is excluded. Mop-up snaps are a real if modest dilution of
        the per-play rates every threshold here is compared against: over 2024's
        134 teams, dropping them moves net PPA by .014/play on average (up to
        .059 for Navy, -.055 for Mississippi State), flips 1.0% of pairwise
        net-PPA matchups and moves 33 teams ten or more ranks in explosiveness.
        Havoc and points-per-opportunity are unaffected -- CFBD does not apply
        the filter to them.

        Falls back to the prior season if the current one is still empty
        (preseason), and returns an empty book if the key is absent or the
        endpoint is unentitled -- callers treat that as "no signal".
        """
        if not self.available():
            return parse_advanced([], {})
        adv = self._get("/stats/season/advanced", year=season, excludeGarbageTime="true")
        if not isinstance(adv, list) or not adv:
            season -= 1
            adv = self._get("/stats/season/advanced", year=season, excludeGarbageTime="true")
        if not isinstance(adv, list) or not adv:
            return parse_advanced([], {})
        adv_rows = [r for r in adv if isinstance(r, dict)]

        season_stats: dict[str, dict[str, float]] = {}
        raw = self._get("/stats/season", year=season)
        if isinstance(raw, list):
            for row in raw:
                if not isinstance(row, dict):
                    continue
                team = row.get("team")
                name = row.get("statName")
                val = row.get("statValue")
                if not team or not name or not isinstance(val, (int, float)):
                    continue
                season_stats.setdefault(school_key(str(team)), {})[str(name)] = float(val)
        return parse_advanced(adv_rows, season_stats)

    def _games(self, season: int) -> list[dict[str, object]]:
        """Every regular- and post-season game for ``season`` (cached)."""
        if season in self._games_cache:
            return self._games_cache[season]
        rows: list[dict[str, object]] = []
        if self.available():
            for season_type in ("regular", "postseason"):
                data = self._get("/games", year=season, seasonType=season_type)
                if isinstance(data, list):
                    rows.extend(r for r in data if isinstance(r, dict))
        self._games_cache[season] = rows
        return rows

    def fetch_results(self, season: int, day: Date) -> list[GameResult]:
        """Final scores for completed games kicking around ``day``.

        The odds board's slate date is US Eastern while CFBD's ``startDate`` is
        UTC, so a late Eastern kickoff lands on the next UTC day. Grading a
        one-day window on each side and matching by team name absorbs that
        boundary without mis-dropping a game.
        """
        allowed = {
            (day + timedelta(days=offset)).isoformat() for offset in (-1, 0, 1)
        }
        out: list[GameResult] = []
        for row in self._games(season):
            start = str(row.get("start_date") or row.get("startDate") or "")
            if start[:10] not in allowed:
                continue
            hp = row.get("home_points", row.get("homePoints"))
            ap = row.get("away_points", row.get("awayPoints"))
            home = row.get("home_team", row.get("homeTeam"))
            away = row.get("away_team", row.get("awayTeam"))
            if not isinstance(hp, (int, float)) or not isinstance(ap, (int, float)):
                continue
            if not home or not away:
                continue
            out.append(GameResult(str(home), str(away), int(hp), int(ap)))
        return out

    def fetch_all_results(self, season: int) -> list[GameResult]:
        """Every completed game's final score for ``season`` (backtest input)."""
        out: list[GameResult] = []
        for row in self._games(season):
            hp = row.get("home_points", row.get("homePoints"))
            ap = row.get("away_points", row.get("awayPoints"))
            home = row.get("home_team", row.get("homeTeam"))
            away = row.get("away_team", row.get("awayTeam"))
            if not isinstance(hp, (int, float)) or not isinstance(ap, (int, float)):
                continue
            if not home or not away:
                continue
            out.append(GameResult(str(home), str(away), int(hp), int(ap)))
        return out

    def fetch_schedule(self, season: int) -> list[GameMeta]:
        """Every game's schedule metadata (venue, neutral flag, kickoff)."""
        out: list[GameMeta] = []
        for row in self._games(season):
            home = row.get("home_team", row.get("homeTeam"))
            away = row.get("away_team", row.get("awayTeam"))
            start = row.get("start_date", row.get("startDate"))
            if not home or not away or not start:
                continue
            venue_id = row.get("venue_id", row.get("venueId"))
            week = row.get("week")
            out.append(
                GameMeta(
                    home=str(home),
                    away=str(away),
                    start_date=str(start),
                    neutral_site=bool(row.get("neutral_site", row.get("neutralSite", False))),
                    venue_id=int(venue_id) if isinstance(venue_id, (int, float)) else None,
                    week=int(week) if isinstance(week, (int, float)) else None,
                    season_type=str(row.get("season_type", row.get("seasonType", ""))),
                )
            )
        return out

    def fetch_team_game_ppa(self, season: int) -> list[TeamGamePPA]:
        """Per-team, per-game offensive/defensive PPA (CFBD's EPA per play)."""
        if not self.available():
            return []
        data = self._get("/ppa/games", year=season)
        if not isinstance(data, list):
            return []
        home_side = self._home_teams(season)
        out: list[TeamGamePPA] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            team = row.get("team")
            opp = row.get("opponent")
            game_id = row.get("gameId", row.get("game_id"))
            off = _nested(row, "offense", "overall")
            if not team or not opp or off is None or not isinstance(game_id, (int, float)):
                continue
            week = row.get("week")
            out.append(
                TeamGamePPA(
                    game_id=int(game_id),
                    season=season,
                    week=int(week) if isinstance(week, (int, float)) else 0,
                    season_type=str(row.get("seasonType", row.get("season_type", "regular"))),
                    team=str(team),
                    opponent=str(opp),
                    offence_ppa=off,
                    home=home_side.get(int(game_id)) == school_key(str(team)),
                )
            )
        return out

    def _home_teams(self, season: int) -> dict[int, str]:
        out: dict[int, str] = {}
        for row in self._games(season):
            gid = row.get("id")
            home = row.get("home_team", row.get("homeTeam"))
            if isinstance(gid, (int, float)) and home:
                out[int(gid)] = school_key(str(home))
        return out

    def neutral_game_ids(self, season: int) -> set[int]:
        out: set[int] = set()
        for row in self._games(season):
            gid = row.get("id")
            if not isinstance(gid, (int, float)):
                continue
            if bool(row.get("neutral_site", row.get("neutralSite", False))):
                out.add(int(gid))
        return out

    def fetch_returning_production(self, season: int) -> dict[str, float]:
        """Share of last season's production returning, keyed by school_key.

        CFBD's ``percentPPA`` is the fraction of a team's prior-year PPA that
        comes back, so 0.60 means a typically-experienced roster.
        """
        if not self.available():
            return {}
        data = self._get("/player/returning", year=season)
        if not isinstance(data, list):
            return {}
        out: dict[str, float] = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            team = row.get("team")
            pct = row.get("percentPPA", row.get("percent_ppa"))
            if team and isinstance(pct, (int, float)):
                out[school_key(str(team))] = float(pct)
        return out

    def fetch_portal(self, season: int) -> PortalBook:
        """Pre-season transfer-portal churn per team (free endpoint, 2021 on).

        Reported on the card, not priced -- :mod:`cfb_engine.data.portal` records
        what it measured against the closing spread.
        """
        if not self.available():
            return {}
        data = self._get("/player/portal", year=season)
        if not isinstance(data, list):
            return {}
        rows = [row for row in data if isinstance(row, dict)]
        return build_portal_book(rows, season)

    def fetch_venues(self) -> dict[int, Venue]:
        """Venue geo + dome flag, keyed by venue id."""
        if not self.available():
            return {}
        data = self._get("/venues")
        out: dict[int, Venue] = {}
        if not isinstance(data, list):
            return out
        for row in data:
            if not isinstance(row, dict):
                continue
            vid = row.get("id")
            lat = row.get("latitude")
            lon = row.get("longitude")
            if not isinstance(vid, (int, float)) or lat is None or lon is None:
                continue
            out[int(vid)] = Venue(
                venue_id=int(vid),
                latitude=float(lat),
                longitude=float(lon),
                dome=bool(row.get("dome", False)),
            )
        return out

    def fetch_team_locations(self, season: int) -> dict[str, TeamLocation]:
        """Each FBS team's home-stadium coordinates, keyed by school_key."""
        if not self.available():
            return {}
        data = self._get("/teams/fbs", year=season)
        out: dict[str, TeamLocation] = {}
        if not isinstance(data, list):
            return out
        for row in data:
            if not isinstance(row, dict):
                continue
            team = row.get("school")
            lat = _nested(row, "location", "latitude")
            lon = _nested(row, "location", "longitude")
            if not team or lat is None or lon is None:
                continue
            out[school_key(str(team))] = TeamLocation(str(team), lat, lon)
        return out

    def fetch_weather(self, season: int) -> list[GameWeather]:
        """Per-game forecast; empty when the key lacks the weather entitlement."""
        if not self.available():
            return []
        out: list[GameWeather] = []
        for season_type in ("regular", "postseason"):
            data = self._get("/games/weather", year=season, seasonType=season_type)
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                home = row.get("home_team", row.get("homeTeam"))
                away = row.get("away_team", row.get("awayTeam"))
                if not home or not away:
                    continue
                out.append(
                    GameWeather(
                        home=str(home),
                        away=str(away),
                        wind_mph=_opt_float(row.get("wind_speed", row.get("windSpeed"))),
                        precipitation=_opt_float(row.get("precipitation")),
                        temperature_f=_opt_float(row.get("temperature")),
                        dome=bool(row.get("game_indoors", row.get("gameIndoors", False))),
                    )
                )
        return out


def _nested(row: dict[str, object], *path: str) -> float | None:
    cur: object = row
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _opt_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
