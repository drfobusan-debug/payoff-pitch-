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

Every response is cached on disk, because CFBD bills by the call and the free
key allows 1,000 a month: one season of box scores (``/games/players``, a call
per week, re-walked on every run) was 44% of the 1,851 calls that exhausted the
first key. A played week's box score never changes, so the cache is what makes a
daily schedule affordable -- and when a call does fail (quota, revoked key,
dropped connection) an expired entry is served rather than nothing, so a card
still prices off yesterday's ratings instead of the market fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import requests

from cfb_engine.data.advanced import AdvancedBook, parse_advanced
from cfb_engine.data.portal import PortalBook, build_portal_book
from cfb_engine.data.roster import (
    ProductionBook,
    RosterBook,
    build_incoming_shares,
    parse_production,
)
from cfb_engine.data.starters import PasserGame, StarterBook, build_starter_book
from cfb_engine.data.teamnames import school_key

# Transport only, and nothing in it is baseball-specific: retries and a default
# deadline so one dropped connection does not cost a Saturday's card the way it
# cost an August one.
from mlb_engine.data import http

log = logging.getLogger(__name__)

BASE = "https://api.collegefootballdata.com"

# How long a cached response stays fresh, per endpoint. Anything CFBD recomputes
# through the week (ratings, season aggregates) takes the default; a payload that
# is final once its games are played takes the long one. ``/games`` is the one
# final payload that still has to expire inside a slate, because the nightly
# audit reads scores out of it hours after a pre-game run cached it.
DEFAULT_TTL = 6 * 3600
LONG_TTL = 30 * 86400
SHORT_TTL = 1800
ENDPOINT_TTL: dict[str, int] = {
    "/games": SHORT_TTL,
    "/games/players": LONG_TTL,
    "/ppa/games": LONG_TTL,
    "/ppa/players/season": LONG_TTL,
    "/lines": LONG_TTL,
    "/venues": LONG_TTL,
    "/teams/fbs": LONG_TTL,
    "/player/portal": LONG_TTL,
    "/player/returning": LONG_TTL,
}


def current_season(today: Date | None = None) -> int:
    """The season CFBD is currently filling in.

    A season is labelled by the year it kicks off in, so a January bowl still
    belongs to the previous year's season.
    """
    today = today or Date.today()
    return today.year if today.month >= 3 else today.year - 1


def default_cache_dir() -> Path | None:
    """``<data dir>/cache/cfbd``, or ``None`` when caching is switched off."""
    if os.getenv("CFBE_CFBD_CACHE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    root = os.getenv("CFBE_DATA_DIR") or str(Path.home() / ".cfb_engine")
    return Path(root) / "cache" / "cfbd"


def _read_cache(path: Path | None) -> tuple[object | None, float | None]:
    """The cached payload and its age in seconds, or ``(None, None)``."""
    if path is None or not path.exists():
        return None, None
    try:
        return json.loads(path.read_text()), time.time() - path.stat().st_mtime
    except (OSError, ValueError):
        return None, None


def _write_cache(path: Path | None, payload: object) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except (OSError, ValueError) as exc:
        log.warning("could not cache CFBD response (%s)", exc)


def _env_ttl() -> int:
    raw = os.getenv("CFBE_CFBD_CACHE_TTL", "").strip()
    try:
        return int(raw) if raw else DEFAULT_TTL
    except ValueError:
        return DEFAULT_TTL


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
    def __init__(
        self,
        api_key: str | None,
        timeout: int = 25,
        *,
        cache_dir: Path | None | str = "",
        cache_ttl: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        # "" means "whatever the environment says"; None explicitly disables the
        # disk cache, which is how the tests keep their calls observable.
        self.cache_dir = default_cache_dir() if cache_dir == "" else cache_dir
        self.cache_ttl = _env_ttl() if cache_ttl is None else cache_ttl
        # A full-season /games pull is reused across results + schedule filters.
        self._games_cache: dict[int, list[dict[str, object]]] = {}
        # Billable calls this process made, so a run's spend is in the log.
        self.calls = 0

    def available(self) -> bool:
        return bool(self.api_key)

    def _ttl(self, path: str, params: dict[str, str | int]) -> int:
        """Freshness window for one request.

        A finished season is immutable, so anything asked about a past year is
        cached for the long window whatever the endpoint's usual policy is --
        that is what stops a backtest from re-buying the same history.
        """
        year = params.get("year")
        if isinstance(year, (int, str)):
            try:
                if int(year) < current_season():
                    return LONG_TTL
            except ValueError:
                pass
        return ENDPOINT_TTL.get(path, self.cache_ttl)

    def _cache_path(self, path: str, params: dict[str, str | int]) -> Path | None:
        if self.cache_dir is None:
            return None
        stamp = json.dumps({"path": path, **{k: str(v) for k, v in params.items()}}, sort_keys=True)
        slug = path.strip("/").replace("/", "-") or "root"
        digest = hashlib.sha256(stamp.encode()).hexdigest()[:16]
        return Path(self.cache_dir) / f"{slug}-{digest}.json"

    def _get(self, path: str, **params: str | int) -> object:
        cache = self._cache_path(path, params)
        cached, age = _read_cache(cache)
        if cached is not None and age is not None and age < self._ttl(path, params):
            return cached

        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        self.calls += 1
        log.debug("CFBD call %d: %s", self.calls, path)
        try:
            resp = http.get(
                f"{BASE}{path}", params=params, headers=headers, timeout=self.timeout
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("CFBD request failed (%s): %s", path, exc)
            if cached is not None:
                log.warning("serving %s from cache %.1fh old", path, (age or 0) / 3600)
            return cached
        _write_cache(cache, payload)
        return payload

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

    def fetch_lines(self, season: int) -> list[dict[str, object]]:
        """Every game's closing numbers by provider, with the final score.

        Raw payload rather than a typed book: the only consumer is the line-shop
        distribution fit, which needs each provider's spread and total so it can
        take a median over real sportsbooks and drop the projection feeds CFBD
        mixes in alongside them.
        """
        if not self.available():
            return []
        rows: list[dict[str, object]] = []
        for season_type in ("regular", "postseason"):
            data = self._get("/lines", year=season, seasonType=season_type)
            if isinstance(data, list):
                rows.extend(r for r in data if isinstance(r, dict))
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

    def fetch_player_production(self, season: int) -> ProductionBook:
        """Per-player season PPA, garbage time excluded (the transfer join's key)."""
        if not self.available():
            return parse_production([])
        data = self._get("/ppa/players/season", year=season, excludeGarbageTime="true")
        if not isinstance(data, list):
            return parse_production([])
        return parse_production([row for row in data if isinstance(row, dict)])

    def fetch_roster_book(self, season: int) -> RosterBook | None:
        """Production kept *and* bought, for the ratings-only margin.

        Two extra calls, so callers build this lazily: it is only read when a game
        has no consensus spread to price from (:mod:`cfb_engine.data.roster`).
        """
        retained = self.fetch_returning_production(season)
        if not retained:
            return None
        prior = self.fetch_player_production(season - 1)
        data = self._get("/player/portal", year=season)
        entries = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        bought = build_incoming_shares(entries, prior, season)
        log.info(
            "roster book: %d teams returning, %d bought production", len(retained), len(bought)
        )
        return RosterBook(retained=retained, bought=bought)

    def fetch_starters(self, season: int, through_week: int) -> StarterBook:
        """Primary passer per team from box scores in weeks before ``through_week``.

        Read against the injury feed by :mod:`cfb_engine.data.injuries`: a
        designation only matters in proportion to the usage it removes.
        """
        if not self.available() or through_week <= 1:
            return {}
        rows: list[PasserGame] = []
        for week in range(1, min(through_week, 20)):
            data = self._get("/games/players", year=season, week=week)
            if not isinstance(data, list):
                continue
            rows.extend(_passer_games(data, week))
        if not rows:
            return {}
        return build_starter_book(rows, through_week=through_week)

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


def _passer_games(games: list[object], week: int) -> list[PasserGame]:
    """Pull ``C/ATT`` out of one week of ``/games/players`` box scores."""
    rows: list[PasserGame] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        teams = game.get("teams")
        if not isinstance(teams, list):
            continue
        for team in teams:
            if not isinstance(team, dict):
                continue
            name = team.get("team")
            categories = team.get("categories")
            if not name or not isinstance(categories, list):
                continue
            for category in categories:
                if not isinstance(category, dict) or category.get("name") != "passing":
                    continue
                types = category.get("types")
                if not isinstance(types, list):
                    continue
                for stat_type in types:
                    if not isinstance(stat_type, dict) or stat_type.get("name") != "C/ATT":
                        continue
                    athletes = stat_type.get("athletes")
                    if not isinstance(athletes, list):
                        continue
                    for athlete in athletes:
                        if not isinstance(athlete, dict):
                            continue
                        attempts = _attempts(athlete.get("stat"))
                        if attempts is None:
                            continue
                        rows.append(
                            PasserGame(
                                week=week,
                                team=school_key(str(name)),
                                player_id=str(athlete.get("id", "")),
                                name=str(athlete.get("name", "")),
                                attempts=attempts,
                            )
                        )
    return rows


def _attempts(stat: object) -> int | None:
    """Attempts out of a ``"18/22"`` completions-slash-attempts string."""
    if not isinstance(stat, str) or "/" not in stat:
        return None
    try:
        return int(stat.split("/")[1])
    except (IndexError, ValueError):
        return None


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
