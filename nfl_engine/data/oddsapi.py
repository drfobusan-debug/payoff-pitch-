"""The Odds API (https://the-odds-api.com) NFL client.

One bulk ``/odds`` request returns every game on the board with multi-book
American prices for the three markets the engine prices from its own score
distribution -- moneyline, spread and total -- and the same payload doubles as
the schedule, since each event carries both teams and a kickoff time.

Two NFL-specific choices. **Every quoted line is kept, not just the main one**:
books hang -2.5, -3 and -3.5 on the same game and the rung matters more here
than anywhere else, so :class:`~nfl_engine.market.board.GameOdds` stores the
whole ladder. And **the week is the unit, not the day** -- a slate spans Thursday
night to Monday night, so the fetch window is a date range and every event is
tagged with the season and week it belongs to.

With no API key the client is inert and returns an empty board rather than
raising, so a missing credential degrades to "no prices", not a dead slate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from datetime import time as Time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from mlb_engine.data import http
from nfl_engine.data import teamnames
from nfl_engine.market.board import GameOdds, MarketQuote
from nfl_engine.schemas import Game, Slate, TeamGameInfo

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
GAME_MARKETS = ("h2h", "spreads", "totals")
# Kickoffs are quoted in US Eastern, and an NFL "slate" runs from the Thursday
# night game to the Monday night one.
SLATE_TZ = ZoneInfo("America/New_York")

Board = dict[str, GameOdds]  # keyed by matchup ("AWAY @ HOME")


class _Event:
    __slots__ = ("event_id", "matchup", "home_code", "away_code", "home_name", "away_name")

    def __init__(
        self, event_id: str, home_code: str, away_code: str, home_name: str, away_name: str
    ) -> None:
        self.event_id = event_id
        self.matchup = f"{away_code} @ {home_code}"
        self.home_code = home_code
        self.away_code = away_code
        self.home_name = home_name
        self.away_name = away_name


class OddsAPIClient:
    def __init__(
        self,
        api_key: str | None,
        timeout: int = 25,
        *,
        regions: str = "us",
        cache_dir: Path | None = None,
        cache_ttl: int = 900,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.regions = regions
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.credits_remaining: int | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_board(
        self,
        *,
        season: int,
        week: int,
        first_day: Date,
        days: int = 5,
    ) -> tuple[Slate, Board]:
        """The week's slate and its multi-book prices.

        ``first_day`` is the Thursday; ``days`` spans forward to Monday night.
        Returns an empty slate and board with no key or on a failed request.
        """
        empty: tuple[Slate, Board] = (
            Slate(season=season, week=week, slate_date=first_day),
            {},
        )
        if not self.available():
            return empty

        start = datetime.combine(first_day, Time(0, 0), tzinfo=SLATE_TZ).astimezone(timezone.utc)
        end = start + timedelta(days=days)
        data = self._get_json(
            f"{BASE}/odds",
            markets=",".join(GAME_MARKETS),
            commenceTimeFrom=_iso(start),
            commenceTimeTo=_iso(end),
        )
        if not isinstance(data, list):
            return empty

        games: list[Game] = []
        board: Board = {}
        for raw in data:
            if not isinstance(raw, dict):
                continue
            event = self._to_event(raw)
            if event is None:
                continue
            games.append(
                Game(
                    game_id=event.event_id,
                    season=season,
                    week=week,
                    game_date=_kickoff_date(str(raw.get("commence_time", "")), first_day),
                    kickoff_utc=raw.get("commence_time"),
                    home=TeamGameInfo(
                        name=str(raw["home_team"]), abbrev=event.home_code, is_home=True
                    ),
                    away=TeamGameInfo(
                        name=str(raw["away_team"]), abbrev=event.away_code, is_home=False
                    ),
                )
            )
            board[event.matchup] = self._parse_game(raw, event)

        games.sort(key=lambda game: game.kickoff_utc or "")
        return Slate(season=season, week=week, slate_date=first_day, games=games), board

    # -- parsing ----------------------------------------------------------
    def _parse_game(self, raw: dict, event: _Event) -> GameOdds:
        odds = GameOdds(matchup=event.matchup)
        for bookmaker in raw.get("bookmakers", []):
            book = str(bookmaker.get("key", ""))
            for market in bookmaker.get("markets", []):
                key = str(market.get("key", ""))
                outcomes = market.get("outcomes", [])
                opposite = _opposite_prices(outcomes)
                for outcome in outcomes:
                    price = outcome.get("price")
                    if price is None:
                        continue
                    quote = MarketQuote(
                        book=book,
                        american=float(price),
                        opposite_american=opposite.get(id(outcome)),
                    )
                    self._route(key, outcome, event, odds, quote)
        return odds

    def _route(
        self, key: str, outcome: dict, event: _Event, odds: GameOdds, quote: MarketQuote
    ) -> None:
        name = _norm(str(outcome.get("name", "")))
        code = (
            event.home_code
            if name == event.home_name
            else event.away_code
            if name == event.away_name
            else None
        )
        point = outcome.get("point")
        if key == "h2h" and code is not None:
            odds.add_ml(code, quote)
        elif key == "spreads" and code is not None and point is not None:
            home_point = float(point) if code == event.home_code else -float(point)
            odds.add_spread(home_point, code, quote)
        elif key == "totals" and point is not None:
            odds.add_total(float(point), name.startswith("over"), quote)

    def _to_event(self, raw: dict) -> _Event | None:
        """Build an event, or ``None`` if either team cannot be identified.

        An unrecognized name is dropped rather than guessed: pricing the wrong
        team is a worse failure than skipping a game.
        """
        home_raw = str(raw.get("home_team", ""))
        away_raw = str(raw.get("away_team", ""))
        event_id = raw.get("id")
        home_code = teamnames.code_for(home_raw)
        away_code = teamnames.code_for(away_raw)
        if not event_id or home_code is None or away_code is None:
            if event_id and (home_code is None or away_code is None):
                log.warning("unmapped Odds API team name: %r vs %r", away_raw, home_raw)
            return None
        return _Event(str(event_id), home_code, away_code, _norm(home_raw), _norm(away_raw))

    # -- transport --------------------------------------------------------
    def _cache_path(self, url: str, params: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        stamp = json.dumps({"url": url, **params}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(stamp.encode()).hexdigest()[:20]}.json"

    def _get_json(self, url: str, **params: str) -> object:
        query = {"regions": self.regions, "oddsFormat": "american", **params}
        cache = self._cache_path(url, query)
        if cache is not None and cache.exists():
            if time.time() - cache.stat().st_mtime < self.cache_ttl:
                try:
                    return json.loads(cache.read_text())
                except ValueError:
                    pass
        try:
            resp = http.get(url, params={"apiKey": self.api_key or "", **query}, timeout=self.timeout)
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                self.credits_remaining = int(float(remaining))
            resp.raise_for_status()
            payload = resp.json()
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(payload))
            return payload
        except (requests.RequestException, ValueError) as exc:
            # The failing URL carries the API key; never log it unredacted.
            log.warning(
                "Odds API request failed (%s): %s",
                url.rsplit("/", 1)[-1],
                _redact(str(exc), self.api_key),
            )
            return None


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _kickoff_date(commence_utc: str, fallback: Date) -> Date:
    if not commence_utc:
        return fallback
    try:
        moment = datetime.fromisoformat(commence_utc.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    return moment.astimezone(SLATE_TZ).date()


def _opposite_prices(outcomes: list[dict]) -> dict[int, float]:
    """Each outcome mapped to the opposing price in the same two-way market."""
    priced = [oc for oc in outcomes if oc.get("price") is not None]
    if len(priced) == 2:
        first, second = priced
        return {id(first): float(second["price"]), id(second): float(first["price"])}
    return {}


def _redact(message: str, api_key: str | None) -> str:
    if api_key:
        message = message.replace(api_key, "***")
    return re.sub(r"\?[^\s]*", "?<redacted>", message)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
