"""The Odds API (https://the-odds-api.com) NCAAF client.

Pulls multi-book American prices for the three full-game markets the engine
bets -- moneyline (``h2h``), spread (``spreads``), and total (``totals``) -- in
a single bulk request, and maps them onto the ``(matchup, market, selection)``
quote keys in :mod:`cfb_engine.market.keys`.

The same board doubles as the schedule source: every event carries the two
teams and a kickoff time, so :meth:`fetch_board` returns both the
:class:`~cfb_engine.schemas.Slate` and the priced quotes. With no API key the
client is inert and the caller must supply a slate another way.
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

from cfb_engine.market.board import GameOdds
from cfb_engine.market.ev import MarketQuote
from cfb_engine.schemas import Game, Slate, TeamGameInfo

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf"
_GAME_MARKETS = ["h2h", "spreads", "totals"]
# College kickoffs are quoted in US Eastern; a "Saturday slate" is the set of
# games that kick between Eastern midnight and the next Eastern midnight.
_SLATE_TZ = ZoneInfo("America/New_York")

Board = dict[str, GameOdds]  # keyed by matchup ("AWAY @ HOME")


class _Event:
    __slots__ = ("event_id", "matchup", "home_ab", "away_ab", "home_name", "away_name")

    def __init__(
        self, event_id: str, home_ab: str, away_ab: str, home_name: str, away_name: str
    ) -> None:
        self.event_id = event_id
        self.matchup = f"{away_ab} @ {home_ab}"
        self.home_ab = home_ab
        self.away_ab = away_ab
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
        cache_ttl: int = 1800,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.regions = regions
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.credits_remaining: int | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_board(self, slate_date: Date) -> tuple[Slate, Board]:
        """Return the day's slate and structured multi-book odds per game.

        One bulk ``/odds`` request covers every game and all three markets.
        Returns an empty slate/board if there is no key or the request fails.
        """
        empty: tuple[Slate, Board] = Slate(slate_date=slate_date), {}
        if not self.available():
            return empty

        start = datetime.combine(slate_date, Time(0, 0), tzinfo=_SLATE_TZ).astimezone(timezone.utc)
        end = start + timedelta(days=1)
        data = self._get_json(
            f"{BASE}/odds",
            markets=",".join(_GAME_MARKETS),
            commenceTimeFrom=_iso(start),
            commenceTimeTo=_iso(end),
        )
        if not isinstance(data, list):
            return empty

        games: list[Game] = []
        board: Board = {}
        for raw in data:
            ev = self._to_event(raw)
            if ev is None:
                continue
            games.append(
                Game(
                    game_id=ev.event_id,
                    game_date=slate_date,
                    commence_time_utc=raw.get("commence_time"),
                    home=TeamGameInfo(name=raw["home_team"], abbrev=ev.home_ab, is_home=True),
                    away=TeamGameInfo(name=raw["away_team"], abbrev=ev.away_ab, is_home=False),
                )
            )
            board[ev.matchup] = self._parse_game(raw, ev)

        games.sort(key=lambda g: g.commence_time_utc or "")
        return Slate(slate_date=slate_date, games=games), board

    # -- parsing ----------------------------------------------------------
    def _parse_game(self, raw: dict, ev: _Event) -> GameOdds:
        odds = GameOdds(matchup=ev.matchup)
        for bm in raw.get("bookmakers", []):
            book = bm.get("key", "")
            for mkt in bm.get("markets", []):
                mk = mkt.get("key", "")
                opposite = _opposite_prices(mkt.get("outcomes", []))
                for oc in mkt.get("outcomes", []):
                    price = oc.get("price")
                    if price is None:
                        continue
                    quote = MarketQuote(
                        book=book,
                        american=float(price),
                        opposite_american=opposite.get(id(oc)),
                    )
                    self._route(mk, oc, ev, odds, quote)
        return odds

    def _route(self, mk: str, oc: dict, ev: _Event, odds: GameOdds, quote: MarketQuote) -> None:
        name = _norm(str(oc.get("name", "")))
        ab = ev.home_ab if name == ev.home_name else ev.away_ab if name == ev.away_name else None
        point = oc.get("point")
        if mk == "h2h" and ab is not None:
            odds.add_ml(ab, quote)
        elif mk == "spreads" and ab is not None and point is not None:
            # Store every spread on the home-point axis so the two sides pair up.
            home_point = float(point) if ab == ev.home_ab else -float(point)
            odds.add_spread(home_point, ab, quote)
        elif mk == "totals" and point is not None:
            over = str(oc.get("name", "")).lower().startswith("over")
            odds.add_total(float(point), over, quote)

    # -- helpers ----------------------------------------------------------
    def _to_event(self, raw: dict) -> _Event | None:
        home_name = _norm(str(raw.get("home_team", "")))
        away_name = _norm(str(raw.get("away_team", "")))
        eid = raw.get("id")
        if not home_name or not away_name or not eid:
            return None
        from cfb_engine.data.teamnames import short_code

        return _Event(
            str(eid),
            short_code(str(raw.get("home_team", ""))),
            short_code(str(raw.get("away_team", ""))),
            home_name,
            away_name,
        )

    def _cache_path(self, url: str, params: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        stamp = json.dumps({"url": url, **params}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(stamp.encode()).hexdigest()[:20]}.json"

    def _get_json(self, url: str, **params: str) -> object:
        q = {"regions": self.regions, "oddsFormat": "american", **params}
        cache = self._cache_path(url, q)
        if cache is not None and cache.exists():
            if time.time() - cache.stat().st_mtime < self.cache_ttl:
                try:
                    return json.loads(cache.read_text())
                except ValueError:
                    pass
        try:
            resp = requests.get(
                url, params={"apiKey": self.api_key or "", **q}, timeout=self.timeout
            )
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
            # The request URL (with the API key) is embedded in the exception; never log it raw.
            log.warning(
                "Odds API request failed (%s): %s",
                url.rsplit("/", 1)[-1],
                _redact(str(exc), self.api_key),
            )
            return None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _opposite_prices(outcomes: list[dict]) -> dict[int, float]:
    """Map each outcome to the opposing side's price in the same two-way market."""
    priced = [oc for oc in outcomes if oc.get("price") is not None]
    if len(priced) == 2:
        a, b = priced
        return {id(a): float(b["price"]), id(b): float(a["price"])}
    return {}


def _redact(message: str, api_key: str | None) -> str:
    if api_key:
        message = message.replace(api_key, "***")
    return re.sub(r"\?[^\s]*", "?<redacted>", message)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
