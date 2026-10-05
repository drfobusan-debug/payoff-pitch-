"""The Odds API client for ``basketball_nba``: live board and historical snapshots.

Live: the bulk ``/odds`` call returns the featured markets (ML / spread / total)
for every posted game, 3 credits; first halves and props are per event at about
a credit per market that returns data.

Historical (paid plans only): ``/historical/.../events`` costs 1 credit and
lists the events posted at a timestamp; ``/historical/.../events/{id}/odds``
returns the snapshot at or before a timestamp (5-minute snapshots since
September 2022) at 10 credits per market returned per region. One 2025-12-11
event at T-5 returned all eleven market keys for 110 credits.

Everything comes back as archive rows. Nothing here forms a probability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from mlb_engine.data import http
from nba_engine.data import capture, teamnames
from nba_engine.data.capture import EVENT_MARKETS, GAME_MARKETS, MARKET_MAP, QuoteRow
from nba_engine.schemas import Game, Slate

log = logging.getLogger(__name__)

SPORT_KEY = "basketball_nba"
API = "https://api.the-odds-api.com/v4"
BASE = f"{API}/sports/{SPORT_KEY}"
HIST_BASE = f"{API}/historical/sports/{SPORT_KEY}"

# A 10:30pm ET tip is a 02:30 UTC start; the slate date is the Eastern one.
SLATE_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Snapshot:
    """One historical response: the snapshot time the API served and its payload."""

    timestamp: str
    data: object


class OddsAPIClient:
    def __init__(
        self,
        api_key: str | None,
        timeout: int = 30,
        *,
        regions: str = "us",
        cache_dir: Path | None = None,
        cache_ttl: int = 900,
        sport_key: str = SPORT_KEY,
    ) -> None:
        self.api_key = api_key
        self.sport_key = sport_key
        self.base = f"{API}/sports/{sport_key}"
        self.hist_base = f"{API}/historical/sports/{sport_key}"
        self.timeout = timeout
        self.regions = regions
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.credits_remaining: int | None = None
        self.credits_last: int | None = None
        self.last_status: int | None = None
        self._session = http.session(user_agent="nba-engine/0.1", timeout=timeout)

    def available(self) -> bool:
        return bool(self.api_key)

    # -- live -------------------------------------------------------------
    def fetch_events(self, *, slate_date: Date, horizon_hours: int = 30) -> Slate:
        """Games starting between the slate's Eastern midnight and the horizon (free)."""
        slate = Slate(slate_date=slate_date)
        if not self.available():
            return slate
        start, end = _window(slate_date, horizon_hours)
        data = self._get_json(
            f"{self.base}/events", commenceTimeFrom=_iso(start), commenceTimeTo=_iso(end)
        )
        slate.games = games_from(data, slate_date) if isinstance(data, list) else []
        return slate

    def fetch_board(
        self, *, slate_date: Date, horizon_hours: int = 30, captured_at: str | None = None
    ) -> tuple[Slate, list[QuoteRow]]:
        """Featured markets for every game in the window, as archive rows."""
        slate = Slate(slate_date=slate_date)
        if not self.available():
            return slate, []
        start, end = _window(slate_date, horizon_hours)
        data = self._get_json(
            f"{self.base}/odds",
            markets=",".join(GAME_MARKETS),
            commenceTimeFrom=_iso(start),
            commenceTimeTo=_iso(end),
        )
        if not isinstance(data, list):
            return slate, []
        taken = captured_at or capture.now_utc()
        rows: list[QuoteRow] = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            game = to_game(raw, slate_date)
            if game is None:
                continue
            slate.games.append(game)
            rows.extend(event_rows(raw, game, taken))
        slate.games.sort(key=lambda g: g.start_utc)
        return slate, rows

    def fetch_event_markets(
        self,
        games: list[Game],
        *,
        markets: tuple[str, ...] = EVENT_MARKETS,
        captured_at: str | None = None,
        max_events: int = 16,
    ) -> list[QuoteRow]:
        """First-half and prop markets, per event, as archive rows."""
        if not self.available():
            return []
        taken = captured_at or capture.now_utc()
        out: list[QuoteRow] = []
        for game in games[:max_events]:
            if not game.event_id:
                continue
            data = self._get_json(
                f"{self.base}/events/{game.event_id}/odds", markets=",".join(markets)
            )
            if isinstance(data, dict):
                out.extend(event_rows(data, game, taken))
        return out

    # -- historical -------------------------------------------------------
    def historical_events(self, at: datetime) -> Snapshot | None:
        """Events posted at ``at`` (1 credit)."""
        return self._snapshot(f"{self.hist_base}/events", date=_iso(at))

    def historical_event_odds(
        self, event_id: str, at: datetime, markets: tuple[str, ...]
    ) -> Snapshot | None:
        """One event's markets at the last snapshot at or before ``at``."""
        return self._snapshot(
            f"{self.hist_base}/events/{event_id}/odds", date=_iso(at), markets=",".join(markets)
        )

    def historical_board(
        self, at: datetime, markets: tuple[str, ...] = GAME_MARKETS
    ) -> Snapshot | None:
        """Featured markets for every posted game at ``at`` (10 credits per market)."""
        return self._snapshot(f"{self.hist_base}/odds", date=_iso(at), markets=",".join(markets))

    def _snapshot(self, url: str, **params: str) -> Snapshot | None:
        payload = self._get_json(url, use_cache=False, **params)
        if not isinstance(payload, dict) or "data" not in payload:
            return None
        return Snapshot(timestamp=str(payload.get("timestamp", "")), data=payload["data"])

    # -- transport --------------------------------------------------------
    def _cache_path(self, url: str, params: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        stamp = json.dumps({"url": url, **params}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(stamp.encode()).hexdigest()[:20]}.json"

    def _get_json(self, url: str, *, use_cache: bool = True, **params: str) -> object:
        query = {"regions": self.regions, "oddsFormat": "american", **params}
        cache = self._cache_path(url, query) if use_cache else None
        if cache is not None and cache.exists():
            if time.time() - cache.stat().st_mtime < self.cache_ttl:
                try:
                    return json.loads(cache.read_text())
                except ValueError:
                    pass
        try:
            resp = self._session.get(url, params={"apiKey": self.api_key or "", **query})
            self.last_status = resp.status_code
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                self.credits_remaining = int(float(remaining))
            last = resp.headers.get("x-requests-last")
            self.credits_last = int(float(last)) if last is not None else None
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
                redact(str(exc), self.api_key),
            )
            return None


# -- parsing ----------------------------------------------------------------
def _window(slate_date: Date, horizon_hours: int) -> tuple[datetime, datetime]:
    start = datetime.combine(slate_date, datetime.min.time(), tzinfo=SLATE_TZ)
    return start.astimezone(timezone.utc), (start + timedelta(hours=horizon_hours)).astimezone(
        timezone.utc
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def slate_date_of(commence_utc: str) -> Date | None:
    moment = parse_utc(commence_utc)
    return None if moment is None else moment.astimezone(SLATE_TZ).date()


def to_game(raw: dict, fallback: Date) -> Game | None:
    home_raw = str(raw.get("home_team", ""))
    away_raw = str(raw.get("away_team", ""))
    event_id = raw.get("id")
    home = teamnames.code_for(home_raw)
    away = teamnames.code_for(away_raw)
    if not event_id or home is None or away is None:
        if event_id and (home is None or away is None):
            log.warning("unmapped Odds API team name: %r vs %r", away_raw, home_raw)
        return None
    start = str(raw.get("commence_time", ""))
    return Game(
        game_date=slate_date_of(start) or fallback,
        home=home,
        away=away,
        start_utc=start,
        event_id=str(event_id),
    )


def games_from(events: list, slate_date: Date | None = None) -> list[Game]:
    """Mapped games, optionally only those tipping on ``slate_date`` (ET), by tip time."""
    out: list[Game] = []
    for raw in events:
        if not isinstance(raw, dict):
            continue
        game = to_game(raw, slate_date or Date.min)
        if game is None or (slate_date is not None and game.game_date != slate_date):
            continue
        out.append(game)
    out.sort(key=lambda g: (g.start_utc, g.matchup))
    return out


def event_rows(payload: dict, game: Game, captured_at: str) -> list[QuoteRow]:
    """Flatten one event payload (bulk, per-event or historical) into archive rows.

    Pairing is per (entity, line) within a market so an over is never paired
    with another player's under; a spread pairs on the line seen from the home
    side (BOS -3.5 with NYK +3.5). A side whose partner is missing stays unpaired.
    """
    out: list[QuoteRow] = []
    for bookmaker in payload.get("bookmakers", []):
        if not isinstance(bookmaker, dict):
            continue
        book = str(bookmaker.get("key", ""))
        for market in bookmaker.get("markets", []):
            if not isinstance(market, dict):
                continue
            provider_key = str(market.get("key", ""))
            mapped = MARKET_MAP.get(provider_key)
            if mapped is None:
                continue
            engine_key = mapped[0]
            groups: dict[tuple[str, float | None], dict[str, float]] = {}
            start = len(out)
            for oc in market.get("outcomes", []):
                if not isinstance(oc, dict) or not isinstance(oc.get("price"), (int, float)):
                    continue
                side, entity = _side_entity(oc, game, provider_key)
                if side is None:
                    continue
                point = None if oc.get("point") is None else float(oc["point"])
                groups.setdefault((entity, _pair_line(side, point, engine_key, game)), {})[side] = (
                    float(oc["price"])
                )
                out.append(
                    QuoteRow(
                        captured_at=captured_at,
                        game_date=game.game_date.isoformat(),
                        matchup=game.matchup,
                        event_id=game.event_id,
                        market=engine_key,
                        side=side,
                        entity=entity,
                        line=point,
                        book=book,
                        american=float(oc["price"]),
                        opposite_american=None,
                    )
                )
            for i in range(start, len(out)):
                row = out[i]
                sides = groups.get(
                    (row.entity, _pair_line(row.side, row.line, engine_key, game)), {}
                )
                partner = _opposite(row.side, game)
                if partner is not None and partner in sides and len(sides) == 2:
                    out[i] = QuoteRow(**{**row.__dict__, "opposite_american": sides[partner]})
    return out


def _pair_line(side: str, point: float | None, engine_key: str, game: Game) -> float | None:
    if point is None:
        return None
    if engine_key.endswith("_ats"):
        return point if side == game.home else -point
    return point


def _side_entity(oc: dict, game: Game, provider_key: str) -> tuple[str | None, str]:
    low = _norm(str(oc.get("name", "")))
    if low.startswith("over") or low.startswith("under"):
        side = "over" if low.startswith("over") else "under"
        entity = (
            str(oc.get("description", "")).strip() if provider_key.startswith("player_") else ""
        )
        if provider_key.startswith("player_") and not entity:
            return None, ""
        return side, entity
    if provider_key.startswith("player_"):
        return None, ""
    code = teamnames.code_for(str(oc.get("name", "")))
    if code in (game.home, game.away):
        return code, ""
    return None, ""


def _opposite(side: str, game: Game) -> str | None:
    pairs = {"over": "under", "under": "over", game.home: game.away, game.away: game.home}
    return pairs.get(side)


def redact(message: str, api_key: str | None) -> str:
    if api_key:
        message = message.replace(api_key, "***")
    return re.sub(r"apiKey=[^&\s]*", "apiKey=***", message)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


__all__ = [
    "BASE",
    "HIST_BASE",
    "SLATE_TZ",
    "SPORT_KEY",
    "OddsAPIClient",
    "Snapshot",
    "event_rows",
    "games_from",
    "parse_utc",
    "redact",
    "slate_date_of",
    "to_game",
]
