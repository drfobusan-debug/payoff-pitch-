"""The Odds API client for ``icehockey_nhl``.

Two endpoints, two costs. The bulk ``/odds`` call returns the featured markets
(``h2h``, ``spreads``, ``totals``) for every posted game in one request; period
markets, team totals and player props are only available per event, at roughly
one credit per market that returns data. All 31 market keys in
``capture.MARKET_MAP`` were confirmed valid against a live event on 2026-09-28
(one event, all keys: 25 credits).

Everything comes back as archive rows. Nothing here forms a probability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import date as Date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from mlb_engine.data import http
from nhl_engine.data import capture, teamnames
from nhl_engine.data.capture import EVENT_MARKETS, GAME_MARKETS, MARKET_MAP, QuoteRow
from nhl_engine.schemas import Game, Slate

log = logging.getLogger(__name__)

SPORT_KEY = "icehockey_nhl"
BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}"

# A 10pm ET puck drop is a 02:00 UTC start; the slate date is the Eastern one.
SLATE_TZ = ZoneInfo("America/New_York")


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
        self.credits_last: int | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch_events(self, *, slate_date: Date, horizon_hours: int = 30) -> Slate:
        """Games starting between the slate's Eastern midnight and the horizon.

        The ``/events`` endpoint is free of charge; use it to size a capture
        before spending on per-event markets.
        """
        slate = Slate(slate_date=slate_date)
        if not self.available():
            return slate
        start, end = _window(slate_date, horizon_hours)
        data = self._get_json(
            f"{BASE}/events", commenceTimeFrom=_iso(start), commenceTimeTo=_iso(end)
        )
        if not isinstance(data, list):
            return slate
        for raw in data:
            game = _to_game(raw, slate_date) if isinstance(raw, dict) else None
            if game is not None:
                slate.games.append(game)
        slate.games.sort(key=lambda g: g.start_utc)
        return slate

    def fetch_board(
        self,
        *,
        slate_date: Date,
        horizon_hours: int = 30,
        captured_at: str | None = None,
    ) -> tuple[Slate, list[QuoteRow]]:
        """Featured markets for every game in the window, as archive rows."""
        slate = Slate(slate_date=slate_date)
        if not self.available():
            return slate, []
        start, end = _window(slate_date, horizon_hours)
        data = self._get_json(
            f"{BASE}/odds",
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
            game = _to_game(raw, slate_date)
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
        max_events: int = 20,
    ) -> list[QuoteRow]:
        """Period, team-total and prop markets, per event, as archive rows.

        Each event-market costs about a credit per region, so the slate is capped
        and the caller decides how often to run. An outcome whose partner is
        missing is kept unpaired; its other side is never invented.
        """
        if not self.available():
            return []
        taken = captured_at or capture.now_utc()
        out: list[QuoteRow] = []
        for game in games[:max_events]:
            if not game.event_id:
                continue
            data = self._get_json(f"{BASE}/events/{game.event_id}/odds", markets=",".join(markets))
            if not isinstance(data, dict):
                continue
            out.extend(event_rows(data, game, taken))
        return out

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
            resp = http.get(
                url, params={"apiKey": self.api_key or "", **query}, timeout=self.timeout
            )
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                self.credits_remaining = int(float(remaining))
            last = resp.headers.get("x-requests-last")
            if last is not None:
                self.credits_last = int(float(last))
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


# -- parsing ----------------------------------------------------------------
def _window(slate_date: Date, horizon_hours: int) -> tuple[datetime, datetime]:
    start = datetime.combine(slate_date, datetime.min.time(), tzinfo=SLATE_TZ)
    return start.astimezone(timezone.utc), (start + timedelta(hours=horizon_hours)).astimezone(
        timezone.utc
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _start_date(commence_utc: str, fallback: Date) -> Date:
    try:
        moment = datetime.fromisoformat(commence_utc.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    return moment.astimezone(SLATE_TZ).date()


def _to_game(raw: dict, slate_date: Date) -> Game | None:
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
        game_date=_start_date(start, slate_date),
        home=home,
        away=away,
        start_utc=start,
        event_id=str(event_id),
    )


def event_rows(payload: dict, game: Game, captured_at: str) -> list[QuoteRow]:
    """Flatten one event payload (bulk or per-event) into archive rows.

    Pairing is per (entity, line) within a market: an alternate-spread market
    holds a ladder of lines per team, a prop market dozens of players. Pairing
    an over with the wrong player's under would fabricate a hold, so partners
    must match on entity and line, and 3-way markets get no partner at all.
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
            engine_key, _ = mapped
            outcomes = [
                oc
                for oc in market.get("outcomes", [])
                if isinstance(oc, dict) and oc.get("price") is not None
            ]
            three_way = engine_key.endswith("ml3")
            groups: dict[tuple[str, float | None], dict[str, float]] = {}
            start = len(out)
            for oc in outcomes:
                side, entity = _side_entity(oc, game, provider_key)
                if side is None:
                    continue
                point = None if oc.get("point") is None else float(oc["point"])
                group_key = (entity, _pair_line(side, point, engine_key, game))
                groups.setdefault(group_key, {})[side] = float(oc["price"])
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
            if three_way:
                continue
            # Second pass attaches the partner now that every side is known.
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
    """The line that identifies a two-way pair.

    Totals and props pair on the same number. A puck line pairs +1.5 with -1.5,
    and an alternate ladder holds both MTL -2.5 and MTL +2.5, so the pair is
    identified by the line seen from the home side: MTL -2.5 and TOR +2.5 are
    one pair, MTL +2.5 and TOR -2.5 another.
    """
    if point is None:
        return None
    if "_pl" in engine_key:
        return point if side == game.home else -point
    return point


def _side_entity(oc: dict, game: Game, provider_key: str) -> tuple[str | None, str]:
    name = str(oc.get("name", ""))
    desc = str(oc.get("description", ""))
    low = _norm(name)
    if provider_key.startswith("player_"):
        if low.startswith("over"):
            return "over", desc
        if low.startswith("under"):
            return "under", desc
        if low in ("yes", "no"):
            return low, desc
        # Some books name the player in ``name`` for scorer markets.
        return "yes", desc or name
    if low.startswith("over") or low.startswith("under"):
        entity = (teamnames.code_for(desc) or "") if desc else ""
        return ("over" if low.startswith("over") else "under"), entity
    if low == "draw":
        return "draw", ""
    code = teamnames.code_for(name)
    if code in (game.home, game.away):
        return code, ""
    return None, ""


def _opposite(side: str, game: Game) -> str | None:
    if side == "over":
        return "under"
    if side == "under":
        return "over"
    if side == "yes":
        return "no"
    if side == "no":
        return "yes"
    if side == game.home:
        return game.away
    if side == game.away:
        return game.home
    return None


def _redact(message: str, api_key: str | None) -> str:
    if api_key:
        message = message.replace(api_key, "***")
    return re.sub(r"\?[^\s]*", "?<redacted>", message)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


__all__ = ["BASE", "SPORT_KEY", "OddsAPIClient", "event_rows"]
