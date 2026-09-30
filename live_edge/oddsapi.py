"""The Odds API in-play board for NFL / NCAAF: every book's live ML, spread and total.

One ``/odds`` call per sport per tick (3 credits) returns pregame and in-play
events alike; the caller decides which are live from the ESPN clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from live_edge.espn import norm_team

log = logging.getLogger(__name__)

SPORT_KEY = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}
BASE = "https://api.the-odds-api.com/v4/sports"
MARKETS = "h2h,spreads,totals"


@dataclass(frozen=True)
class Quote:
    book: str
    market: str  # "ml" | "spread" | "total"
    side: str  # "home" | "away" | "over" | "under"
    line: float | None
    price: float  # American
    updated: str  # ISO-8601 from the book


@dataclass(frozen=True)
class EventQuotes:
    event_id: str
    home: str
    away: str
    commence: str
    quotes: tuple[Quote, ...]

    def pairs(self) -> list[tuple[Quote, Quote]]:
        """Two-way pairs (same book, market, line) so each can be devigged."""
        by_key: dict[tuple[str, str, float | None], dict[str, Quote]] = {}
        for q in self.quotes:
            key_line = (
                q.line if q.market == "total" else (abs(q.line) if q.line is not None else None)
            )
            by_key.setdefault((q.book, q.market, key_line), {})[q.side] = q
        out = []
        for sides in by_key.values():
            a = sides.get("home") or sides.get("over")
            b = sides.get("away") or sides.get("under")
            if a and b:
                out.append((a, b))
        return out


def parse_events(payload: list[dict]) -> list[EventQuotes]:
    events = []
    for ev in payload:
        home, away = str(ev.get("home_team", "")), str(ev.get("away_team", ""))
        hn, an = norm_team(home), norm_team(away)
        quotes: list[Quote] = []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = {"h2h": "ml", "spreads": "spread", "totals": "total"}.get(
                    mk.get("key", "")
                )
                if market is None:
                    continue
                for oc in mk.get("outcomes", []):
                    name = norm_team(str(oc.get("name", "")))
                    if market == "total":
                        side = "over" if name == "over" else "under" if name == "under" else None
                    else:
                        side = "home" if name == hn else "away" if name == an else None
                    if side is None or oc.get("price") is None:
                        continue
                    quotes.append(
                        Quote(
                            book=str(bk.get("key")),
                            market=market,
                            side=side,
                            line=float(oc["point"]) if oc.get("point") is not None else None,
                            price=float(oc["price"]),
                            updated=str(mk.get("last_update") or bk.get("last_update") or ""),
                        )
                    )
        events.append(
            EventQuotes(
                str(ev.get("id")), home, away, str(ev.get("commence_time", "")), tuple(quotes)
            )
        )
    return events


class LiveOddsClient:
    def __init__(self, api_key: str | None, *, regions: str = "us", timeout: float = 20.0) -> None:
        self.api_key = api_key
        self.regions = regions
        self.timeout = timeout
        self.credits_remaining: int | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, sport: str) -> list[EventQuotes]:
        if not self.available():
            return []
        url = f"{BASE}/{SPORT_KEY[sport]}/odds"
        params = {
            "apiKey": self.api_key or "",
            "regions": self.regions,
            "markets": MARKETS,
            "oddsFormat": "american",
        }
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            rem = resp.headers.get("x-requests-remaining")
            self.credits_remaining = int(rem) if rem and rem.isdigit() else None
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Odds API live board failed (%s): %s", sport, exc)
            return []
        if not isinstance(payload, list):
            return []
        return parse_events(payload)


def age_seconds(updated: str, now: datetime | None = None) -> float | None:
    if not updated:
        return None
    try:
        ts = datetime.fromisoformat(updated.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ((now or datetime.now(timezone.utc)) - ts).total_seconds()
