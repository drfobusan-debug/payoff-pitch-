"""The multi-book board, normalised across sports and kept whole.

Both engines already parse The Odds API, but each collapses the payload into its
own pricing objects on the way in. The scanner needs the opposite: every quote
from every book at every rung, with the book's name attached, because the
product here is the disagreement between books rather than a consensus price.
So this is a thin reader over the same endpoint, shared by CFB and NFL.

With no API key it returns an empty board rather than raising -- a missing
credential is "nothing to shop", not a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mlb_engine.data import http

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4/sports"
SPORT_KEYS = {"cfb": "americanfootball_ncaaf", "nfl": "americanfootball_nfl"}
MARKETS = ("h2h", "spreads", "totals")
OVER, UNDER = "Over", "Under"


@dataclass(frozen=True)
class Quote:
    book: str
    american: int
    point: float | None = None


@dataclass
class Game:
    sport: str
    game_id: str
    home: str
    away: str
    commence: str
    # (market, side) -> quotes, where side is a team name or Over/Under
    quotes: dict[tuple[str, str], list[Quote]] = field(default_factory=dict)

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"

    @property
    def books(self) -> set[str]:
        return {q.book for quotes in self.quotes.values() for q in quotes}

    def sides(self, market: str) -> list[str]:
        return [side for (m, side) in self.quotes if m == market]

    def get(self, market: str, side: str) -> list[Quote]:
        return self.quotes.get((market, side), [])


def restrict(games: list[Game], books: tuple[str, ...]) -> list[Game]:
    """The same board as seen from a shopper who only holds ``books``.

    A number nobody can bet is not an edge, so the scan is run against the
    accounts that exist rather than the whole screen. Matching is
    case-insensitive on the book's display title.
    """
    wanted = {b.casefold() for b in books}
    out: list[Game] = []
    for game in games:
        kept = {
            key: [q for q in quotes if q.book.casefold() in wanted]
            for key, quotes in game.quotes.items()
        }
        kept = {key: quotes for key, quotes in kept.items() if quotes}
        if kept:
            out.append(
                Game(
                    sport=game.sport,
                    game_id=game.game_id,
                    home=game.home,
                    away=game.away,
                    commence=game.commence,
                    quotes=kept,
                )
            )
    return out


def fetch(sport: str, api_key: str | None, *, regions: str = "us", timeout: int = 30) -> list[Game]:
    if not api_key:
        return []
    session = http.session()
    try:
        resp = session.get(
            f"{BASE}/{SPORT_KEYS[sport]}/odds",
            params={
                "apiKey": api_key,
                "regions": regions,
                "markets": ",".join(MARKETS),
                "oddsFormat": "american",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # network, auth, or malformed payload
        log.warning("line-shop board fetch failed (%s): %s", sport, exc)
        return []
    return parse(sport, payload)


def parse(sport: str, payload: object) -> list[Game]:
    if not isinstance(payload, list):
        return []
    games: list[Game] = []
    for raw in payload:
        if not isinstance(raw, dict) or not raw.get("home_team"):
            continue
        game = Game(
            sport=sport,
            game_id=str(raw.get("id", "")),
            home=str(raw["home_team"]),
            away=str(raw.get("away_team", "")),
            commence=str(raw.get("commence_time", "")),
        )
        for bookmaker in raw.get("bookmakers") or []:
            book = str(bookmaker.get("title") or bookmaker.get("key") or "")
            for market in bookmaker.get("markets") or []:
                key = str(market.get("key", ""))
                if key not in MARKETS:
                    continue
                for outcome in market.get("outcomes") or []:
                    price = outcome.get("price")
                    side = str(outcome.get("name", ""))
                    if price is None or not side:
                        continue
                    point = outcome.get("point")
                    game.quotes.setdefault((key, side), []).append(
                        Quote(
                            book=book,
                            american=int(price),
                            point=None if point is None else float(point),
                        )
                    )
        if game.quotes:
            games.append(game)
    return games
