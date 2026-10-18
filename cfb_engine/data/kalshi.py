"""Kalshi CFB full-game total ladders, read from the public market-data API.

Each game is a ladder of binary "Over X.5" contracts paying $1. No key and no
Odds API credits: the ``/markets`` endpoint is public. The ladder is recorded
next to the books' totals so a graded total can be scored at Kalshi's ask.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests

from cfb_engine.data.teamnames import label_matches
from cfb_engine.schemas import Game
from mlb_engine.data import http

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2/markets"
TOTALS_SERIES = "KXNCAAFTOTAL"
_TEAMS = re.compile(r"in the (.+?) vs (.+?) college football game")
# A contract settles hours after kickoff; one more than that is another game.
_SAME_GAME = timedelta(hours=30)
_MAX_PAGES = 20


@dataclass(frozen=True)
class Strike:
    """One rung: Yes pays if the game total goes over ``strike``."""

    strike: float
    yes_bid: float
    yes_ask: float
    open_interest: float


@dataclass(frozen=True)
class Ladder:
    teams: tuple[str, str]
    occurs: datetime | None
    strikes: tuple[Strike, ...]


def _float(raw: object) -> float | None:
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return None


def _when(raw: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_markets(markets: list[dict]) -> list[Ladder]:
    """Group open total contracts by event into one ladder per game."""
    by_event: dict[str, tuple[tuple[str, str], datetime | None, list[Strike]]] = {}
    for m in markets:
        teams = _TEAMS.search(str(m.get("rules_primary", "")))
        strike = _float(m.get("floor_strike"))
        bid, ask = _float(m.get("yes_bid_dollars")), _float(m.get("yes_ask_dollars"))
        if teams is None or strike is None or bid is None or ask is None:
            continue
        oi = _float(m.get("open_interest_fp")) or 0.0
        _, _, rungs = by_event.setdefault(
            str(m.get("event_ticker")),
            ((teams.group(1), teams.group(2)), _when(m.get("occurrence_datetime")), []),
        )
        rungs.append(Strike(strike, bid, ask, oi))
    return [
        Ladder(teams, occurs, tuple(sorted(rungs, key=lambda s: s.strike)))
        for teams, occurs, rungs in by_event.values()
    ]


def match(game: Game, ladders: list[Ladder]) -> Ladder | None:
    """The one ladder naming both of ``game``'s teams around its kickoff."""
    kick = _when(game.commence_time_utc) if game.commence_time_utc else None
    found = []
    for lad in ladders:
        a, b = lad.teams
        names = (game.home.name, game.away.name)
        if not (
            (label_matches(a, names[0]) and label_matches(b, names[1]))
            or (label_matches(a, names[1]) and label_matches(b, names[0]))
        ):
            continue
        if kick is not None and lad.occurs is not None and abs(lad.occurs - kick) > _SAME_GAME:
            continue
        found.append(lad)
    return found[0] if len(found) == 1 else None


class KalshiClient:
    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def totals(self) -> list[Ladder]:
        """Every open CFB full-game total ladder; empty on any failure."""
        markets: list[dict] = []
        cursor = ""
        try:
            for _ in range(_MAX_PAGES):
                params = {"series_ticker": TOTALS_SERIES, "status": "open", "limit": "1000"}
                if cursor:
                    params["cursor"] = cursor
                resp = http.get(BASE, params=params, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                markets.extend(body.get("markets") or [])
                cursor = body.get("cursor") or ""
                if not cursor:
                    break
        except (requests.RequestException, ValueError) as exc:
            log.warning("Kalshi totals unavailable: %s", exc)
            return []
        return parse_markets(markets)
