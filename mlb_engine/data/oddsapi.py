"""The Odds API (https://the-odds-api.com) client.

Fetches multi-book American prices for the markets VSIN does not expose a price
for -- full-game run line & total, first-five (F5) moneyline/run-line/total, and
batter/pitcher player props -- and maps them onto the engine's
``(matchup, market, selection)`` quote keys (see ``market.keys``).

Full-game markets come from the bulk odds endpoint (one request). F5 and player
props are per-event markets, so they cost one request per game; they are only
fetched when ``include_props`` is set. All access is via an API key; with no key
the client is inert and the engine falls back to VSIN/model-only behavior.
"""

from __future__ import annotations

import logging
import re

import requests

from mlb_engine.market import keys
from mlb_engine.market.ev import MarketQuote
from mlb_engine.schemas import Slate

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
_GAME_MARKETS = ["h2h", "spreads", "totals"]
_F5_MARKETS = ["h2h_1st_5_innings", "spreads_1st_5_innings", "totals_1st_5_innings"]
# The Odds API player-prop market -> (engine market, engine stat symbol / label).
_BATTER_MARKETS = {
    "batter_hits": ("batter_h", "H"),
    "batter_singles": ("batter_1b", "1B"),
    "batter_doubles": ("batter_2b", "2B"),
    "batter_home_runs": ("batter_hr", "HR"),
    "batter_runs_scored": ("batter_r", "R"),
    "batter_rbis": ("batter_rbi", "RBI"),
}
_PITCHER_MARKETS = {
    "pitcher_strikeouts": ("pitcher_k", "Ks"),
    "pitcher_outs": ("pitcher_outs", "Outs"),
    "pitcher_hits_allowed": ("pitcher_h", "Hits"),
    "pitcher_walks": ("pitcher_bb", "Walks"),
    "pitcher_earned_runs": ("pitcher_er", "ER"),
}
_PROP_MARKETS = list(_BATTER_MARKETS) + list(_PITCHER_MARKETS)

Quotes = dict[tuple[str, str, str], list[MarketQuote]]


class _Event:
    __slots__ = ("event_id", "matchup", "home_ab", "away_ab", "home_name", "away_name")

    def __init__(self, event_id: str, home_ab: str, away_ab: str,
                 home_name: str, away_name: str) -> None:
        self.event_id = event_id
        self.matchup = f"{away_ab} @ {home_ab}"
        self.home_ab = home_ab
        self.away_ab = away_ab
        self.home_name = home_name
        self.away_name = away_name


class OddsAPIClient:
    def __init__(self, api_key: str | None, timeout: int = 25) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, slate: Slate, *, include_props: bool = True) -> Quotes:
        """Return priced quotes across books for the slate.

        Full-game ML/run-line/total from one bulk call; F5 and props per event
        (only when ``include_props``). Returns ``{}`` if no key or on failure.
        """
        if not self.available():
            return {}
        norm_to_ab: dict[str, str] = {}
        for g in slate.games:
            norm_to_ab[_norm(g.home.name)] = g.home.abbrev
            norm_to_ab[_norm(g.away.name)] = g.away.abbrev

        data = self._get_json(f"{BASE}/odds", markets=",".join(_GAME_MARKETS))
        if not isinstance(data, list):
            return {}

        out: Quotes = {}
        events: list[_Event] = []
        for raw in data:
            ev = self._to_event(raw, norm_to_ab)
            if ev is None:
                continue
            events.append(ev)
            self._parse_game(raw, ev, out, f5=False)

        if include_props:
            for ev in events:
                self._fetch_event(ev, out)
        return out

    # -- per-event (F5 + props) -------------------------------------------
    def _fetch_event(self, ev: _Event, out: Quotes) -> None:
        markets = ",".join(_F5_MARKETS + _PROP_MARKETS)
        raw = self._get_json(f"{BASE}/events/{ev.event_id}/odds", markets=markets)
        if not isinstance(raw, dict):
            return
        self._parse_game(raw, ev, out, f5=True)
        self._parse_props(raw, ev, out)

    # -- parsing ----------------------------------------------------------
    def _parse_game(self, raw: dict, ev: _Event, out: Quotes, *, f5: bool) -> None:
        for bm in raw.get("bookmakers", []):
            book = bm.get("key", "")
            for mkt in bm.get("markets", []):
                mk = mkt.get("key", "")
                for oc in mkt.get("outcomes", []):
                    price = oc.get("price")
                    if price is None:
                        continue
                    sel = self._game_selection(mk, oc, ev, f5)
                    if sel is None:
                        continue
                    market, selection = sel
                    out.setdefault((ev.matchup, market, selection), []).append(
                        MarketQuote(book=book, american=float(price))
                    )

    def _game_selection(
        self, mk: str, oc: dict, ev: _Event, f5: bool
    ) -> tuple[str, str] | None:
        name = _norm(str(oc.get("name", "")))
        ab = ev.home_ab if name == ev.home_name else ev.away_ab if name == ev.away_name else None
        point = oc.get("point")
        if mk in ("h2h", "h2h_1st_5_innings"):
            if ab is None:
                return None
            return ("f5_ml", keys.f5_ml(ab)) if f5 else ("game_ml", keys.game_ml(ab))
        if mk in ("spreads", "spreads_1st_5_innings"):
            if ab is None or point is None:
                return None
            pt = float(point)
            return ("f5_rl", keys.f5_rl(ab, pt)) if f5 else ("game_rl", keys.game_rl(ab, pt))
        if mk in ("totals", "totals_1st_5_innings"):
            if point is None:
                return None
            over = str(oc.get("name", "")).lower().startswith("over")
            pt = float(point)
            return (
                ("f5_total", keys.f5_total(over, pt)) if f5
                else ("game_total", keys.game_total(over, pt))
            )
        return None

    def _parse_props(self, raw: dict, ev: _Event, out: Quotes) -> None:
        for bm in raw.get("bookmakers", []):
            book = bm.get("key", "")
            for mkt in bm.get("markets", []):
                mk = mkt.get("key", "")
                if mk in _BATTER_MARKETS:
                    market, stat = _BATTER_MARKETS[mk]
                    is_pitcher = False
                elif mk in _PITCHER_MARKETS:
                    market, stat = _PITCHER_MARKETS[mk]
                    is_pitcher = True
                else:
                    continue
                for oc in mkt.get("outcomes", []):
                    if str(oc.get("name", "")).lower() != "over":
                        continue
                    price, point, player = oc.get("price"), oc.get("point"), oc.get("description")
                    if price is None or point is None or not player:
                        continue
                    line = float(point)
                    selection = (
                        keys.pitcher_prop(str(player), stat, line) if is_pitcher
                        else keys.batter_prop(str(player), stat, line)
                    )
                    out.setdefault((ev.matchup, market, selection), []).append(
                        MarketQuote(book=book, american=float(price))
                    )

    # -- helpers ----------------------------------------------------------
    def _to_event(self, raw: dict, norm_to_ab: dict[str, str]) -> _Event | None:
        home_name = _norm(str(raw.get("home_team", "")))
        away_name = _norm(str(raw.get("away_team", "")))
        home_ab = norm_to_ab.get(home_name)
        away_ab = norm_to_ab.get(away_name)
        eid = raw.get("id")
        if home_ab is None or away_ab is None or not eid:
            return None
        return _Event(str(eid), home_ab, away_ab, home_name, away_name)

    def _get_json(self, url: str, **params: str) -> object:
        q = {"apiKey": self.api_key, "regions": "us", "oddsFormat": "american", **params}
        try:
            resp = requests.get(url, params=q, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Odds API request failed (%s): %s", url.rsplit("/", 1)[-1], exc)
            return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
