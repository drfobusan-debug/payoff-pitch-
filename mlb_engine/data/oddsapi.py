"""The Odds API (https://the-odds-api.com) client.

Fetches multi-book American prices for the markets VSIN does not expose a price
for -- full-game run line & total, first-five (F5) moneyline/run-line/total, and
batter/pitcher player props -- and maps them onto the engine's
``(matchup, market, selection)`` quote keys (see ``market.keys``).

Full-game markets come from the bulk odds endpoint (one request). F5 and player
props are per-event markets, so they cost one request per game; they are only
fetched when ``include_props`` is set. All access is via an API key; with no key
the client is inert and the engine falls back to VSIN/model-only behavior.

The vendor charges *markets x regions* per request, so a 16-game slate asking
for every market it can name costs ~230 credits -- enough to drain a 20k plan in
three months. Three things keep that down: ``_PROP_MARKETS`` lists only the
props the engine actually bets, responses are cached on disk for the slate, and
``x-requests-remaining`` is tracked so the per-event loop stops before it
exhausts the plan instead of after.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from mlb_engine.data import http
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
# Every market above can be *parsed*; only these are worth *paying* for. Over 54
# graded slates the engine produced zero favored picks in HR, doubles, runs and
# RBI (75-87% NPV -- it is right to abstain), so buying those prices is spend
# with no bet attached; they stay excluded. Earned runs (45.5% PPV) is priced
# below break-even and is not bought. Singles is fetched not to buy the over
# (50.4% PPV) but to capture the *under* price: the model fades ~90% of singles
# overs at a ~74% NPV, and persisting the under quote lets the audit grade that
# fade as a bettable under. Override with MLBE_ODDS_PROPS.
DEFAULT_PROP_MARKETS = (
    "batter_hits",
    "batter_singles",
    "pitcher_strikeouts",
    "pitcher_outs",
    "pitcher_hits_allowed",
    "pitcher_walks",
)
# Markets fetched only to record the *other* side's price (the under), never to
# bet the side we price. Pricing a market normally makes it bettable -- a pick
# is only forced to Pass when it has no quote -- so these are hard-passed after
# classification so the under quote is persisted without ever recommending the
# over. Singles is priced at ~48.8% PPV, below break-even, so its over is never
# a bet; we keep the under for the NPV-fade audit.
PRICE_ONLY_MARKETS = frozenset({"batter_1b"})
_PROP_MARKETS = list(_BATTER_MARKETS) + list(_PITCHER_MARKETS)

Quotes = dict[tuple[str, str, str], list[MarketQuote]]


class _Event:
    __slots__ = (
        "event_id", "matchup", "home_ab", "away_ab", "home_name", "away_name", "commence",
    )

    def __init__(self, event_id: str, home_ab: str, away_ab: str,
                 home_name: str, away_name: str,
                 commence: datetime | None = None) -> None:
        self.event_id = event_id
        self.matchup = f"{away_ab} @ {home_ab}"
        self.home_ab = home_ab
        self.away_ab = away_ab
        self.home_name = home_name
        self.away_name = away_name
        self.commence = commence

    def started(self, now: datetime | None = None) -> bool:
        """Whether first pitch has passed, so any price quoted is in-play."""
        if self.commence is None:
            return False
        return self.commence <= (now or datetime.now(timezone.utc))


class OddsAPIClient:
    def __init__(
        self,
        api_key: str | None,
        timeout: int = 25,
        *,
        prop_markets: tuple[str, ...] = DEFAULT_PROP_MARKETS,
        include_f5: bool = True,
        cache_dir: Path | None = None,
        cache_ttl: int = 1800,
        min_credits: int = 200,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.prop_markets = tuple(m for m in prop_markets if m in _PROP_MARKETS)
        self.include_f5 = include_f5
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.min_credits = min_credits
        self.credits_remaining: int | None = None

    def available(self) -> bool:
        return bool(self.api_key)

    def event_markets(self) -> list[str]:
        """Markets requested per event. Each one costs a credit per region."""
        return (list(_F5_MARKETS) if self.include_f5 else []) + list(self.prop_markets)

    def fetch(
        self, slate: Slate, *, include_props: bool = True, pregame_only: bool = False
    ) -> Quotes:
        """Return priced quotes across books for the slate.

        Full-game ML/run-line/total from one bulk call; F5 and props per event
        (only when ``include_props``). Returns ``{}`` if no key or on failure.

        ``pregame_only`` drops games that have already started. The vendor keeps
        quoting a game once it is under way, and those in-play prices are not
        comparable to the ones we bet -- a team up 6-0 in the 7th is -2000, which
        as a "close" would read as an enormous edge or an enormous miss purely
        from the score. The closing capture sets it; a pregame run does not need
        it, and a re-run of a finished slate (the audit regenerating yesterday's
        picks) must not have it or it would price nothing at all.
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
        started = 0
        for raw in data:
            ev = self._to_event(raw, norm_to_ab)
            if ev is None:
                continue
            if pregame_only and ev.started():
                started += 1
                continue
            events.append(ev)
            self._parse_game(raw, ev, out, f5=False)
        if started:
            log.info("Odds API: skipped %d game(s) already under way", started)

        markets = self.event_markets()
        if include_props and markets:
            cost = len(markets)
            log.info(
                "Odds API: %d events x %d markets = ~%d credits (%s remaining)",
                len(events), cost, len(events) * cost,
                self.credits_remaining if self.credits_remaining is not None else "?",
            )
            for ev in events:
                if not self._afford(cost):
                    break
                self._fetch_event(ev, out, markets)
        return out

    def _afford(self, cost: int) -> bool:
        """Stop the per-event loop before the plan is drained, not after."""
        if self.credits_remaining is None or self.credits_remaining - cost >= self.min_credits:
            return True
        log.warning(
            "Odds API: %d credits left, holding back the %d-credit reserve; "
            "remaining events priced from the model only",
            self.credits_remaining, self.min_credits,
        )
        return False

    # -- per-event (F5 + props) -------------------------------------------
    def _fetch_event(self, ev: _Event, out: Quotes, markets: list[str]) -> None:
        raw = self._get_json(f"{BASE}/events/{ev.event_id}/odds", markets=",".join(markets))
        if not isinstance(raw, dict):
            return
        if self.include_f5:
            self._parse_game(raw, ev, out, f5=True)
        self._parse_props(raw, ev, out)

    # -- parsing ----------------------------------------------------------
    def _parse_game(self, raw: dict, ev: _Event, out: Quotes, *, f5: bool) -> None:
        for bm in raw.get("bookmakers", []):
            book = bm.get("key", "")
            for mkt in bm.get("markets", []):
                mk = mkt.get("key", "")
                opposite = _opposite_prices(mkt.get("outcomes", []))
                for oc in mkt.get("outcomes", []):
                    price = oc.get("price")
                    if price is None:
                        continue
                    sel = self._game_selection(mk, oc, ev, f5)
                    if sel is None:
                        continue
                    market, selection = sel
                    out.setdefault((ev.matchup, market, selection), []).append(
                        MarketQuote(
                            book=book,
                            american=float(price),
                            opposite_american=opposite.get(id(oc)),
                        )
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
                opposite = _opposite_prices(mkt.get("outcomes", []))
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
                        MarketQuote(
                            book=book,
                            american=float(price),
                            # The under is not bet, but it is what makes the
                            # over's fair probability computable.
                            opposite_american=opposite.get(id(oc)),
                        )
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
        return _Event(
            str(eid), home_ab, away_ab, home_name, away_name,
            _commence_time(raw.get("commence_time")),
        )

    def _cache_path(self, url: str, params: dict[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        stamp = json.dumps({"url": url, **params}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(stamp.encode()).hexdigest()[:20]}.json"

    def _get_json(self, url: str, **params: str) -> object:
        q = {"regions": "us", "oddsFormat": "american", **params}
        # Keyed on the query minus the key, so a re-run of the same slate is free.
        cache = self._cache_path(url, q)
        if cache is not None and cache.exists():
            if time.time() - cache.stat().st_mtime < self.cache_ttl:
                try:
                    return json.loads(cache.read_text())
                except ValueError:
                    pass
        try:
            resp = http.get(url, params={"apiKey": self.api_key or "", **q},
                            timeout=self.timeout)
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
            # requests puts the full request URL -- query string and API key
            # included -- into the exception message, so never log it verbatim.
            log.warning(
                "Odds API request failed (%s): %s",
                url.rsplit("/", 1)[-1],
                _redact(str(exc), self.api_key),
            )
            return None


def _commence_time(raw: object) -> datetime | None:
    """First pitch from the vendor's ISO-8601 UTC stamp, or ``None`` if absent."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _opposite_prices(outcomes: list[dict]) -> dict[int, float]:
    """Map each outcome to the price of the other side of the same two-way line.

    A two-outcome market is a pair outright, which covers moneylines and run
    lines (whose two sides carry *different* points, -1.5 and +1.5). Anything
    larger is grouped by player and line, so a prop or alternate total pairs its
    own over with its own under. Only exact pairs qualify -- a three-way or
    orphaned side is left unpaired rather than devigged against the wrong price.
    """
    priced = [oc for oc in outcomes if oc.get("price") is not None]
    if len(priced) == 2:
        a, b = priced
        return {id(a): float(b["price"]), id(b): float(a["price"])}
    groups: dict[tuple[str, object], list[dict]] = {}
    for oc in outcomes:
        if oc.get("price") is None:
            continue
        groups.setdefault((str(oc.get("description", "")), oc.get("point")), []).append(oc)
    pairs: dict[int, float] = {}
    for members in groups.values():
        if len(members) != 2:
            continue
        a, b = members
        pairs[id(a)] = float(b["price"])
        pairs[id(b)] = float(a["price"])
    return pairs


def _redact(message: str, api_key: str | None) -> str:
    """Strip the API key (and any other query string) out of a message."""
    if api_key:
        message = message.replace(api_key, "***")
    return re.sub(r"\?[^\s]*", "?<redacted>", message)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
