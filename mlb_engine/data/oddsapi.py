"""The Odds API (https://the-odds-api.com) client.

Fetches multi-book American prices for the markets VSIN does not expose a price
for -- full-game run line & total, first-five (F5) moneyline/run-line/total, and
batter/pitcher player props -- and maps them onto the engine's
``(matchup, market, selection)`` quote keys (see ``market.keys``).

Full-game markets come from the bulk odds endpoint (one request). F5 and player
props are per-event markets, so they cost one request per game; they are only
fetched when ``include_props`` is set. All access is via an API key; with no key
the client is inert and the engine falls back to VSIN/model-only behavior.

The two halves are deliberately independent. The bulk call used to double as the
event-id source, so a single failure on it -- a 401 from an unreadable key file,
a 422, a read timeout past the retry budget -- returned ``{}`` and the per-event
loop never ran, leaving a whole slate priced from VSIN moneylines alone. Ten of
twenty audited slates lost every prop that way. Event ids now come from
``/events``, which the vendor bills at *zero* credits, so the props survive a
bad bulk call and vice versa.

The vendor charges *markets x regions* per request (a market that returns no
book is not billed), so a 15-game slate asking for every market it can name
costs ~250 credits -- ~7.5k a month against a 100k plan. Responses are cached on
disk for the slate and ``x-requests-remaining`` is tracked so the per-event loop
stops before it exhausts the plan instead of after.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
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
    "batter_total_bases": ("batter_tb", "TB"),
    "batter_hits_runs_rbis": ("batter_hrr", "H+R+RBI"),
    "batter_walks": ("batter_bb", "BB"),
    "batter_strikeouts": ("batter_k", "K"),
    "batter_stolen_bases": ("batter_sb", "SB"),
}
_PITCHER_MARKETS = {
    "pitcher_strikeouts": ("pitcher_k", "Ks"),
    "pitcher_outs": ("pitcher_outs", "Outs"),
    "pitcher_hits_allowed": ("pitcher_h", "Hits"),
    "pitcher_walks": ("pitcher_bb", "Walks"),
    "pitcher_earned_runs": ("pitcher_er", "ER"),
}
# Every parseable market is now bought, because the argument for excluding some
# of them was circular. The old list dropped HR, doubles, runs, RBI and earned
# runs on the grounds that the engine "produced zero favored picks (75-87% NPV
# -- it is right to abstain)". But a market the engine fades 100% of the time
# scores NPV equal to its base rate for free: measured against that baseline the
# fade skill in HR, doubles and RBI is +0.000, so the high NPV was arithmetic,
# not evidence of correct abstention -- and dropping the price meant the audit
# could never find out. Total bases and H+R+RBI were never mapped at all, so
# 23.5k ledger rows were graded at an assumed -110 against a price that never
# existed; total bases' "-40% ROI" was measured entirely against that phantom.
# The cost of buying all of it is ~7.5k credits a month on a 100k plan.
# Override with MLBE_ODDS_PROPS.
DEFAULT_PROP_MARKETS = (
    "batter_hits",
    "batter_singles",
    "batter_doubles",
    "batter_home_runs",
    "batter_runs_scored",
    "batter_rbis",
    "batter_total_bases",
    "batter_hits_runs_rbis",
    "batter_walks",
    "batter_strikeouts",
    "batter_stolen_bases",
    "pitcher_strikeouts",
    "pitcher_outs",
    "pitcher_hits_allowed",
    "pitcher_walks",
    "pitcher_earned_runs",
)
# Markets fetched only to record the *other* side's price (the under), never to
# bet the side we price. Pricing a market normally makes it bettable -- a pick
# is only forced to Pass when it has no quote -- so these are hard-passed after
# classification so the under quote is persisted without ever recommending the
# over. Singles is priced at ~48.8% PPV, below break-even, so its over is never
# a bet; we keep the under for the NPV-fade audit.
#
# The markets restored to the fetch list above join it. Buying a price must not
# be what decides whether we bet a market: the point of paying for them is to
# replace the assumed -110 the audit has been grading them at with a real
# number, and that has to happen before -- not after -- they become bettable.
# The set of markets the engine actually bets is therefore unchanged by the
# restoration; promoting one out of here is a separate decision, on evidence
# this capture is what finally supplies.
#
# The capture has now supplied that evidence, so five markets are promoted out.
# Each carries the price rule its own graded record earns, and the rules are
# hard screens rather than EV inputs because the EV is what was wrong:
#
#   batter_2b    +12.8u, +25.7% on 50 bets     no rule -- it was the one winner
#   batter_hr    -36.4u                        buy only +400..+700 (see config)
#   batter_1b    -22.5u -> +3.5u at plus money singles price floor
#   batter_rbi   -21.5u -> -1.0u above p=.40   RBI conviction floor
#   batter_hrr   never bet, so never measured  reopened to measure it
#   batter_tb    -12.3% counterfactual         the #74 power/contact gate
#
# Total bases is the weakest case of the six, and it is reopened on the strength
# of already having a screen: #74 gates it on barrel rate, exit velocity and the
# opposing starter's contact suppression, and that gate has never once been
# allowed to decide a bet. Nothing in the counterfactual argues for a second
# one -- a 0.50 conviction floor lifts -11.0% to only -6.7%, and it helps in one
# half of the window while hurting in the other, which is the shape of noise.
#
# ``batter_r`` stays shut: no probability band and no price bucket of it has
# ever paid (-41.4u, -31.8%), so there is no rule to reopen it behind. Pitcher
# ER stays shut for want of anyone having looked.
# A batter's own walks and strikeouts are new to the fetch list and have never
# been graded here, so they are quoted and never bought -- the order every other
# market was reopened in: buy the price, let the ledger earn the bet. They are
# priced at all because an outside prop board carries them heavily (two of EV
# Analytics' largest sections) and a market the engine does not price cannot be
# compared against anybody.
# Stolen bases joins them, and has the furthest to earn: it is the one market
# here whose event the simulator does not draw at all (the run models hand out
# bases anonymously at a league rate), so it is priced from a season rate on top
# of the simulated times on first. Quoted only, until the ledger says otherwise.
PRICE_ONLY_MARKETS = frozenset(
    {
        "batter_r",
        "pitcher_er",
        "batter_bb",
        "batter_k",
        "batter_sb",
    }
)
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


# How far an event's first pitch may sit from a scheduled one and still be the
# same game. Wide enough for a rain delay or the nightcap of a doubleheader
# (~3.5h later, and listed on the slate in its own right), far short of the ~24h
# to the next meeting of the same two clubs.
_SAME_GAME_WINDOW = timedelta(hours=8)


class _SlateIndex:
    """Which vendor events belong to the slate being priced.

    Team names alone are not enough. The vendor's board runs days ahead, and a
    series means the same two clubs appear on it three nights running -- all of
    them matching the slate's name map. A 10-game slate resolved 17 events that
    way, and the seven extras were tomorrow's games: billed for, and empty of
    props because the books had not posted them yet.
    """

    __slots__ = ("norm_to_ab", "scheduled")

    def __init__(self, slate: Slate) -> None:
        self.norm_to_ab: dict[str, str] = {}
        self.scheduled: dict[tuple[str, str], list[datetime]] = {}
        for g in slate.games:
            self.norm_to_ab[_norm(g.home.name)] = g.home.abbrev
            self.norm_to_ab[_norm(g.away.name)] = g.away.abbrev
            first_pitch = _commence_time(g.game_datetime_utc)
            if first_pitch is not None:
                self.scheduled.setdefault((g.home.abbrev, g.away.abbrev), []).append(first_pitch)

    def on_slate(self, ev: _Event) -> bool:
        """Whether this event is one of the slate's games rather than a later one.

        Unknown times are kept, not dropped: the vendor omitting a commence
        stamp should cost us a wasted credit, never a missing price.
        """
        times = self.scheduled.get((ev.home_ab, ev.away_ab))
        if not times or ev.commence is None:
            return True
        return any(abs(ev.commence - t) <= _SAME_GAME_WINDOW for t in times)


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

        The bulk call and the per-event loop fail independently: losing the game
        board must not also lose every prop on the slate.

        ``pregame_only`` drops games that have already started. The vendor keeps
        quoting a game once it is under way, and those in-play prices are not
        comparable to the ones we bet -- a team up 6-0 in the 7th is -2000, which
        as a "close" would read as an enormous edge or an enormous miss purely
        from the score. The closing capture sets it; a pregame run does not need
        it, and a re-run of a finished slate (the audit regenerating yesterday's
        picks) must not have it or it would price nothing at all.
        """
        if not self.available():
            log.warning(
                "Odds API: no key configured -- the slate is priced from VSIN "
                "and the model only, and every prop will grade at an assumed price"
            )
            return {}
        index = _SlateIndex(slate)

        out: Quotes = {}
        events = self._fetch_game_board(index, out, pregame_only=pregame_only)
        if not events:
            # The board failed or matched nothing. Event ids are free, so fall
            # back to them rather than abandoning the props too.
            events = self._list_events(index, pregame_only=pregame_only)
        if not events:
            log.error(
                "Odds API: no events resolved for the slate -- nothing priced. "
                "Check the API key and the plan's remaining credits"
            )
            return out

        markets = self.event_markets()
        if include_props and markets:
            cost = len(markets)
            log.info(
                "Odds API: %d events x %d markets = ~%d credits (%s remaining)",
                len(events), cost, len(events) * cost,
                self.credits_remaining if self.credits_remaining is not None else "?",
            )
            priced = 0
            for ev in events:
                if not self._afford(cost):
                    break
                priced += int(self._fetch_event(ev, out, markets))
            if priced == 0:
                log.error(
                    "Odds API: all %d per-event requests came back empty -- the "
                    "slate has no prop prices and they will grade at an assumed "
                    "price. This is a fetch failure, not an absent market",
                    len(events),
                )
            elif priced < len(events):
                log.warning(
                    "Odds API: priced %d of %d events; the rest have no prop quotes",
                    priced, len(events),
                )
        return out

    # -- events -----------------------------------------------------------
    def _fetch_game_board(
        self, index: _SlateIndex, out: Quotes, *, pregame_only: bool
    ) -> list[_Event]:
        """Bulk game markets. Returns the events it resolved, or ``[]`` on failure."""
        data = self._get_json(f"{BASE}/odds", markets=",".join(_GAME_MARKETS))
        if not isinstance(data, list):
            log.warning(
                "Odds API: the bulk game board failed; falling back to the free "
                "event list so props are still priced"
            )
            return []
        events: list[_Event] = []
        started = 0
        later = 0
        for raw in data:
            ev = self._to_event(raw, index.norm_to_ab)
            if ev is None:
                continue
            if not index.on_slate(ev):
                later += 1
                continue
            if pregame_only and ev.started():
                started += 1
                continue
            events.append(ev)
            self._parse_game(raw, ev, out, f5=False)
        if started:
            log.info("Odds API: skipped %d game(s) already under way", started)
        if later:
            log.info("Odds API: skipped %d game(s) on a later date", later)
        return events

    def _list_events(self, index: _SlateIndex, *, pregame_only: bool) -> list[_Event]:
        """Event ids only. The vendor bills this endpoint at zero credits."""
        data = self._get_json(f"{BASE}/events")
        if not isinstance(data, list):
            return []
        events = (self._to_event(raw, index.norm_to_ab) for raw in data)
        return [
            ev for ev in events
            if ev is not None and index.on_slate(ev) and not (pregame_only and ev.started())
        ]

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
    def _fetch_event(self, ev: _Event, out: Quotes, markets: list[str]) -> bool:
        """Price one event. Returns whether the vendor returned a usable board."""
        raw = self._get_json(f"{BASE}/events/{ev.event_id}/odds", markets=",".join(markets))
        if not isinstance(raw, dict):
            return False
        if self.include_f5:
            self._parse_game(raw, ev, out, f5=True)
        self._parse_props(raw, ev, out)
        return bool(raw.get("bookmakers"))

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
                    side = str(oc.get("name", "")).lower()
                    if side not in ("over", "under"):
                        continue
                    price, point, player = oc.get("price"), oc.get("point"), oc.get("description")
                    if price is None or point is None or not player:
                        continue
                    line = float(point)
                    selection = (
                        keys.pitcher_prop(str(player), stat, line, side) if is_pitcher
                        else keys.batter_prop(str(player), stat, line, side)
                    )
                    out.setdefault((ev.matchup, market, selection), []).append(
                        MarketQuote(
                            book=book,
                            american=float(price),
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

    A two-outcome *team* market is a pair outright, which covers moneylines and
    run lines (whose two sides carry *different* points, -1.5 and +1.5, so they
    cannot be grouped by point). Anything else is grouped by player and line, so
    a prop or alternate total pairs its own over with its own under. Only exact
    pairs qualify -- a three-way or orphaned side is left unpaired rather than
    devigged against the wrong price.

    The two-outcome path must check that the two sides really are opposite. A
    book that lists only the over for two different players is also a
    two-outcome market, and pairing those devigs one longshot against another:
    an over at +390 against an unrelated over at +575 returns .579 where the
    honest number is .196, and a probability *above* the raw implied is the
    signature, since removing vig can only move a side down.
    """
    priced = [oc for oc in outcomes if oc.get("price") is not None]
    if len(priced) == 2 and _is_two_way(priced[0], priced[1]):
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
        if not _is_opposite_side(a, b):
            continue
        pairs[id(a)] = float(b["price"])
        pairs[id(b)] = float(a["price"])
    return pairs


def _is_two_way(a: dict, b: dict) -> bool:
    """Are these two outcomes the two sides of one line?

    True for a team market, where the outcome names are the two teams and no
    player is named. For anything carrying a player, the two sides must belong
    to that same player and be opposite sides of the same line.
    """
    da, db = str(a.get("description", "")), str(b.get("description", ""))
    if not da and not db:
        return _is_opposite_side(a, b)
    return da == db and _is_opposite_side(a, b)


def _is_opposite_side(a: dict, b: dict) -> bool:
    """Two outcomes are opposite sides only if their names differ."""
    return str(a.get("name", "")).strip().lower() != str(b.get("name", "")).strip().lower()


def _redact(message: str, api_key: str | None) -> str:
    """Strip the API key (and any other query string) out of a message."""
    if api_key:
        message = message.replace(api_key, "***")
    return re.sub(r"\?[^\s]*", "?<redacted>", message)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
