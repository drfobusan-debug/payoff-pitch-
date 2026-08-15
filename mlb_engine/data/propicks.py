"""VSiN's VOLT and JOLT model picks, shown beside our own on the same bet.

VSiN publishes two model card feeds at ``data.vsin.com/propicks``: VOLT on game
markets (totals, sides) and JOLT on player props. Both pages are ordinary
server-rendered HTML -- no login, no key -- and every card carries the side, the
line, the book price, the model's claimed edge and an AI-written write-up::

    <div class="fp-card">
      <span class="fp-band-expert">VSiN VOLT Model</span>
      <span class="fp-edge">7.9% EDGE</span>
      <div class="fp-player">White Sox at Tigers</div>
      <div class="fp-team-label">Game Total</div>
      <span class="fp-side fp-side-under">UNDER 7.5</span>
      <span class="fp-market">Total Runs</span>
      <span class="fp-price">-102</span>

This is a benchmark, never an input. Nothing here touches a probability, a tier
or a price: an outside model that agrees is not evidence, it is a second opinion
with its own unmeasured biases, and folding it into our number would double-count
whatever it and we both read off the same public data. What it does buy is a
disagreement column -- where VOLT and the engine take opposite sides of one
total, one of us is wrong, and the ledger can eventually say which.

The feed is same-day only, so the record only exists if it is captured daily.
Fetches are best-effort throughout: a missing benchmark costs nothing, a card
that fails to parse is dropped rather than guessed at.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from html import unescape
from pathlib import Path

from mlb_engine.data import http
from mlb_engine.market import keys
from mlb_engine.recommendations import Recommendation

log = logging.getLogger(__name__)

BASE = "https://data.vsin.com/propicks"
VOLT_URL = f"{BASE}/volt/"
JOLT_URL = f"{BASE}/jolt/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}

_CARD = '<div class="fp-card">'
_TAG = re.compile(r"<[^>]+>")
_EDGE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SIDE = re.compile(r"^(OVER|UNDER)\s+(-?\d+(?:\.\d+)?)$", re.I)
_TEAM_LINE = re.compile(r"^(.+?)\s*([+-]\d+(?:\.\d+)?)$")
_AMERICAN = re.compile(r"[+-]\d+")

# VSiN writes game cards as "Away at Home" using nicknames only.
NICKNAMES = {
    "angels": "LAA", "astros": "HOU", "athletics": "ATH", "as": "ATH",
    "blue jays": "TOR", "braves": "ATL", "brewers": "MIL", "cardinals": "STL",
    "cubs": "CHC", "diamondbacks": "AZ", "d-backs": "AZ", "dbacks": "AZ",
    "dodgers": "LAD", "giants": "SF", "guardians": "CLE", "mariners": "SEA",
    "marlins": "MIA", "mets": "NYM", "nationals": "WSH", "orioles": "BAL",
    "padres": "SD", "phillies": "PHI", "pirates": "PIT", "rangers": "TEX",
    "rays": "TB", "red sox": "BOS", "reds": "CIN", "rockies": "COL",
    "royals": "KC", "tigers": "DET", "twins": "MIN", "white sox": "CWS",
    "yankees": "NYY",
}

# VSiN's market label -> the engine market it prices. Anything absent is kept as
# a pick with no engine market, so an unrecognised label shows up in the capture
# instead of vanishing quietly.
GAME_MARKETS = {
    "total runs": "game_total",
    "game total": "game_total",
    "moneyline": "game_ml",
    "run line": "game_rl",
    "runline": "game_rl",
    "f5 total runs": "f5_total",
    "first 5 total": "f5_total",
    "first 5 innings": "f5_ml",
    "f5 moneyline": "f5_ml",
}
PROP_MARKETS = {
    "total bases": "batter_tb",
    "h+r+rbi": "batter_hrr",
    "hits+runs+rbis": "batter_hrr",
    "hits": "batter_h",
    "home runs": "batter_hr",
    "doubles": "batter_2b",
    "runs scored": "batter_r",
    "rbis": "batter_rbi",
    "singles": "batter_1b",
    "outs recorded": "pitcher_outs",
    "strikeouts": "pitcher_k",
    "pitcher strikeouts": "pitcher_k",
    "earned runs": "pitcher_er",
    "hits allowed": "pitcher_h",
    "walks allowed": "pitcher_bb",
}


@dataclass(frozen=True)
class ProPick:
    """One VOLT or JOLT card, normalised onto the engine's market vocabulary."""

    model: str  # "VOLT" | "JOLT"
    league: str  # "MLB", "CFB", ...
    date: str  # ISO date of the event, "" when the card omits it
    subject: str  # "White Sox at Tigers" for a game, a player's name for a prop
    label: str  # VSiN's sub-label: "Game Total", or the player's team
    raw_market: str  # VSiN's own market name, verbatim
    market: str  # engine market key, "" where the label is unmapped
    matchup: str  # "AWAY @ HOME" for game cards, "" for props
    side: str  # "over" | "under" | team abbrev
    line: float | None
    price: float | None
    edge: float | None  # VSiN's claimed edge, as a fraction
    book: str

    @property
    def key(self) -> str:
        return "|".join((self.date, self.model, self.subject, self.raw_market, self.side))

    @property
    def summary(self) -> str:
        """The pick as VSiN states it, for the column beside our own selection."""
        head = self.side.upper() if self.side in ("over", "under") else self.side
        parts = [self.model, head]
        if self.line is not None:
            parts.append(f"{self.line:g}")
        if self.price is not None:
            parts.append(f"({self.price:+.0f})")
        return " ".join(parts)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG.sub(" ", fragment))).strip()


def _field(card: str, pattern: str) -> str:
    m = re.search(pattern, card, re.S)
    return _text(m.group(1)) if m else ""


def _american(text: str) -> float | None:
    m = _AMERICAN.search(text.replace("\u2212", "-"))
    return float(m.group()) if m else None


def _edge(text: str) -> float | None:
    m = _EDGE.search(text)
    return float(m.group(1)) / 100.0 if m else None


def _date(text: str) -> str:
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _abbrev(nickname: str) -> str:
    return NICKNAMES.get(nickname.strip().lower().replace(".", ""), "")


def _matchup(subject: str) -> str:
    """"White Sox at Tigers" -> "CWS @ DET"; "" when either side is unknown."""
    parts = re.split(r"\s+(?:at|@|vs\.?)\s+", subject, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return ""
    away, home = (_abbrev(p) for p in parts)
    return f"{away} @ {home}" if away and home else ""


def _side_and_line(text: str, market: str) -> tuple[str, float | None]:
    """The side taken, as ``over``/``under`` or a team abbreviation.

    A team card is spelled either "Tigers -1.5" (run line) or "Tigers" alone
    (moneyline), so the point rides on the same string as the side.
    """
    plain = text.strip()
    ou = _SIDE.match(plain)
    if ou:
        return ou.group(1).lower(), float(ou.group(2))
    spread = _TEAM_LINE.match(plain)
    if spread:
        abbrev = _abbrev(spread.group(1))
        return abbrev or spread.group(1), float(spread.group(2))
    if market in ("game_ml", "f5_ml"):
        return _abbrev(plain) or plain, None
    return "", None


def parse_cards(html: str, model: str) -> list[ProPick]:
    """Every pick card on a VOLT/JOLT page, skipping any that will not parse."""
    picks: list[ProPick] = []
    for card in html.split(_CARD)[1:]:
        league = _field(card, r'alt="([^"]+)" class="fp-band-league-logo"')
        raw_market = _field(card, r'class="fp-market">(.*?)</span>')
        label = _field(card, r'class="fp-team-label">(.*?)</div>')
        lower = raw_market.casefold()
        market = GAME_MARKETS.get(lower) or PROP_MARKETS.get(lower, "")
        side, line = _side_and_line(_field(card, r'class="fp-side[^"]*">(.*?)</span>'), market)
        if not side:
            log.debug("%s card with no readable side (%s); dropped", model, raw_market)
            continue
        subject = _field(card, r'class="fp-player">(.*?)</div>')
        is_game = market in GAME_MARKETS.values() or bool(_matchup(subject))
        picks.append(
            ProPick(
                model=model,
                league=league or "MLB",
                date=_date(_field(card, r'class="fp-date">(.*?)</span>')),
                subject=subject,
                label=label,
                raw_market=raw_market,
                market=market,
                matchup=_matchup(subject) if is_game else "",
                side=side,
                line=line,
                price=_american(_field(card, r'class="fp-price">(.*?)</span>')),
                edge=_edge(_field(card, r'class="fp-edge">(.*?)</span>')),
                book=_field(card, r'class="fp-book">(.*?)</span>'),
            )
        )
    return picks


def fetch(*, timeout: float = 20.0, league: str = "MLB") -> list[ProPick]:
    """Today's VOLT and JOLT cards. Empty on any failure -- never raises."""
    picks: list[ProPick] = []
    for model, url in (("VOLT", VOLT_URL), ("JOLT", JOLT_URL)):
        try:
            resp = http.get(url, headers=_HEADERS, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - a benchmark cannot break the card
            log.warning("VSiN %s picks unavailable (%s): %s", model, url, exc)
            continue
        found = parse_cards(resp.text, model)
        picks.extend(p for p in found if not league or p.league.upper() == league.upper())
        log.info("VSiN %s: %d cards", model, len(found))
    return picks


def _prop_key(market: str, player: str) -> str:
    return f"{market}|{keys.canonical(player)}"


def _game_key(market: str, matchup: str) -> str:
    return f"{market}|{matchup}"


def annotate(recs: list[Recommendation], picks: list[ProPick]) -> int:
    """Stamp each recommendation with VSiN's pick on the same bet.

    Matching is deliberately looser than a selection-string join, because the
    question is who is on which *side*, not who wrote the same line. An Under
    9.5 against our Under 9 is the same opinion about the same game and earns a
    star; an Over against it earns the cross. A pick on a bet we did not price,
    or a market VSiN labels in a way we do not recognise, matches nothing.
    """
    by_key: dict[str, ProPick] = {}
    for pick in picks:
        if not pick.market:
            continue
        key = (
            _game_key(pick.market, pick.matchup)
            if pick.matchup
            else _prop_key(pick.market, pick.subject)
        )
        by_key.setdefault(key, pick)
    hits = 0
    for rec in recs:
        player = _rec_player(rec)
        key = (
            _prop_key(rec.market, player)
            if player
            else _game_key(rec.market, rec.matchup)
        )
        theirs = by_key.get(key)
        if theirs is None:
            continue
        rec.vsin_pick = theirs.summary
        rec.vsin_edge = theirs.edge
        rec.vsin_agrees = _agrees(rec, theirs)
        hits += 1
    return hits


def _rec_player(rec: Recommendation) -> str:
    """The player a prop recommendation is about, or "" for a game market."""
    if not rec.market.startswith(("batter_", "pitcher_")):
        return ""
    # "Michael Busch TB o1.5" -> "Michael Busch": the stat symbol and the side
    # are the last two tokens the key builder appends.
    return " ".join(rec.selection.split()[:-2])


def _agrees(rec: Recommendation, pick: ProPick) -> bool | None:
    """Whether VSiN is on our side of this bet. ``None`` when it is unclear.

    Over/under is compared as a direction, since two models can bet the same
    total at different numbers. Team sides are compared as teams for the same
    reason: a run line taken at -1.5 and a moneyline are not the same bet, but
    they are the same team, and the join only ever pairs like markets.
    """
    if pick.side in ("over", "under"):
        if rec.side in ("over", "under"):
            return pick.side == rec.side
        return None
    head = rec.selection.split()[0] if rec.selection.split() else ""
    return pick.side == head if head else None


def save_picks(path: Path, picks: list[ProPick]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in picks], indent=1), encoding="utf-8")


def load_picks(path: Path) -> list[ProPick]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    out: list[ProPick] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(ProPick(**row))
        except TypeError:
            log.debug("skipping propick row with unexpected fields")
    return out


def merge_picks(old: list[ProPick], new: list[ProPick]) -> list[ProPick]:
    """Union two captures of a day, keyed on the pick itself. Later wins."""
    merged: dict[str, ProPick] = {p.key: p for p in old}
    for pick in new:
        merged[pick.key] = pick
    return [merged[k] for k in sorted(merged)]
