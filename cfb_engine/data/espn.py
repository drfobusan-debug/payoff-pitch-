"""ESPN's public game summary, read for the slate article's colour.

``site.api.espn.com`` serves, without a key, the scoreboard for a date and a
per-game summary that carries the AP preview story, each team's season stat
leaders, ATS records, the venue and the kickoff forecast. None of it is priced;
it is the context a reader wants beside the number, and it is only ever printed
when ESPN actually has it for that game.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import requests

from cfb_engine.data.teamnames import school_key
from mlb_engine.data import http

log = logging.getLogger(__name__)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
    "?dates={ymd}&groups={group}&limit=400"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
    "?event={event_id}"
)
# ESPN's site API answers a browser-shaped agent and refuses the engine's default one.
_UA = "Mozilla/5.0 (X11; Linux x86_64)"
_GROUPS = ("80", "81")  # FBS, FCS

# Words in a preview that mark the game as more than a Saturday: printed as tags.
_STORY_TAGS: list[tuple[str, re.Pattern[str]]] = [
    ("rivalry", re.compile(r"\brival(?:ry|s)?\b|\btrophy\b|\bborder war\b|\bholy war\b", re.I)),
    ("revenge", re.compile(r"\brevenge\b|\bavenge\b|\bpayback\b", re.I)),
    ("must-win", re.compile(r"\bmust[- ]win\b|\bseason on the line\b|\bbowl eligib", re.I)),
    ("first meeting", re.compile(r"\bfirst(?:-ever)? meeting\b|\bfirst time\b.{0,40}\bmeet", re.I)),
    ("homecoming", re.compile(r"\bhomecoming\b", re.I)),
    ("streak", re.compile(r"\b(?:winning|losing|win|loss) streak\b|\bstraight (?:wins|losses)\b", re.I)),
    ("ranked", re.compile(r"\bNo\. ?\d{1,2}\b")),
    ("conference opener", re.compile(r"\bconference opener\b|\b(?:SEC|Big Ten|Big 12|ACC) opener\b", re.I)),
]
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_LEADER_CATS = ("passingYards", "rushingYards", "receivingYards")


@dataclass
class TeamColor:
    record: str | None = None  # "1-0"
    ats: str | None = None  # "1-0-0"
    leaders: list[str] = field(default_factory=list)  # "QB Marcel Reed — 21/30, 233 YDS, 2 TD"


@dataclass
class GameColor:
    """What ESPN says about one game, all optional."""

    home: TeamColor = field(default_factory=TeamColor)
    away: TeamColor = field(default_factory=TeamColor)
    venue: str | None = None
    city: str | None = None
    grass: bool | None = None
    neutral_site: bool = False
    conference_game: bool = False
    headline: str | None = None
    story: str | None = None  # first two sentences of the AP preview
    tags: list[str] = field(default_factory=list)
    fpi_home: float | None = None  # ESPN matchup predictor, home win %
    temperature_f: float | None = None
    precip_pct: float | None = None
    gust_mph: float | None = None


ColorBook = dict[frozenset[str], GameColor]


def _pair(home: str, away: str) -> frozenset[str]:
    return frozenset({school_key(home), school_key(away)})


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", text))).strip()


def _lede(story: str, sentences: int = 2) -> str:
    text = _clean(story)
    # AP datelines: "COLLEGE STATION, Texas -- — Arizona State coach ..."
    text = re.sub(r"^[A-Z][A-Z .,'-]+(?:, [A-Za-z. ]+)?\s*(?:--|—|-)\s*(?:—\s*)?", "", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"])", text)
    return " ".join(parts[:sentences]).strip()


def _tags(*texts: str | None) -> list[str]:
    blob = " ".join(t for t in texts if t)
    return [tag for tag, rx in _STORY_TAGS if rx.search(blob)]


def _team_names(team: object) -> list[str]:
    """ESPN's school name first ("Richmond"), then the mascot form ("Richmond Spiders")."""
    if not isinstance(team, dict):
        return []
    return [v for v in (team.get("location"), team.get("displayName")) if isinstance(v, str) and v]


def _rows(x: object) -> list[object]:
    return x if isinstance(x, list) else []


def _num(x: object) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def parse_summary(payload: dict[str, object], home: str, away: str) -> GameColor:
    """Pure parse of one ``/summary`` payload for the game ``away @ home``."""
    out = GameColor()
    hk, ak = school_key(home), school_key(away)

    by_id: dict[str, TeamColor] = {}

    def side(team: object) -> TeamColor | None:
        """Home or away for an ESPN team blob; the header's id resolves mascot-only names."""
        if not isinstance(team, dict):
            return None
        tid = team.get("id")
        if isinstance(tid, str) and tid in by_id:
            return by_id[tid]
        for name in _team_names(team):
            k = school_key(name)
            tc = out.home if k == hk else out.away if k == ak else None
            if tc is not None:
                if isinstance(tid, str):
                    by_id[tid] = tc
                return tc
        return None

    header = payload.get("header")
    comps = header.get("competitions") if isinstance(header, dict) else None
    comp = comps[0] if isinstance(comps, list) and comps and isinstance(comps[0], dict) else {}
    out.neutral_site = bool(comp.get("neutralSite", False))
    out.conference_game = bool(comp.get("conferenceCompetition", False))
    notes: list[str] = []
    for n in comp.get("notes") or []:
        if isinstance(n, dict) and isinstance(n.get("headline"), str):
            notes.append(n["headline"])
    for c in comp.get("competitors") or []:
        if not isinstance(c, dict):
            continue
        team = c.get("team")
        tc = side(team)
        if tc is None:
            continue
        for rec in c.get("record") or []:
            if isinstance(rec, dict) and rec.get("type") == "total":
                tc.record = str(rec.get("displayValue") or rec.get("summary") or "") or None

    for row in _rows(payload.get("againstTheSpread")):
        if not isinstance(row, dict):
            continue
        team = row.get("team")
        tc = side(team)
        if tc is None:
            continue
        for rec in row.get("records") or []:
            if isinstance(rec, dict) and isinstance(rec.get("displayValue"), str):
                tc.ats = rec["displayValue"]
                break

    for row in _rows(payload.get("leaders")):
        if not isinstance(row, dict):
            continue
        team = row.get("team")
        tc = side(team)
        if tc is None:
            continue
        for cat in row.get("leaders") or []:
            if not isinstance(cat, dict) or cat.get("name") not in _LEADER_CATS:
                continue
            top = cat.get("leaders")
            if not isinstance(top, list) or not top or not isinstance(top[0], dict):
                continue
            ath = top[0].get("athlete")
            if not isinstance(ath, dict):
                continue
            pos = ath.get("position")
            pos_abbr = pos.get("abbreviation") if isinstance(pos, dict) else None
            name = ath.get("displayName")
            stat = top[0].get("displayValue")
            if isinstance(name, str) and isinstance(stat, str):
                tc.leaders.append(f"{pos_abbr + ' ' if isinstance(pos_abbr, str) else ''}{name} — {stat}")

    info = payload.get("gameInfo")
    if isinstance(info, dict):
        venue = info.get("venue")
        if isinstance(venue, dict):
            out.venue = venue.get("fullName") if isinstance(venue.get("fullName"), str) else None
            grass = venue.get("grass")
            out.grass = grass if isinstance(grass, bool) else None
            addr = venue.get("address")
            if isinstance(addr, dict):
                city, state = addr.get("city"), addr.get("state")
                if isinstance(city, str):
                    out.city = f"{city}, {state}" if isinstance(state, str) else city
        wx = info.get("weather")
        if isinstance(wx, dict):
            out.temperature_f = _num(wx.get("temperature"))
            out.precip_pct = _num(wx.get("precipitation"))
            out.gust_mph = _num(wx.get("gust"))

    article = payload.get("article")
    if isinstance(article, dict) and article.get("type") == "Preview":
        head = article.get("headline")
        story = article.get("story")
        out.headline = _clean(head) if isinstance(head, str) else None
        out.story = _lede(story) if isinstance(story, str) else None

    pred = payload.get("predictor")
    if isinstance(pred, dict):
        ht = pred.get("homeTeam")
        if isinstance(ht, dict):
            out.fpi_home = _num(ht.get("gameProjection"))

    out.tags = _tags(out.headline, out.story, *notes)
    return out


class ESPNColor:
    """Fetch and cache ESPN summaries for a slate date."""

    def __init__(self, cache_dir: Path | None = None, *, ttl: int = 3 * 3600, timeout: float = 20.0):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.timeout = timeout

    def _get(self, key: str, url: str) -> object:
        path = self.cache_dir / f"espn_{key}.json" if self.cache_dir else None
        if path is not None and path.exists():
            if time.time() - path.stat().st_mtime < self.ttl:
                try:
                    return json.loads(path.read_text())
                except ValueError:
                    pass
        try:
            resp = http.get(url, timeout=self.timeout, user_agent=_UA)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("espn %s: %s", key, exc)
            return None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return data

    def events(self, day: Date) -> dict[frozenset[str], str]:
        """ESPN event id per (home, away) pair kicking on ``day``."""
        ymd = day.strftime("%Y%m%d")
        out: dict[frozenset[str], str] = {}
        for group in _GROUPS:
            data = self._get(f"sb_{ymd}_{group}", SCOREBOARD_URL.format(ymd=ymd, group=group))
            if not isinstance(data, dict):
                continue
            for ev in data.get("events") or []:
                if not isinstance(ev, dict):
                    continue
                comps = ev.get("competitions")
                comp = comps[0] if isinstance(comps, list) and comps and isinstance(comps[0], dict) else None
                if comp is None:
                    continue
                names = [
                    _team_names(c.get("team") if isinstance(c, dict) else None)
                    for c in _rows(comp.get("competitors"))
                ]
                if len(names) == 2 and names[0] and names[1] and isinstance(ev.get("id"), str):
                    out[_pair(names[0][0], names[1][0])] = ev["id"]
        return out

    def fetch(self, day: Date, games: list[tuple[str, str]]) -> ColorBook:
        """:class:`GameColor` for each ``(home, away)`` ESPN lists on ``day``."""
        ids = self.events(day)
        if not ids:
            return {}
        book: ColorBook = {}
        for home, away in games:
            key = _pair(home, away)
            event_id = ids.get(key)
            if event_id is None:
                continue
            data = self._get(f"sum_{event_id}", SUMMARY_URL.format(event_id=event_id))
            if isinstance(data, dict):
                book[key] = parse_summary(data, home, away)
        log.info("espn colour: %d/%d games", len(book), len(games))
        return book


def color_for(book: ColorBook, home: str, away: str) -> GameColor | None:
    return book.get(_pair(home, away))
