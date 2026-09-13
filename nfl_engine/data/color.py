"""ESPN's public NFL game summary, read for the weekly card's colour.

``site.api.espn.com`` serves, without a key, the week's scoreboard and a per-game
summary carrying the AP preview, each side's season stat leaders, the injury
report, the last five results, ATS records, the venue and the kickoff forecast.
None of it is priced: it is the context a reader wants beside the number, shown
only when ESPN actually has it for that game, and it moves no probability, no
screen and no tier. FPI, ESPN's own forecast, is read elsewhere as a benchmark;
here it is quoted as one more thing the reader may want to know.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from cfb_engine.data.espn import _clean, _lede, _num, _rows
from mlb_engine.data import http
from nfl_engine.data.teamnames import canonical

log = logging.getLogger(__name__)

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    "?dates={season}&seasontype=2&week={week}"
)
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={event_id}"
# ESPN's site API answers a browser-shaped agent and refuses the engine's default one.
_UA = "Mozilla/5.0 (X11; Linux x86_64)"

# Words in a preview that mark the game as more than a Sunday: printed as tags.
_STORY_TAGS: list[tuple[str, re.Pattern[str]]] = [
    ("rivalry", re.compile(r"\brival(?:ry|s)?\b", re.I)),
    (
        "division",
        re.compile(
            r"\bdivision(?:al)? (?:game|rival|matchup|opener|title|lead)\b|\b(?:AFC|NFC) (?:East|West|North|South)\b",
            re.I,
        ),
    ),
    ("revenge", re.compile(r"\brevenge\b|\bavenge\b|\bpayback\b|\brematch\b", re.I)),
    (
        "must-win",
        re.compile(
            r"\bmust[- ]win\b|\bseason on the line\b|\bplayoff (?:hopes|picture|spot|race|berth)\b|\belimination\b",
            re.I,
        ),
    ),
    (
        "qb change",
        re.compile(
            r"\bbackup quarterback\b|\bfirst (?:career |NFL )?start\b|\bmakes his first start\b|\bnew starting quarterback\b",
            re.I,
        ),
    ),
    (
        "streak",
        re.compile(
            r"\b(?:winning|losing|win|loss) streak\b|\bstraight (?:wins|losses|games)\b|\bunbeaten\b|\bwinless\b",
            re.I,
        ),
    ),
    (
        "homecoming",
        re.compile(
            r"\breturn(?:s|ing)? to\b.{0,40}\b(?:former|old) team\b|\bfaces? his (?:former|old) team\b",
            re.I,
        ),
    ),
    (
        "prime time",
        re.compile(r"\bMonday night\b|\bThursday night\b|\bSunday night\b|\bprime[- ]time\b", re.I),
    ),
]
_LEADER_CATS = ("passingYards", "rushingYards", "receivingYards")
# ESPN status strings for a player who will not play; "Questionable" is not one.
_OUT_STATUSES = ("out", "doubtful", "injured reserve", "suspension", "physically unable")


def _tags(*texts: str | None) -> list[str]:
    blob = " ".join(t for t in texts if t)
    return [tag for tag, rx in _STORY_TAGS if rx.search(blob)]


@dataclass
class TeamColor:
    record: str | None = None  # "3-1"
    ats: str | None = None  # "2-2-0"
    leaders: list[str] = field(default_factory=list)  # "QB Jared Goff — 1,254 YDS, 9 TD"
    last_five: list[str] = field(default_factory=list)  # "W 34-26 @ TEN", newest first
    out: list[str] = field(default_factory=list)  # "RB Isiah Pacheco (IR, back)"
    questionable: list[str] = field(default_factory=list)  # "RB Alvin Kamara (knee)"

    def streak(self) -> str:
        """``W3`` / ``L2`` off the last five, or ``""`` when nothing has been played."""
        letters = [g[0] for g in self.last_five if g and g[0] in "WLT"]
        if not letters:
            return ""
        first = letters[0]
        run = 0
        for letter in letters:
            if letter != first:
                break
            run += 1
        return f"{first}{run}"


@dataclass
class GameColor:
    """What ESPN says about one game, all optional."""

    home: TeamColor = field(default_factory=TeamColor)
    away: TeamColor = field(default_factory=TeamColor)
    venue: str | None = None
    city: str | None = None
    grass: bool | None = None
    indoor: bool | None = None
    neutral_site: bool = False
    broadcast: str | None = None
    headline: str | None = None
    story: str | None = None  # first two sentences of the AP preview
    tags: list[str] = field(default_factory=list)
    fpi_home: float | None = None  # ESPN matchup predictor, home win %
    condition: str | None = None
    temperature_f: float | None = None
    precip_pct: float | None = None
    gust_mph: float | None = None
    spread_note: str | None = None  # ESPN's posted line, e.g. "DET -7 / 49.5"


ColorBook = dict[str, GameColor]


def _matchup(home: str, away: str) -> str:
    return f"{canonical(away)} @ {canonical(home)}"


def _abbrev(team: object) -> str | None:
    if not isinstance(team, dict):
        return None
    code = team.get("abbreviation")
    return canonical(code) if isinstance(code, str) and code else None


def _status_line(entry: dict[str, object]) -> tuple[str, bool] | None:
    athlete = entry.get("athlete")
    if not isinstance(athlete, dict) or not isinstance(athlete.get("displayName"), str):
        return None
    pos = athlete.get("position")
    pos_abbr = pos.get("abbreviation") if isinstance(pos, dict) else None
    status = str(entry.get("status") or "")
    details = entry.get("details")
    injury = details.get("type") if isinstance(details, dict) else None
    label = f"{pos_abbr + ' ' if isinstance(pos_abbr, str) else ''}{athlete['displayName']}"
    unavailable = any(word in status.lower() for word in _OUT_STATUSES)
    short = "IR" if "reserve" in status.lower() else status
    note = ", ".join(
        x
        for x in ((short if unavailable else None), (str(injury).lower() if injury else None))
        if x
    )
    return (f"{label} ({note})" if note else label), unavailable


def parse_summary(payload: dict[str, object], home: str, away: str) -> GameColor:
    """Pure parse of one ``/summary`` payload for the game ``away @ home``."""
    out = GameColor()
    hk, ak = canonical(home), canonical(away)

    def side(team: object) -> TeamColor | None:
        code = _abbrev(team)
        return out.home if code == hk else out.away if code == ak else None

    header = payload.get("header")
    comps = header.get("competitions") if isinstance(header, dict) else None
    comp = comps[0] if isinstance(comps, list) and comps and isinstance(comps[0], dict) else {}
    out.neutral_site = bool(comp.get("neutralSite", False))
    notes: list[str] = []
    for n in comp.get("notes") or []:
        if isinstance(n, dict) and isinstance(n.get("headline"), str):
            notes.append(n["headline"])
    for c in comp.get("competitors") or []:
        if not isinstance(c, dict):
            continue
        tc = side(c.get("team"))
        if tc is None:
            continue
        for rec in c.get("record") or []:
            if isinstance(rec, dict) and rec.get("type") == "total":
                tc.record = str(rec.get("displayValue") or rec.get("summary") or "") or None
    for b in comp.get("broadcasts") or []:
        media = b.get("media") if isinstance(b, dict) else None
        name = media.get("shortName") if isinstance(media, dict) else None
        if isinstance(name, str) and name:
            out.broadcast = name
            break

    for row in _rows(payload.get("againstTheSpread")):
        if not isinstance(row, dict):
            continue
        tc = side(row.get("team"))
        if tc is None:
            continue
        for rec in row.get("records") or []:
            if isinstance(rec, dict) and isinstance(rec.get("displayValue"), str):
                tc.ats = rec["displayValue"]
                break

    for row in _rows(payload.get("leaders")):
        if not isinstance(row, dict):
            continue
        tc = side(row.get("team"))
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
                tc.leaders.append(
                    f"{pos_abbr + ' ' if isinstance(pos_abbr, str) else ''}{name} — {stat}"
                )

    for row in _rows(payload.get("injuries")):
        if not isinstance(row, dict):
            continue
        tc = side(row.get("team"))
        if tc is None:
            continue
        for entry in row.get("injuries") or []:
            if not isinstance(entry, dict):
                continue
            parsed = _status_line(entry)
            if parsed is None:
                continue
            text, unavailable = parsed
            (tc.out if unavailable else tc.questionable).append(text)

    for row in _rows(payload.get("lastFiveGames")):
        if not isinstance(row, dict):
            continue
        tc = side(row.get("team"))
        if tc is None:
            continue
        for ev in row.get("events") or []:
            if not isinstance(ev, dict):
                continue
            result, score = ev.get("gameResult"), ev.get("score")
            opp = _abbrev(ev.get("opponent"))
            at = "@" if ev.get("atVs") == "@" else "vs"
            if isinstance(result, str) and isinstance(score, str) and opp:
                tc.last_five.append(f"{result} {score} {at} {opp}")

    info = payload.get("gameInfo")
    if isinstance(info, dict):
        venue = info.get("venue")
        if isinstance(venue, dict):
            out.venue = venue.get("fullName") if isinstance(venue.get("fullName"), str) else None
            grass = venue.get("grass")
            out.grass = grass if isinstance(grass, bool) else None
            indoor = venue.get("indoor")
            out.indoor = indoor if isinstance(indoor, bool) else None
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
            cond = wx.get("displayValue")
            out.condition = cond if isinstance(cond, str) and cond else None

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

    for pick in _rows(payload.get("pickcenter")):
        if not isinstance(pick, dict):
            continue
        details, total = pick.get("details"), _num(pick.get("overUnder"))
        if isinstance(details, str) and details:
            out.spread_note = f"{details} / {total:g}" if total is not None else details
            break

    out.tags = _tags(out.headline, out.story, *notes)
    return out


class ESPNColor:
    """Fetch and cache ESPN summaries for a week."""

    def __init__(
        self, cache_dir: Path | None = None, *, ttl: int = 3 * 3600, timeout: float = 20.0
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.timeout = timeout

    def _get(self, key: str, url: str) -> object:
        path = self.cache_dir / f"espn_{key}.json" if self.cache_dir else None
        if path is not None and path.exists() and time.time() - path.stat().st_mtime < self.ttl:
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

    def events(self, season: int, week: int) -> dict[str, str]:
        """ESPN event id per ``AWAY @ HOME`` matchup in the week."""
        data = self._get(f"nfl_sb_{season}_{week}", SCOREBOARD_URL.format(season=season, week=week))
        out: dict[str, str] = {}
        if not isinstance(data, dict):
            return out
        for ev in data.get("events") or []:
            if not isinstance(ev, dict) or not isinstance(ev.get("id"), str):
                continue
            comps = ev.get("competitions")
            comp = (
                comps[0]
                if isinstance(comps, list) and comps and isinstance(comps[0], dict)
                else None
            )
            if comp is None:
                continue
            home = away = None
            for c in _rows(comp.get("competitors")):
                if not isinstance(c, dict):
                    continue
                code = _abbrev(c.get("team"))
                if c.get("homeAway") == "home":
                    home = code
                elif c.get("homeAway") == "away":
                    away = code
            if home and away:
                out[_matchup(home, away)] = ev["id"]
        return out

    def fetch(self, season: int, week: int, matchups: list[str]) -> ColorBook:
        """:class:`GameColor` for each ``AWAY @ HOME`` matchup ESPN lists in the week."""
        ids = self.events(season, week)
        if not ids:
            return {}
        book: ColorBook = {}
        for matchup in matchups:
            event_id = ids.get(matchup)
            if event_id is None:
                continue
            away, _, home = matchup.partition(" @ ")
            data = self._get(f"nfl_sum_{event_id}", SUMMARY_URL.format(event_id=event_id))
            if isinstance(data, dict):
                book[matchup] = parse_summary(data, home.strip(), away.strip())
        log.info("espn colour: %d/%d games", len(book), len(matchups))
        return book
