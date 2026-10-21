"""Doc's Sports handicappers' free plays, read off their capper pages into the store.

Each Doc's handicapper page carries a "Daily Free Picks" box with that capper's
complimentary play for the day: the pick, the game's start time, the expert and
their write-up. It is up before the game and gone after, so it is read at each
podcast pass and filed like a VSiN Pro pick, the capper as the host: printed
beside the card, graded in the league's ledger, never in a price, gate or stake.

Only full-game markets are kept (spread, moneyline, total); first halves, team
totals, props and parlays are not something the cards place. A play already
started when read is dropped: nothing is filed after the fact. The sellers'
own "documented" results tables are not read; they carry no release time.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from engine_common.podcasts.extract import PICKS_DIR, Extraction, Pick
from engine_common.podcasts.shows import (
    CBB,
    CFB,
    DOCS_NAME,
    DOCS_SHOW,
    LEAGUE_WORDS,
    MLB,
    NBA,
    NFL,
    NHL,
)

PARSER_VERSION = "docs-1"
CAPPER_URL = "https://www.docsports.com/cappers.html?cap_id={cap_id}"
# The handicappers whose free play is read: Doc's capper id -> name as it prints.
CAPPERS: dict[int, str] = {36: "Scott Rickenbach"}
EASTERN = ZoneInfo("America/New_York")
TIMEOUT = 20

LEAGUE_OF = {
    "NFL": NFL,
    "CFB": CFB,
    "NCAAF": CFB,
    "COLLEGE FOOTBALL": CFB,
    "MLB": MLB,
    "NBA": NBA,
    "CBB": CBB,
    "NCAAB": CBB,
    "COLLEGE BASKETBALL": CBB,
    "NHL": NHL,
}

_BOX = re.compile(r'<div class="featured-pick">(.*?)</table>', re.S)
_HEAD = re.compile(r'white-transparent-dougan">(.*?)</span>', re.S)
_CELL = re.compile(
    r"<td[^>]*>\s*<span[^>]*>\s*(Pick|Start|Expert):\s*</span>\s*</td>\s*<td[^>]*>(.*?)</td>", re.S
)
_ANALYSIS = re.compile(r'<span class="analysis-content">(.*?)</span>', re.S)
_TAGS = re.compile(r"<[^>]+>")
_GAME = re.compile(
    r"^(?:(?P<league>[A-Z]{3,5}):\s*)?(?P<away>.+?)\s+(?:at|@|vs\.?)\s+(?P<home>.+?)$", re.I
)
_PARTIAL = re.compile(
    r"\b(?:[12]H|[1-4]Q|[1-3]P|F5|(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+"
    r"(?:half|quarter|period|inning|five|5)|team total|parlay|teaser|prop)\b",
    re.I,
)
_TOTAL = re.compile(
    r"^(?P<side>over|under)\s+(?P<num>\d+(?:\.5)?)\s*(?:\(?(?P<price>[+-]\d{3,4})\)?)?$", re.I
)
_ML = re.compile(r"^(?P<team>.+?)\s+(?:ML|money ?line|\((?P<price>[+-]\d{3,4})\))$", re.I)
_ATS = re.compile(
    r"^(?P<team>.+?)\s+(?P<num>[+-]\d+(?:\.5)?|PK|pick'?em)\s*(?:\(?(?P<price>[+-]\d{3,4})\)?)?$",
    re.I,
)


@dataclass(frozen=True)
class Play:
    cap_id: int
    expert: str
    head: str  # "NFL: Patriots at Bears"
    pick: str  # "Patriots +2.5"
    start: datetime  # UTC
    analysis: str


@dataclass(frozen=True)
class Parsed:
    market: str
    team: str
    opponent: str
    side: str | None
    line: float | None
    price: float | None
    description: str


def _text(html: str) -> str:
    text = _TAGS.sub(" ", html)
    for a, b in (
        ("&rsquo;", "'"),
        ("&amp;", "&"),
        ("&ndash;", "-"),
        ("&nbsp;", " "),
        ("&#39;", "'"),
    ):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def _start(text: str) -> datetime | None:
    try:
        local = datetime.strptime(text, "%B %d, %Y @ %I:%M %p")
    except ValueError:
        return None
    return local.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def plays_in(html: str, cap_id: int) -> list[Play]:
    """The free-play boxes on a capper page that hold a game pick (promos skipped)."""
    out: list[Play] = []
    for box in _BOX.finditer(html):
        cells = {k: _text(v) for k, v in _CELL.findall(box.group(1))}
        head_m = _HEAD.search(box.group(1))
        if not head_m or not {"Pick", "Start", "Expert"} <= cells.keys():
            continue
        start = _start(cells["Start"])
        if start is None:
            continue
        an = _ANALYSIS.search(box.group(1)) or _ANALYSIS.search(html[box.end() : box.end() + 20000])
        out.append(
            Play(
                cap_id,
                cells["Expert"],
                _text(head_m.group(1)),
                cells["Pick"],
                start,
                _text(an.group(1)) if an else "",
            )
        )
    return out


def _league(play: Play) -> str | None:
    m = _GAME.match(play.head)
    if m and m.group("league"):
        tag = LEAGUE_OF.get(m.group("league").upper())
        if tag:
            return tag
    text = f"{play.head} {play.analysis[:400]}"
    hits = [lg for lg, pat in LEAGUE_WORDS.items() if pat.search(text)]
    return hits[0] if len(hits) == 1 else None


def parse_play(play: Play) -> Parsed | None:
    """The play as a pick's fields; ``None`` for a half, prop, parlay or unreadable text."""
    pick = re.sub(r"\s+", " ", play.pick).strip()
    if _PARTIAL.search(pick) or _PARTIAL.search(play.head):
        return None
    m = _GAME.match(play.head)
    if m is None:
        return None
    away, home = m.group("away").strip(), m.group("home").strip()

    def other(team: str) -> str:
        return home if team.lower() == away.lower() else away

    tot = _TOTAL.match(pick)
    if tot:
        price = float(tot.group("price")) if tot.group("price") else None
        return Parsed(
            "game_total",
            away,
            home,
            tot.group("side").lower(),
            float(tot.group("num")),
            price,
            pick,
        )
    ml = _ML.match(pick)
    if ml:
        team = ml.group("team").strip()
        price = float(ml.group("price")) if ml.group("price") else None
        return Parsed("game_ml", team, other(team), None, None, price, pick)
    ats = _ATS.match(pick)
    if ats:
        team = ats.group("team").strip()
        tok = ats.group("num").upper()
        line = 0.0 if tok.startswith(("PK", "PICK")) else float(tok)
        price = float(ats.group("price")) if ats.group("price") else None
        if abs(line) >= 100 and price is None:  # "Phillies +140" is a moneyline price, not a spread
            return Parsed("game_ml", team, other(team), None, None, line, pick)
        return Parsed("game_ats", team, other(team), None, line, price, pick)
    return None


def pick_of(play: Play, read_at: datetime) -> Pick | None:
    """The play as a stored pick, or ``None`` if unreadable, in no league, or already started."""
    if read_at >= play.start:
        return None
    league = _league(play)
    parsed = parse_play(play)
    if league is None or parsed is None:
        return None
    raw = "|".join(
        (DOCS_SHOW, str(play.cap_id), play.expert, parsed.description, play.start.isoformat())
    )
    pid = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return Pick(
        pick_id=pid,
        show=DOCS_SHOW,
        show_name=DOCS_NAME,
        host=play.expert,
        episode_guid=f"docs-{pid}",
        episode_title=f"{play.expert} free play {play.start.date()}",
        published=read_at.isoformat(),
        kind="official",
        market=parsed.market,
        team=parsed.team,
        opponent=parsed.opponent,
        side=parsed.side,
        line=parsed.line,
        price=parsed.price,
        units=None,
        seconds=0,
        quote=play.analysis[:300],
        reason=f"{play.head}, {play.start.astimezone(EASTERN):%Y-%m-%d %I:%M %p ET}",
        description=parsed.description,
        league=league,
        edge="",
    )


def fetch_page(cap_id: int) -> str:
    resp = requests.get(
        CAPPER_URL.format(cap_id=cap_id), headers={"User-Agent": "Mozilla/5.0"}, timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.text


def extraction_path(root: Path, pick: Pick) -> Path:
    return root / PICKS_DIR / f"{DOCS_SHOW}__{pick.pick_id}.json"


def read_docs(
    root: Path,
    read_at: datetime | None = None,
    leagues: Iterable[str] | None = None,
    *,
    pages: dict[int, str] | None = None,
) -> list[Extraction]:
    """Store each capper's current free play; the same play read again is one file.

    ``read_at`` is the moment the page was read (now); a play whose game has
    started by then is not filed, whatever window the pass covers.
    """
    read_at = read_at or datetime.now(timezone.utc)
    wanted = set(leagues) if leagues is not None else None
    kept: list[Extraction] = []
    for cap_id in CAPPERS:
        html = pages[cap_id] if pages is not None else fetch_page(cap_id)
        for play in plays_in(html, cap_id):
            pick = pick_of(play, read_at)
            if pick is None or (wanted is not None and pick.league not in wanted):
                continue
            path = extraction_path(root, pick)
            if path.exists():
                continue  # first read keeps its release time
            sha = hashlib.sha1(f"{play.pick}|{play.start}".encode()).hexdigest()
            ex = Extraction(
                pick.episode_guid, "rules", PARSER_VERSION, sha, [pick], [], [pick.league]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ex.to_json())
            kept.append(ex)
    return kept


__all__ = [
    "CAPPERS",
    "PARSER_VERSION",
    "Parsed",
    "Play",
    "parse_play",
    "pick_of",
    "plays_in",
    "read_docs",
]
