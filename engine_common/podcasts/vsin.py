"""VSiN's written best bets, read off its articles into the podcast store.

VSiN's writers sign their bets in a fixed phrase ("College Football Best Bet:
Nebraska +7.5", "Bet: Under 53.5 (-108)", "Pick: White Sox +114", or a list
after "Here are my Week 6 best bets:"). Each article becomes one
:class:`Extraction` under ``picks/`` with the writer as the host, so every
engine places, prints, grades and audits these exactly like a podcast pick:
beside the card, in their own ledger, never in a price, gate or stake.

The articles come from VSiN's public WordPress API; no login is needed. A line
is kept only as written: a pick with no number the writer gave is printed and
never graded at a number of ours.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from engine_common.podcasts.extract import PICKS_DIR, Extraction, Pick
from engine_common.podcasts.shows import CBB, CFB, MLB, NBA, NFL, NHL, VSIN_NAME, VSIN_SHOW

API = "https://vsin.com/wp-json/wp/v2"
SHOW_KEY = VSIN_SHOW
SHOW_NAME = VSIN_NAME
PARSER_VERSION = "vsin-rules-1"
CATEGORY = {CFB: 19, NFL: 15, NHL: 18, MLB: 17, CBB: 20, NBA: 16}
_HEADERS = {"User-Agent": "Mozilla/5.0 (payoff-pitch)"}
_PER_PAGE = 50
_PAGES = 4

_LEAGUE_WORD = r"(?:college football|college basketball|cfb|cbb|nfl|nba|nhl|mlb)"
_LABEL = re.compile(
    rf"^(?:{_LEAGUE_WORD}\s+)?(?:{_LEAGUE_WORD}\s+predictions and\s+)?"
    r"(?:best bets?|bets?|picks?)\s*:\s*(\S.*)$",
    re.I,
)
_INTRO = re.compile(r"^here are (?:my|the|tonight'?s)\b.*\bbest bets\b", re.I)
_OUTRO = re.compile(r"^(?:for more|more:|>>)", re.I)
_FILLER = re.compile(
    r"^(?:(?:give me|i'?ll|i will|let'?s|it'?s(?: got to be)?|i like|i'?m (?:on|taking)|"
    r"take|back|go(?: with)?|lay(?: the [\d.]+(?: points)?(?: with)?)?|the)\s+)+",
    re.I,
)
_TOTAL = re.compile(r"\b(over|under)\s+(\d+(?:\.\d+)?)\b", re.I)
_ML = re.compile(r"^(?P<team>.+?)\s+(?:ml|moneyline)\b", re.I)
_NUM = re.compile(r"(?<![\w.(])(?P<num>[+-]\d+(?:\.\d+)?|pk|pick'?em)(?![\w.])", re.I)
_PRICE = re.compile(r"\(\s*([+-]\d{3,4})\b|(?<![\w.])([+-]\d{3,4})(?![\w.])")
_UNITS = re.compile(r"([\d.]+|[¼½¾])\s*units?\b", re.I)
_FRACTION = {"¼": 0.25, "½": 0.5, "¾": 0.75}
_NOT_GAME = re.compile(
    r"\b(teasers?|parlay|1h|2h|first half|1st half|quarter|period|inning|f5|team total|tt|"
    r"in regulation|shots?|goals?|assists?|points?|strikeouts?|walks?|hits?|yards?|"
    r"touchdowns?|tds?|passing|rushing|receiving|saves|sog|anytime|rebounds?|props?|series|"
    r"games|to win|futures?|alt)\b",
    re.I,
)
_GAME_SEP = re.compile(r"\s+(?:at|vs\.?|versus|@)\s+|/|-(?=[A-Z])", re.I)
_SPLIT = re.compile(
    r"(?<=units\))\s*&?\s*|(?<=unit\))\s*&?\s*|(?<=units)\s+(?=\S)|(?<=unit)\s+(?=\S)|"
    r"(?<=point)\s+(?=[A-Z])|\s*;\s*|\s+&\s+"
)


@dataclass(frozen=True)
class Post:
    id: int
    published: str  # ISO8601 UTC
    link: str
    title: str
    author: str
    league: str
    content: str  # rendered HTML


def _clean(text: str) -> str:
    text = html.unescape(text).replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _teams(heading: str) -> tuple[str, str] | None:
    """The two sides a game heading names ("UCLA (+11.5) at Oregon - 7:30 pm ET")."""
    text = re.sub(r"\([^)]*\)", " ", heading)
    text = re.sub(r"\s-\s.*$|\bprediction.*$|\bgame \d.*$", "", text, flags=re.I)
    parts = [p.strip(" .:") for p in _GAME_SEP.split(text) if p and p.strip(" .:")]
    if len(parts) != 2 or any(len(p.split()) > 4 for p in parts):
        return None
    return parts[0], parts[1]


_NOT_NAME = re.compile(r"^(?:pick|prediction|predictions|preview|tnf|snf|mnf|not|i|it)$", re.I)


def _team_name(text: str) -> str | None:
    """The run of capitalised words a bet names ("the Eagles getting" -> "Eagles")."""
    words = _FILLER.sub("", text).strip(" ,.:").split()
    while words and not words[0][0].isupper():
        words.pop(0)
    name: list[str] = []
    for w in words:
        if not (w[0].isupper() or w[0].isdigit() and name) or _NOT_NAME.match(w):
            break
        name.append(w.strip(",.:'s") if w.endswith("'s") else w.strip(",.:"))
    if not name or len(name) > 4:
        return None
    return " ".join(name)


_VERSUS = re.compile(
    r"\b(?:at|vs\.?|versus|against|over|beat)\s+(?:the\s+)?(?P<opp>[A-Z][\w&.']*(?:\s+[A-Z][\w&.']*){0,3})"
)


def _named_opponent(rest: str) -> str | None:
    m = _VERSUS.search(rest)
    return _team_name(m.group("opp")) if m else None


def _units(text: str) -> float | None:
    m = _UNITS.search(text)
    if m is None:
        return None
    tok = m.group(1)
    if tok in _FRACTION:
        return _FRACTION[tok]
    try:
        return float(tok)
    except ValueError:
        return None


def _price(text: str) -> float | None:
    for m in _PRICE.finditer(text):
        return float(m.group(1) or m.group(2))
    return None


@dataclass(frozen=True)
class Parsed:
    market: str
    team: str | None
    opponent: str | None
    side: str | None
    line: float | None
    price: float | None
    units: float | None
    description: str


def parse_bet(text: str, game: tuple[str, str] | None) -> Parsed | None:
    """One written bet as a podcast pick's fields; ``None`` for a pass."""
    text = _clean(text)
    if not text or re.match(r"^pass\b", text, re.I):
        return None
    desc = re.sub(r"\s*\((?:pool play|\d+-\d+).*$", "", text, flags=re.I).strip()
    price, units = _price(desc), _units(desc)
    body = re.sub(r"\([^)]*\)", " ", desc)
    body = re.sub(r"[-\s]*[\d.¼½¾]+\s*units?\b", " ", body, flags=re.I).strip()
    other = Parsed("other", None, None, None, None, price, units, desc)
    if _NOT_GAME.search(body):
        return other
    opp: str | None = None
    tot = _TOTAL.search(body)
    if tot:
        before = body[: tot.start()].strip(" ,")
        after = body[tot.end() :].strip()
        pair = _teams(before) if before else None
        if pair is None and after:
            pair = _teams(re.sub(r"^(?:in|for)\s+", "", after, flags=re.I).split(" on ")[0])
        if pair is None and not _FILLER.sub("", before).strip():
            pair = game
        names = (_team_name(pair[0]), _team_name(pair[1])) if pair else (None, None)
        if names[0] is None or names[1] is None:
            return other
        return Parsed(
            "game_total",
            names[0],
            names[1],
            tot.group(1).lower(),
            float(tot.group(2)),
            price,
            units,
            desc,
        )
    ml = _ML.match(body)
    if ml:
        team = _team_name(ml.group("team"))
        if team is None:
            return other
        opp = _named_opponent(body[ml.end() :]) or (_opponent(team, game) if game else None)
        return Parsed("game_ml", team, opp, None, None, price, units, desc)
    num = _NUM.search(body)
    if num is None:
        return other
    team = _team_name(body[: num.start()])
    if team is None:
        return other
    opp = _named_opponent(body[num.end() :]) or (_opponent(team, game) if game else None)
    tok = num.group("num").lower()
    value = 0.0 if tok.startswith("p") else float(tok)
    if abs(value) >= 100:
        return Parsed("game_ml", team, opp, None, None, value, units, desc)
    return Parsed("game_ats", team, opp, None, value, price, units, desc)


def _opponent(team: str, game: tuple[str, str]) -> str | None:
    t = team.lower()
    a, b = game
    if t in a.lower() or a.lower() in t:
        return _team_name(b)
    if t in b.lower() or b.lower() in t:
        return _team_name(a)
    return None


def _items(text: str) -> list[str]:
    return [s.strip(" -,") for s in _SPLIT.split(text) if s and s.strip(" -,")]


def bets_in(content: str) -> Iterator[tuple[str, tuple[str, str] | None]]:
    """Each written bet in an article, with the game heading it sits under."""
    soup = BeautifulSoup(content, "html.parser")
    game: tuple[str, str] | None = None
    listing = False
    for el in soup.find_all(["h2", "h3", "h4", "p", "li"]):
        text = _clean(el.get_text(" ", strip=True))
        if not text:
            continue
        if el.name in ("h2", "h3", "h4"):
            # A writer who lists bets as headings ("North Carolina +4 at Pittsburgh")
            # does so after an intro; a game heading puts its number in brackets.
            if listing and "(" not in text:
                parsed = parse_bet(text, None)
                if parsed is not None and parsed.market != "other":
                    yield text, None
                    continue
            game = _teams(text) or game
            continue
        if _INTRO.search(text):
            listing = True
        elif _OUTRO.search(text):
            listing = False
        m = _LABEL.match(text)
        if m:
            for item in _items(m.group(1)):
                yield item, game


def picks_in(post: Post) -> list[Pick]:
    out: dict[str, Pick] = {}
    for n, (item, game) in enumerate(bets_in(post.content)):
        parsed = parse_bet(item, game)
        if parsed is not None:
            raw = "|".join(
                (
                    SHOW_KEY,
                    str(post.id),
                    post.author,
                    parsed.market,
                    parsed.description,
                    post.league,
                )
            )
            pid = hashlib.sha1(raw.encode()).hexdigest()[:12]
            out.setdefault(
                pid,
                Pick(
                    pick_id=pid,
                    show=SHOW_KEY,
                    show_name=SHOW_NAME,
                    host=post.author or None,
                    episode_guid=f"vsin-{post.id}",
                    episode_title=post.title,
                    published=post.published,
                    kind="official",
                    market=parsed.market,
                    team=parsed.team,
                    opponent=parsed.opponent,
                    side=parsed.side,
                    line=parsed.line,
                    price=parsed.price,
                    units=parsed.units,
                    seconds=n,
                    quote=item[:300],
                    reason=post.link,
                    description=parsed.description,
                    league=post.league,
                    edge="",
                ),
            )
    return list(out.values())


def extraction(post: Post) -> Extraction:
    sha = hashlib.sha1(post.content.encode()).hexdigest()
    return Extraction(
        f"vsin-{post.id}", "rules", PARSER_VERSION, sha, picks_in(post), [], [post.league]
    )


def extraction_path(root: Path, post: Post) -> Path:
    return root / PICKS_DIR / f"{SHOW_KEY}__{post.league}_{post.id}.json"


def _utc(raw: str) -> str:
    return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc).isoformat()


def fetch_posts(
    league: str,
    since: datetime,
    until: datetime,
    *,
    session: requests.Session | None = None,
) -> list[Post]:
    """The league's VSiN articles published in ``[since, until)``."""
    http = session or requests.Session()
    authors: dict[int, str] = {}
    posts: list[Post] = []
    after = (since - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    before = (until + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    for page in range(1, _PAGES + 1):
        params: dict[str, str | int] = {
            "categories": CATEGORY[league],
            "after": after,
            "before": before,
            "per_page": _PER_PAGE,
            "page": page,
            "_fields": "id,date_gmt,link,title,author,content",
        }
        resp = http.get(
            f"{API}/posts",
            params=params,
            headers=_HEADERS,
            timeout=30,
        )
        if resp.status_code == 400 and page > 1:
            break
        resp.raise_for_status()
        rows = resp.json()
        for r in rows:
            published = _utc(r["date_gmt"])
            if not since <= datetime.fromisoformat(published) < until:
                continue
            aid = int(r["author"])
            if aid not in authors:
                authors[aid] = _author(http, aid)
            posts.append(
                Post(
                    int(r["id"]),
                    published,
                    r["link"],
                    _clean(r["title"]["rendered"]),
                    authors[aid],
                    league,
                    r["content"]["rendered"],
                )
            )
        if len(rows) < _PER_PAGE:
            break
    return posts


def _author(http: requests.Session, aid: int) -> str:
    try:
        resp = http.get(
            f"{API}/users/{aid}", params={"_fields": "name"}, headers=_HEADERS, timeout=30
        )
        resp.raise_for_status()
        return _clean(str(resp.json().get("name", "")))
    except (requests.RequestException, ValueError):
        return ""


def read_articles(
    root: Path,
    since: datetime,
    until: datetime,
    leagues: Iterable[str],
    *,
    session: requests.Session | None = None,
) -> list[Extraction]:
    """Store each article's best bets; an article re-read with new text replaces its file."""
    kept: list[Extraction] = []
    for league in leagues:
        if league not in CATEGORY:
            continue
        for post in fetch_posts(league, since, until, session=session):
            ex = extraction(post)
            if not ex.picks:
                continue
            path = extraction_path(root, post)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ex.to_json())
            kept.append(ex)
    return kept


__all__ = [
    "CATEGORY",
    "PARSER_VERSION",
    "SHOW_KEY",
    "SHOW_NAME",
    "Parsed",
    "Post",
    "bets_in",
    "extraction",
    "fetch_posts",
    "parse_bet",
    "picks_in",
    "read_articles",
]
