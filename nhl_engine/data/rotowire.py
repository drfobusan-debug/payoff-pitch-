"""RotoWire NHL lineups page: expected goalies, PP units, injuries (§5.5, §5.7).

``https://www.rotowire.com/hockey/nhl-lineups.php`` (``?date=tomorrow`` for
the next slate) lists, per game, each team's goalie with a ``Confirmed`` or
``Expected`` dot, its two power-play units and its injury list. No full 5v5
lines are published there, so "lineup" here means goalie + PP units + injuries.

The raw page is archived under ``<data_dir>/rotowire/<slate>/<stamp>.html``
before parsing so a later audit can re-read exactly what the engine saw. The
parser is regex on the page's own CSS hooks (``lineup__abbr``,
``lineup__player-highlight``, ``lineup__title``, ``lineup__player``,
``lineup__inj``); a hook that disappears yields an empty slate, never a wrong
one.

Status mapping into the starter ladder (``features/starters.py``):

* ``Confirmed`` -> ``confirmed``
* ``Expected``  -> ``probable``

Names are resolved to NHL player ids through MoneyPuck's season summaries by
normalised full name; an unresolved goalie is reported and left alone (the
projected ice-time leader stands and the ``goalie_unconfirmed`` gate stays).
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timezone
from pathlib import Path

import requests

from nhl_engine.data.teamnames import canonical

log = logging.getLogger("nhl_engine")

URL = "https://www.rotowire.com/hockey/nhl-lineups.php"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) payoff-pitch nhl-engine"
SOURCE = "rotowire"

STATUS_MAP = {"confirmed": "confirmed", "expected": "probable"}

_GAME_RE = re.compile(
    r'<div class="(lineup is-nhl[^"]*)">(.*?)(?=<div class="lineup is-nhl|<div class="lineups__footer|$)',
    re.S,
)
_ABBR_RE = re.compile(r'lineup__abbr">\s*([A-Z]{2,4})\s*<')
_TIME_RE = re.compile(r'class="lineup__time">([^<]*)<')
_LIST_RE = re.compile(r'<ul class="lineup__list (is-visit|is-home)">(.*?)</ul>', re.S)
_GOALIE_RE = re.compile(
    r'lineup__player-highlight-name">\s*<a[^>]*>([^<]+)</a>.*?'
    r"(?:is-(confirmed|expected)|</li>)",
    re.S,
)
_ITEM_RE = re.compile(r'<li class="lineup__(title|player)[^"]*">(.*?)</li>', re.S)
_POS_RE = re.compile(r'lineup__pos">([^<]*)<')
_NAME_RE = re.compile(r'<a title="([^"]+)"')
_NAME_FALLBACK_RE = re.compile(r"<a[^>]*>([^<]+)</a>")
_INJ_RE = re.compile(r'lineup__inj">([^<]*)<')


@dataclass(frozen=True)
class RotoPlayer:
    name: str
    pos: str
    unit: str  # "PP1" | "PP2" | "INJ"
    status: str = ""  # injury designation (IR, IR-NR, O, D, Q, ...) for INJ rows


@dataclass(frozen=True)
class RotoTeam:
    team: str
    goalie: str  # full name, '' if none listed
    goalie_status: str  # confirmed | probable | ''
    players: tuple[RotoPlayer, ...] = ()

    @property
    def injuries(self) -> list[RotoPlayer]:
        return [p for p in self.players if p.unit == "INJ"]

    @property
    def pp_units(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self.players:
            if p.unit.startswith("PP"):
                out.setdefault(p.unit, []).append(p.name)
        return out


@dataclass(frozen=True)
class RotoGame:
    away: RotoTeam
    home: RotoTeam
    time_et: str
    started: bool = False

    @property
    def matchup(self) -> str:
        return f"{self.away.team}@{self.home.team}"


@dataclass
class RotoSlate:
    slate: Date
    fetched_at: str
    games: list[RotoGame] = field(default_factory=list)
    raw_path: Path | None = None


def norm_name(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z ]", "", text.lower().replace("-", " ").replace(".", "")).strip()


def _parse_team(code: str, body: str) -> RotoTeam:
    goalie, gstatus = "", ""
    gm = _GOALIE_RE.search(body)
    if gm:
        goalie = html.unescape(gm.group(1)).strip()
        gstatus = STATUS_MAP.get((gm.group(2) or "").lower(), "")
    players: list[RotoPlayer] = []
    unit = ""
    for kind, inner in _ITEM_RE.findall(body):
        if kind == "title":
            title = re.sub(r"<[^>]+>", "", inner).strip().upper()
            if title.startswith("POWER PLAY"):
                unit = "PP" + title.rsplit("#", 1)[-1].strip()
            elif title.startswith("INJUR"):
                unit = "INJ"
            else:
                unit = title
            continue
        if not unit:
            continue
        pm = _POS_RE.search(inner)
        nm = _NAME_RE.search(inner) or _NAME_FALLBACK_RE.search(inner)
        if nm is None:
            continue
        im = _INJ_RE.search(inner)
        players.append(
            RotoPlayer(
                name=html.unescape(nm.group(1)).strip(),
                pos=(pm.group(1).strip() if pm else ""),
                unit=unit,
                status=(im.group(1).strip() if im else ""),
            )
        )
    return RotoTeam(code, goalie, gstatus, tuple(players))


def parse_page(page: str, slate: Date, fetched_at: str = "") -> RotoSlate:
    out = RotoSlate(slate=slate, fetched_at=fetched_at or _now())
    for classes, block in _GAME_RE.findall(page):
        abbrs = _ABBR_RE.findall(block)
        lists = _LIST_RE.findall(block)
        if len(abbrs) < 2 or len(lists) < 2:
            continue
        away_code, home_code = canonical(abbrs[0]), canonical(abbrs[1])
        by_side = {side: body for side, body in lists}
        if "is-visit" not in by_side or "is-home" not in by_side:
            continue
        tm = _TIME_RE.search(block)
        out.games.append(
            RotoGame(
                away=_parse_team(away_code, by_side["is-visit"]),
                home=_parse_team(home_code, by_side["is-home"]),
                time_et=(tm.group(1).strip() if tm else ""),
                started="has-started" in classes,
            )
        )
    return out


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def raw_dir(data_dir: Path, slate: Date) -> Path:
    return data_dir / "rotowire" / slate.isoformat()


def fetch(
    slate: Date,
    *,
    today: Date,
    data_dir: Path | None = None,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> RotoSlate:
    """Download the lineups page for ``slate`` (today or tomorrow only) and parse it."""
    delta = (slate - today).days
    if delta == 0:
        url = URL
    elif delta == 1:
        url = URL + "?date=tomorrow"
    else:
        raise ValueError(f"RotoWire publishes today/tomorrow only; {slate} is {delta:+d} days")
    sess = session or requests.Session()
    resp = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    fetched = _now()
    raw_path: Path | None = None
    if data_dir is not None:
        raw_path = raw_dir(data_dir, slate) / f"{fetched.replace(':', '').replace('-', '')}.html"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(resp.text, encoding="utf-8")
    parsed = parse_page(resp.text, slate, fetched)
    parsed.raw_path = raw_path
    return parsed


__all__ = [
    "SOURCE",
    "STATUS_MAP",
    "URL",
    "RotoGame",
    "RotoPlayer",
    "RotoSlate",
    "RotoTeam",
    "fetch",
    "norm_name",
    "parse_page",
    "raw_dir",
]
