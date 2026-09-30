"""Who is unavailable, and -- the only question that decides whether it is worth
anything -- whether we hear it before the number moves.

The NFL injury report is the most public information in sports: it is filed with
the league, published on a schedule, and read by every book before it is read by
us. So the prior is that a designation is already in the price by the time it
reaches this module, and nothing here is allowed to touch a probability, a screen
or a tier. What it does is record, per observation:

* the player, his team and his position **group** -- quarterback, skill, line;
* the designation exactly as the source published it;
* when the source posted it, and when we captured it;
* the market number at the last archived capture *before* that posting, and at
  the first capture after it.

That last pair is the measurement. MLB's availability work (#154) established the
shape and the college version (``cfb_engine/data/injuries.py``) established the
honest finding: a retrospective absence study measures a starter hooked in a
blowout as much as it measures a starter who was out, and once the backup is
common knowledge the market prices him correctly. The NFL side has something
neither had -- a timestamped price archive of our own -- so the lead-time
question can be answered prospectively rather than argued about. See
:mod:`nfl_engine.audit.availability` for the movement side.

Until that measurement exists, an absence is a line on the card and nothing else.

Two keyless feeds, because neither alone dates the news: the injury table carries
designations and no posting time, the news RSS carries posting times.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

from mlb_engine.data import http
from nfl_engine.data.teamnames import canonical, is_team

log = logging.getLogger(__name__)

_TABLE = "https://www.rotowire.com/football/tables/injury-report.php"
_NEWS = "https://www.rotowire.com/rss/news.php"
# The visible page is a JS shell; the table endpoint behind it answers plain JSON
# and refuses the request without a browser-shaped referer.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.rotowire.com/football/injury-report.php",
    "Accept": "application/json,text/plain,*/*",
}
ROTOWIRE = "rotowire"

# RFC-822 dates with a named zone parse as naive, and the feed stamps everything
# Pacific: read as UTC that is a seven-hour error in the one number this module
# exists to measure.
_ZONES = {
    "PDT": -7,
    "PST": -8,
    "MDT": -6,
    "MST": -7,
    "CDT": -5,
    "CST": -6,
    "EDT": -4,
    "EST": -5,
    "UTC": 0,
    "GMT": 0,
}

QB = "QB"
SKILL = "SKILL"
LINE = "OL"
OTHER = "other"

# The three groups the watcher is scoped to. Wider than the quarterback the
# engine already knows about (``features/quarterback.py``), because a left tackle
# and a number-one receiver move a number too -- and narrower than the whole
# report, because a fourth safety does not, and counting him would bury the ones
# that do.
_GROUPS: dict[str, str] = {
    "QB": QB,
    "RB": SKILL,
    "FB": SKILL,
    "HB": SKILL,
    "WR": SKILL,
    "TE": SKILL,
    "OL": LINE,
    "T": LINE,
    "LT": LINE,
    "RT": LINE,
    "OT": LINE,
    "G": LINE,
    "LG": LINE,
    "RG": LINE,
    "OG": LINE,
    "C": LINE,
}
WATCHED = (QB, SKILL, LINE)

# Designations that mean he is not playing. "Questionable" is deliberately out:
# it is the designation the league uses when it does not know either, and a
# watcher that treats it as an absence reports half the roster every Friday.
UNAVAILABLE = frozenset(
    {
        "out",
        "doubtful",
        "ir",
        "injured reserve",
        "pup",
        "nfi",
        "susp",
        "suspended",
        "ofs",
        "out for season",
    }
)


def group_of(position: str) -> str:
    """Position group, keyed on the position as the source spells it."""
    return _GROUPS.get(position.strip().upper(), OTHER)


@dataclass(frozen=True)
class InjuryRow:
    """One designation as published, plus where it came from.

    Raw on purpose: ``designation`` is the source's own word, not a severity we
    invented, so a later study can re-bucket it without re-fetching.
    """

    player: str
    player_id: str
    team: str
    position: str
    designation: str
    injury: str
    source: str = ROTOWIRE

    @property
    def group(self) -> str:
        return group_of(self.position)

    @property
    def unavailable(self) -> bool:
        return self.designation.strip().lower() in UNAVAILABLE

    @property
    def watched(self) -> bool:
        return self.unavailable and self.group in WATCHED


InjuryBook = dict[str, list[InjuryRow]]


@dataclass(frozen=True)
class NewsItem:
    player_id: str
    headline: str
    posted: datetime


def _text(row: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def fetch_report(*, timeout: float = 20.0) -> InjuryBook:
    """Current designations, keyed by nflverse team code.

    Fails soft: an empty book means the card says nothing about availability,
    which is what it said before this module existed. An outside feed is never
    allowed to be the reason a week has no output.
    """
    try:
        resp = http.get(
            _TABLE, params={"team": "ALL", "pos": "ALL"}, headers=_HEADERS, timeout=timeout
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - availability is reported, never priced
        log.warning("injury report unavailable: %s", exc)
        return {}
    if not isinstance(payload, list):
        log.warning("injury report returned %s, not a list", type(payload).__name__)
        return {}
    book: InjuryBook = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        team = canonical(_text(raw, "team"))
        player = _text(raw, "player")
        if not player or not is_team(team):
            continue
        row = InjuryRow(
            player=player,
            player_id=_text(raw, "ID"),
            team=team,
            position=_text(raw, "position"),
            designation=_text(raw, "status", "IR"),
            injury=_text(raw, "injury", "injury_type"),
        )
        book.setdefault(row.team, []).append(row)
    return book


def fetch_news(*, cache: Path | None = None, timeout: float = 20.0) -> dict[str, NewsItem]:
    """Latest news item per player id, with the time it was posted.

    The table has no posting time, so this is what dates a designation. The feed
    is a short rolling window, so stamps are accumulated in ``cache`` across
    runs -- that is what lets a Wednesday item date a Sunday absence.
    """
    items = _cached_news(cache) if cache is not None else {}
    try:
        resp = http.get(_NEWS, params={"sport": "NFL"}, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.text)
    except Exception as exc:  # noqa: BLE001 - a missing stamp costs the lead time, not the week
        log.warning("injury news feed unavailable: %s", exc)
        return items
    for item in root.iter("item"):
        link = (item.findtext("link") or "").rstrip("/")
        player_id = link.rsplit("-", 1)[-1] if "-" in link else ""
        posted = posted_at(item.findtext("pubDate") or "")
        if not player_id.isdigit() or posted is None:
            continue
        known = items.get(player_id)
        if known is not None and known.posted >= posted:
            continue
        items[player_id] = NewsItem(
            player_id=player_id,
            headline=(item.findtext("title") or "").strip(),
            posted=posted,
        )
    if cache is not None:
        _save_news(cache, items)
    return items


def posted_at(stamp: str) -> datetime | None:
    """An RSS ``pubDate`` in UTC, or ``None`` when it cannot be read.

    The feed writes ``Sun, 23 Aug 2026 9:07:00 PM PDT``, which is not RFC 822:
    ``parsedate_to_datetime`` drops the meridiem and reads that as 09:07, then
    ignores the named zone as well. Between them that is a nineteen-hour error in
    a lead time, so the 12-hour form is parsed explicitly and RFC 822 is only the
    fallback.
    """
    stamp = stamp.strip()
    if not stamp:
        return None
    if stamp[:4].isdigit():  # our own log writes ISO, which starts with the year
        try:
            return datetime.fromisoformat(stamp).astimezone(timezone.utc)
        except ValueError:
            return None
    body, _, zone = stamp.rpartition(" ")
    offset = _ZONES.get(zone.upper())
    if offset is not None:
        for fmt in ("%a, %d %b %Y %I:%M:%S %p", "%a, %d %b %Y %H:%M:%S"):
            try:
                naive = datetime.strptime(body, fmt)
            except ValueError:
                continue
            return naive.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(timezone.utc)
    try:
        read = parsedate_to_datetime(stamp)
    except (TypeError, ValueError):
        return None
    if read.tzinfo is None:
        read = read.replace(tzinfo=timezone.utc)
    return read.astimezone(timezone.utc)


def _cached_news(path: Path | None) -> dict[str, NewsItem]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, NewsItem] = {}
    for pid, row in raw.items():
        if not isinstance(row, dict):
            continue
        when = posted_at(str(row.get("posted", "")))
        if when is None:
            continue
        out[str(pid)] = NewsItem(str(pid), str(row.get("headline", "")), when)
    return out


def _save_news(path: Path, items: dict[str, NewsItem]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    pid: {"posted": item.posted.isoformat(), "headline": item.headline}
                    for pid, item in items.items()
                }
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("could not save news stamps: %s", exc)


def watched_for(book: InjuryBook, team: str) -> list[InjuryRow]:
    """Unavailable quarterbacks, skill players and linemen on one team."""
    order = {QB: 0, SKILL: 1, LINE: 2}
    rows = [row for row in book.get(canonical(team), []) if row.watched]
    return sorted(rows, key=lambda r: (order[r.group], r.player))
