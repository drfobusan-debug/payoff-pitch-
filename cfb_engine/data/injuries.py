"""Who is unavailable, and -- the part that decides whether it is worth anything --
how long we have known.

College injury news is late and incomplete, so a missing starter is the one input
the market genuinely mis-times. Measured on 2021-2025 box scores, where "out"
means an established starter took zero dropbacks:

    share of prior attempts lost   n     residual vs the close   beats the number
    under 50% (a rotation)        206         +1.54                  53.4%
    50-75%                        872         +0.39                  49.5%
    75-90%                        413         -2.20  (t=-3.00)       44.3%
    90%+                          191         -2.21  (t=-2.04)       43.5%

Fading a team that lost a 75%+ starter won 55.3% (t=+2.60) over 604 team-games,
and it held out of time: 54.2% in 2021-2023 against 56.7% (t=+2.20) in an
untouched 2024-2025. It is not a bad-team artifact -- fading *every* team-game
with any starter absence won 50.8%, and fading teams whose starter played lost at
48.8%.

But it is entirely a timing edge, and that is why nothing here moves a price:

    first game of the absence     406    56.2%   |  2024-25 holdout  59.9%
    he was out last week too      156    55.1%   |  2024-25 holdout  47.1%

Once the backup is common knowledge the market prices him correctly. So this
module's job is to record what we knew and when we knew it -- each observation is
stamped with the line at that moment in the availability log -- so the question
the backtest cannot answer can be answered prospectively: by the time this feed
tells us, has the number already moved? Until that is measured,
``CFBE_INJURY_QB_PTS`` is 0.0 and the absence is reported on the card only.

Two feeds, neither needing a key: the injury table carries designations but no
posting time, and the news RSS carries posting times. Together they date the news.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

from cfb_engine.data.teamnames import school_key
from mlb_engine.data import http

log = logging.getLogger(__name__)

_TABLE = "https://www.rotowire.com/cfootball/tables/injury-report.php"
_NEWS = "https://www.rotowire.com/rss/news.php"
# The page is a JS shell; the table endpoint behind it answers plain JSON, and it
# refuses the request without a browser-shaped referer.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.rotowire.com/cfootball/news.php?view=injuries",
    "Accept": "application/json,text/plain,*/*",
}

# RFC-822 dates with a named zone parse as naive, and the feed stamps everything
# in Pacific time -- taken as UTC that is a seven-hour error in the one number
# this module exists to measure.
_ZONES = {
    "PDT": -7, "PST": -8, "MDT": -6, "MST": -7,
    "CDT": -5, "CST": -6, "EDT": -4, "EST": -5,
    "UTC": 0, "GMT": 0,
}

# Designations that mean he is not playing. The feed abbreviates: "OFS" is out for
# season, "SUSP" suspended. "Questionable" and "Probable" are left out
# deliberately -- the measured population is players who took zero snaps, and a
# questionable tag is not that.
UNAVAILABLE = frozenset({
    "out", "doubtful", "ir", "ofs", "out for season", "susp", "suspended",
})


@dataclass(frozen=True)
class InjuryRow:
    player: str
    player_id: str
    team: str  # school_key
    position: str
    designation: str
    injury_type: str
    expected_return: str
    game_start: str  # feed's kickoff stamp, "" when absent

    @property
    def unavailable(self) -> bool:
        return self.designation.strip().lower() in UNAVAILABLE


InjuryBook = dict[str, list[InjuryRow]]


@dataclass(frozen=True)
class NewsItem:
    player_id: str
    headline: str
    posted: datetime


def _str(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    return str(value).strip() if value not in (None, "") else ""


def fetch_injury_report(*, timeout: float = 20.0) -> InjuryBook:
    """Current designations, keyed by :func:`school_key`.

    Fails soft: an empty book means the card simply says nothing about
    availability, which is what it did before this module existed.
    """
    try:
        resp = http.get(_TABLE, params={"team": "ALL", "pos": "ALL"},
                        headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - availability is a report-only nudge
        log.warning("injury report unavailable: %s", exc)
        return {}
    if not isinstance(payload, list):
        log.warning("injury report returned %s, not a list", type(payload).__name__)
        return {}
    book: InjuryBook = {}
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        team = _str(raw, "team") or _str(raw, "RotoSchoolName")
        player = _str(raw, "player")
        if not team or not player:
            continue
        row = InjuryRow(
            player=player,
            player_id=_str(raw, "ID"),
            team=school_key(team),
            position=_str(raw, "position"),
            designation=_str(raw, "IR"),
            injury_type=_str(raw, "injury_type"),
            expected_return=_str(raw, "ReturnDate"),
            game_start=_str(raw, "game_datetime"),
        )
        book.setdefault(row.team, []).append(row)
    return book


def fetch_news(*, cache: Path | None = None, timeout: float = 20.0) -> dict[str, NewsItem]:
    """Latest news item per player id, with the time it was posted.

    The injury table has no posting time, so this is what dates the news: the
    gap between ``posted`` and our own capture is the lead time the whole feature
    depends on.

    The feed is a short rolling window -- a handful of items -- so a designation
    read on Saturday morning is usually older than anything still in it. Stamps
    are therefore accumulated in ``cache`` across runs, which is what lets a
    Thursday item date a Saturday absence.
    """
    items = _cached_news(cache) if cache is not None else {}
    try:
        resp = http.get(_NEWS, params={"sport": "CFB"}, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.text)
    except Exception as exc:  # noqa: BLE001 - the timestamp is a nicety, not the price
        log.warning("injury news feed unavailable: %s", exc)
        return items
    for item in root.iter("item"):
        link = (item.findtext("link") or "").rstrip("/")
        stamp = item.findtext("pubDate") or ""
        player_id = link.rsplit("-", 1)[-1] if "-" in link else ""
        posted = _posted_at(stamp)
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


def _cached_news(path: Path) -> dict[str, NewsItem]:
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
        posted = _posted_at(str(row.get("posted", "")))
        if posted is None:
            continue
        out[str(pid)] = NewsItem(str(pid), str(row.get("headline", "")), posted)
    return out


def _save_news(path: Path, items: dict[str, NewsItem]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                pid: {"posted": item.posted.isoformat(), "headline": item.headline}
                for pid, item in items.items()
            }),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("could not save news stamps: %s", exc)


def _posted_at(stamp: str) -> datetime | None:
    """An RSS ``pubDate`` in UTC.

    The feed writes ``Fri, 14 Aug 2026 4:25:00 PM PDT``, which is not RFC 822 --
    ``parsedate_to_datetime`` drops the meridiem and reads that as 04:25, then the
    named zone is ignored too. Between them that is a nineteen-hour error in the
    lead time, so the 12-hour form is parsed explicitly and RFC 822 is the
    fallback.
    """
    stamp = stamp.strip()
    if not stamp:
        return None
    if stamp[:4].isdigit():  # our own cache writes ISO, which starts with the year
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
            return naive.replace(tzinfo=timezone(timedelta(hours=offset))).astimezone(
                timezone.utc
            )
    try:
        posted = parsedate_to_datetime(stamp)
    except (TypeError, ValueError):
        return None
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=timezone.utc)
    return posted.astimezone(timezone.utc)


def unavailable_for(book: InjuryBook, team: str) -> list[InjuryRow]:
    return [row for row in book.get(school_key(team), []) if row.unavailable]


def injury_note(book: InjuryBook, home: str, away: str) -> str | None:
    """One line for the card naming who is out on each side."""
    parts: list[str] = []
    for side in (home, away):
        rows = unavailable_for(book, side)
        if not rows:
            continue
        who = ", ".join(f"{r.position} {r.player}".strip() for r in rows[:3])
        extra = f" +{len(rows) - 3}" if len(rows) > 3 else ""
        parts.append(f"{side}: {who}{extra}")
    if not parts:
        return None
    return "Out -- " + "; ".join(parts) + " [reported, not scored]"


def log_availability(
    path: Path,
    *,
    home: str,
    away: str,
    rows: list[InjuryRow],
    spread: float | None,
    news: dict[str, NewsItem],
    observed: datetime | None = None,
) -> None:
    """Append one JSON line per absence, stamped with the line at this moment.

    This is the whole point of phase one. The backtest can say what a missing
    starter was worth against the close; only this log can say whether we hear
    about him before the number moves.
    """
    seen = (observed or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                item = news.get(row.player_id)
                fh.write(json.dumps({
                    "observed_at": seen.isoformat(),
                    "home": home,
                    "away": away,
                    # Normalized too, so the reader can tell which side of the
                    # spread the missing man was on without re-matching names.
                    "home_key": school_key(home),
                    "away_key": school_key(away),
                    "team": row.team,
                    "player": row.player,
                    "player_id": row.player_id,
                    "position": row.position,
                    "designation": row.designation,
                    "injury_type": row.injury_type,
                    "game_start": row.game_start,
                    "spread_home": spread,
                    "news_posted_at": item.posted.isoformat() if item else None,
                    "news_headline": item.headline if item else None,
                    "lead_time_s": (
                        (seen - item.posted).total_seconds() if item else None
                    ),
                }) + "\n")
    except OSError as exc:
        log.warning("could not append availability log: %s", exc)
