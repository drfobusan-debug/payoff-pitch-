"""Hockey Gambling Podcast (SGPN) as a slate-dated source: find the episode
that covers a slate, keep its audio + transcript under ``<data>/podcast``.

The feed is the source of truth, not Spotify: it exposes title, publish time
and the MP3. Slate dates are read from the title ("Tuesday, October 6",
"Thursday, Oct. 1 & Friday, Oct. 2", "9/29 + 9/30") -- an episode whose title
names no date (division previews, futures) is never attached to a slate.
Transcription is local (``faster-whisper``, optional extra); when it is not
installed the caller reports "no transcript" rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import datetime
from pathlib import Path

import requests

from engine_common.podcasts.feed import download, parse_items
from engine_common.podcasts.transcript import (
    WHISPER_MODEL,
    Line,
    PodcastUnavailable,
    read_transcript,
    transcribe,
    write_transcript,
)

FEED_URL = "https://feeds.simplecast.com/caDpJU1D"
# A title date is for this episode's slate: at most this far from publish time.
MAX_LEAD_DAYS = 10

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        1,
    )
    for m in names
}
_MONTH_DAY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b", re.I
)
_NUMERIC = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")


@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    published: str  # ISO 8601, UTC
    audio_url: str
    duration: str
    slate_dates: tuple[str, ...]  # ISO dates named in the title

    @property
    def key(self) -> str:
        return hashlib.sha1(self.guid.encode()).hexdigest()[:12]

    def covers(self, slate: Date) -> bool:
        return slate.isoformat() in self.slate_dates


def _resolve_year(month: int, day: int, published: datetime) -> Date | None:
    """Pick the year that puts the title date near publish time (Dec 31 -> Jan 1)."""
    for year in (published.year - 1, published.year, published.year + 1):
        try:
            d = Date(year, month, day)
        except ValueError:
            return None
        delta = (d - published.date()).days
        if -3 <= delta <= MAX_LEAD_DAYS:
            return d
    return None


def slate_dates_from_title(title: str, published: datetime) -> tuple[str, ...]:
    found: list[Date] = []
    for m in _MONTH_DAY.finditer(title):
        mon = _MONTHS.get(m.group(1).lower())
        if mon is None:
            continue
        d = _resolve_year(mon, int(m.group(2)), published)
        if d and d not in found:
            found.append(d)
    for m in _NUMERIC.finditer(title):
        d = _resolve_year(int(m.group(1)), int(m.group(2)), published)
        if d and d not in found:
            found.append(d)
    return tuple(d.isoformat() for d in sorted(found))


def parse_feed(xml: str) -> list[Episode]:
    return [
        Episode(
            guid=it.guid,
            title=it.title,
            published=it.published,
            audio_url=it.audio_url,
            duration=it.duration,
            slate_dates=slate_dates_from_title(it.title, datetime.fromisoformat(it.published)),
        )
        for it in parse_items(xml)
    ]


def fetch_feed(url: str = FEED_URL, *, session: requests.Session | None = None) -> list[Episode]:
    sess = session or requests.Session()
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    return parse_feed(resp.text)


def episodes_for(episodes: list[Episode], slate: Date) -> list[Episode]:
    """Episodes whose title names ``slate``, newest first."""
    hits = [e for e in episodes if e.covers(slate)]
    hits.sort(key=lambda e: e.published, reverse=True)
    return hits


# ---------------------------------------------------------------- storage


def podcast_dir(data_dir: Path) -> Path:
    return data_dir / "podcast"


def audio_path(data_dir: Path, ep: Episode) -> Path:
    return podcast_dir(data_dir) / f"{ep.key}.mp3"


def transcript_path(data_dir: Path, ep: Episode) -> Path:
    return podcast_dir(data_dir) / f"{ep.key}.txt"


def meta_path(data_dir: Path, ep: Episode) -> Path:
    return podcast_dir(data_dir) / f"{ep.key}.json"


def save_meta(data_dir: Path, ep: Episode) -> Path:
    p = meta_path(data_dir, ep)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(ep), indent=1))
    return p


def download_audio(ep: Episode, data_dir: Path, *, session: requests.Session | None = None) -> Path:
    return download(ep.audio_url, audio_path(data_dir, ep), session=session)


def ensure_transcript(
    ep: Episode,
    data_dir: Path,
    *,
    session: requests.Session | None = None,
    model: str = WHISPER_MODEL,
    transcribe_missing: bool = True,
) -> Path | None:
    """Cached transcript for ``ep``; downloads/transcribes when allowed, else ``None``."""
    save_meta(data_dir, ep)
    dest = transcript_path(data_dir, ep)
    if dest.exists():
        return dest
    if not transcribe_missing:
        return None
    audio = download_audio(ep, data_dir, session=session)
    return transcribe(audio, dest, model=model)


__all__ = [
    "FEED_URL",
    "Episode",
    "Line",
    "PodcastUnavailable",
    "download_audio",
    "ensure_transcript",
    "episodes_for",
    "fetch_feed",
    "parse_feed",
    "podcast_dir",
    "read_transcript",
    "slate_dates_from_title",
    "transcribe",
    "transcript_path",
    "write_transcript",
]
