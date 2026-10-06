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
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date as Date
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

FEED_URL = "https://feeds.simplecast.com/caDpJU1D"
WHISPER_MODEL = "small.en"
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


class PodcastUnavailable(RuntimeError):
    """Audio could not be fetched or transcribed; nothing is invented."""


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
    root = ET.fromstring(xml)
    out: list[Episode] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        enc = item.find("enclosure")
        url = enc.get("url") if enc is not None else None
        if not title or not url:
            continue
        raw_pub = item.findtext("pubDate") or ""
        try:
            pub = parsedate_to_datetime(raw_pub)
        except (TypeError, ValueError):
            continue
        guid = (item.findtext("guid") or url).strip()
        dur = (item.findtext("{http://www.itunes.com/dtds/podcast-1.0.dtd}duration") or "").strip()
        out.append(
            Episode(
                guid=guid,
                title=title,
                published=pub.astimezone(None).isoformat(),
                audio_url=url,
                duration=dur,
                slate_dates=slate_dates_from_title(title, pub),
            )
        )
    return out


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
    dest = audio_path(data_dir, ep)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    try:
        with sess.get(ep.audio_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
            tmp.replace(dest)
    except requests.RequestException as exc:
        raise PodcastUnavailable(f"audio download failed: {exc}") from exc
    return dest


@dataclass(frozen=True)
class Line:
    seconds: int
    text: str

    @property
    def stamp(self) -> str:
        return f"{self.seconds // 60:02d}:{self.seconds % 60:02d}"


_LINE = re.compile(r"^\[(\d+):(\d{2})\]\s*(.*)$")


def read_transcript(path: Path) -> list[Line]:
    out: list[Line] = []
    for raw in path.read_text().splitlines():
        m = _LINE.match(raw.strip())
        if m and m.group(3):
            out.append(Line(int(m.group(1)) * 60 + int(m.group(2)), m.group(3).strip()))
    return out


def write_transcript(path: Path, lines: list[Line]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"[{ln.stamp}] {ln.text}\n" for ln in lines))
    return path


def transcribe(audio: Path, dest: Path, *, model: str = WHISPER_MODEL, threads: int = 4) -> Path:
    """Local Whisper pass; ``faster-whisper`` is an optional extra (``pip install -e .[podcast]``)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PodcastUnavailable(
            "faster-whisper not installed (pip install -e '.[podcast]')"
        ) from exc
    wm = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
    segments, _ = wm.transcribe(str(audio), beam_size=1, vad_filter=True)
    lines = [Line(int(s.start), s.text.strip()) for s in segments if s.text.strip()]
    return write_transcript(dest, lines)


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
