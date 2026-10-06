"""Which episodes to read, and the transcript behind each.

An episode is read once: its audio is downloaded, transcribed locally and the
MP3 deleted, so the archive costs disk only as text. A missing transcript is
reported, never filled in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

from cfb_engine.podcast.shows import Show
from engine_common.podcasts.feed import FeedItem, download, fetch_xml, parse_items
from engine_common.podcasts.transcript import WHISPER_MODEL, PodcastUnavailable, transcribe

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Episode:
    show: Show
    item: FeedItem

    @property
    def key(self) -> str:
        return self.item.key

    @property
    def published(self) -> datetime:
        return datetime.fromisoformat(self.item.published)


def podcast_dir(data_dir: Path) -> Path:
    return data_dir / "podcasts"


def transcript_path(data_dir: Path, ep: Episode) -> Path:
    return podcast_dir(data_dir) / ep.show.key / f"{ep.key}.txt"


def select(show: Show, items: list[FeedItem], since: datetime, until: datetime) -> list[Episode]:
    """The show's episodes published in ``[since, until)`` that may carry CFB picks."""
    out = [
        Episode(show, it)
        for it in items
        if since <= datetime.fromisoformat(it.published) < until
        and show.wants(it.title, it.description)
    ]
    return sorted(out, key=lambda e: e.published)


def fetch_episodes(
    show: Show, since: datetime, until: datetime, *, session: requests.Session | None = None
) -> list[Episode]:
    return select(show, parse_items(fetch_xml(show.feed_url, session=session)), since, until)


def ensure_transcript(
    ep: Episode,
    data_dir: Path,
    *,
    session: requests.Session | None = None,
    model: str = WHISPER_MODEL,
    transcribe_missing: bool = True,
) -> Path | None:
    """The episode's transcript, transcribing it first if allowed; ``None`` if not had."""
    dest = transcript_path(data_dir, ep)
    if dest.exists():
        return dest
    if not transcribe_missing:
        return None
    audio = dest.with_suffix(".mp3")
    try:
        download(ep.item.audio_url, audio, session=session)
        transcribe(audio, dest, model=model)
    except PodcastUnavailable as exc:
        log.warning("%s %r: %s", ep.show.name, ep.item.title, exc)
        return None
    finally:
        if dest.exists():
            audio.unlink(missing_ok=True)
    return dest


__all__ = [
    "Episode",
    "ensure_transcript",
    "fetch_episodes",
    "podcast_dir",
    "select",
    "transcript_path",
]
