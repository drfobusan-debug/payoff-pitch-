"""An RSS feed as a list of items, and the MP3 behind one of them.

The feed, not Spotify or Apple, is the source of truth: it carries the title,
the publish time and the audio URL, and keeps the archive.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from engine_common.podcasts.transcript import PodcastUnavailable

_ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class FeedItem:
    guid: str
    title: str
    published: str  # ISO 8601 with offset
    audio_url: str
    duration: str
    description: str = ""

    @property
    def key(self) -> str:
        return hashlib.sha1(self.guid.encode()).hexdigest()[:12]


def parse_items(xml: str) -> list[FeedItem]:
    """Every item with a title, an enclosure and a parseable ``pubDate``."""
    root = ET.fromstring(xml)
    out: list[FeedItem] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        enc = item.find("enclosure")
        url = enc.get("url") if enc is not None else None
        if not title or not url:
            continue
        try:
            pub = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        desc = item.findtext("description") or item.findtext(f"{_ITUNES}summary") or ""
        out.append(
            FeedItem(
                guid=(item.findtext("guid") or url).strip(),
                title=title,
                published=pub.astimezone(None).isoformat(),
                audio_url=url,
                duration=(item.findtext(f"{_ITUNES}duration") or "").strip(),
                description=" ".join(_TAGS.sub(" ", desc).split()),
            )
        )
    return out


def fetch_xml(url: str, *, session: requests.Session | None = None) -> str:
    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=30, headers={"User-Agent": "payoff-pitch/1.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise PodcastUnavailable(f"feed fetch failed: {exc}") from exc
    return resp.text


def download(url: str, dest: Path, *, session: requests.Session | None = None) -> Path:
    """Stream ``url`` to ``dest`` once; a complete file already there is reused."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    try:
        with sess.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
            tmp.replace(dest)
    except requests.RequestException as exc:
        raise PodcastUnavailable(f"audio download failed: {exc}") from exc
    return dest


__all__ = ["FeedItem", "download", "fetch_xml", "parse_items"]
