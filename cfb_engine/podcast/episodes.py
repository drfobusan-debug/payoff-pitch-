"""CFB episodes and transcripts: the shared store's (see :mod:`engine_common.podcasts.episodes`)."""

from __future__ import annotations

from engine_common.podcasts.episodes import (
    Episode,
    ensure_transcript,
    fetch_episodes,
    podcast_dir,
    select,
    transcript_path,
)
from engine_common.podcasts.feed import download

__all__ = [
    "Episode",
    "download",
    "ensure_transcript",
    "fetch_episodes",
    "podcast_dir",
    "select",
    "transcript_path",
]
