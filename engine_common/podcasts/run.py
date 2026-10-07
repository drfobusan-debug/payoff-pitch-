"""Read every show once: new episodes, one transcript each, one extraction each.

The pass every engine shares. Its output is the store's checked picks, each
already filed under its league; an engine's own command then places and grades
the ones in its league. Nothing here prices anything.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from engine_common.podcasts.episodes import Episode, ensure_transcript, fetch_episodes
from engine_common.podcasts.extract import extract
from engine_common.podcasts.shows import SHOWS, Show
from engine_common.podcasts.transcript import read_transcript


@dataclass
class ReadReport:
    episodes: int = 0
    transcribed: int = 0
    by_league: Counter[str] = field(default_factory=Counter)
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self, since: datetime) -> str:
        leagues = ", ".join(f"{n} {lg.upper()}" for lg, n in sorted(self.by_league.items()))
        return (
            f"Podcasts since {since.date()}: {self.episodes} episodes, "
            f"{self.transcribed} transcribed, picks kept: {leagues or 'none'}."
        )


def read_feeds(
    root: Path,
    since: datetime,
    until: datetime,
    *,
    shows: Iterable[Show] = SHOWS,
    league: str | None = None,
    transcribe_missing: bool = True,
    legacy: Sequence[Path] = (),
    api_key: str | None = None,
    log: bool = True,
) -> ReadReport:
    """Read ``shows``' episodes published in ``[since, until)`` into ``root``.

    ``league`` narrows which episodes are read (a multi-sport show's NBA hour is
    skipped by a CFB pass) but each episode is still read for every league its
    show covers, so it is transcribed and extracted once for all of them.

    Without an OpenAI key the episodes are transcribed only: no pick is kept, so
    none can be printed or graded, and nothing is guessed in its place.
    """
    key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    if not key and log:
        print("OPENAI_API_KEY not set: transcribing only, no picks extracted.")
    rep = ReadReport()
    for show in shows:
        try:
            wanted = show.for_league(league) if league else show
            eps = [Episode(show, e.item) for e in fetch_episodes(wanted, since, until)]
        except Exception as exc:  # noqa: BLE001 - one feed down never stops the others
            rep.failed.append(f"{show.name}: feed unavailable ({exc})")
            continue
        for ep in eps:
            rep.episodes += 1
            path = ensure_transcript(ep, root, transcribe_missing=transcribe_missing, legacy=legacy)
            if path is None:
                rep.missing.append(f"{show.name}: {ep.item.title}")
                continue
            rep.transcribed += 1
            if not key:
                continue
            try:
                got = extract(ep, read_transcript(path), root, api_key=key)
            except Exception as exc:  # noqa: BLE001
                rep.failed.append(f"{show.name} {ep.item.title!r}: extraction failed ({exc})")
                continue
            rep.by_league.update(p.league for p in got.picks)
    if log:
        print(rep.summary(since))
        for m in rep.missing[:20]:
            print(f"  no transcript: {m}")
        for f in rep.failed:
            print(f"  {f}")
    return rep


__all__ = ["ReadReport", "read_feeds"]
