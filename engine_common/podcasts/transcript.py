"""Timestamped transcripts and the local Whisper pass that writes them.

On disk a transcript is one ``[mm:ss] text`` line per Whisper segment, so it can
be read by a person, diffed, and quoted back with its timestamp.
"""

from __future__ import annotations

import fcntl
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

WHISPER_MODEL = "small.en"


class PodcastUnavailable(RuntimeError):
    """Audio could not be fetched or transcribed; nothing is invented."""


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


def write_atomic(path: Path, text: str) -> Path:
    """Write via a ``.part`` sibling and rename, so a reader killed mid-write never
    leaves a truncated transcript or picks file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    tmp.replace(path)
    return path


def write_transcript(path: Path, lines: list[Line]) -> Path:
    return write_atomic(path, "".join(f"[{ln.stamp}] {ln.text}\n" for ln in lines))


@contextmanager
def one_at_a_time() -> Iterator[None]:
    """Whisper passes take turns across every engine on the machine: run together they
    split the cores and none of them finishes. ``WHISPER_LOCK`` names the lock file."""
    path = Path(os.getenv("WHISPER_LOCK") or Path.home() / ".engine_whisper.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def transcribe(audio: Path, dest: Path, *, model: str = WHISPER_MODEL, threads: int = 4) -> Path:
    """Local Whisper pass; ``faster-whisper`` is an optional extra (``pip install -e .[podcast]``)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise PodcastUnavailable(
            "faster-whisper not installed (pip install -e '.[podcast]')"
        ) from exc
    with one_at_a_time():
        wm = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        segments, _ = wm.transcribe(str(audio), beam_size=1, vad_filter=True)
        lines = [Line(int(s.start), s.text.strip()) for s in segments if s.text.strip()]
    return write_transcript(dest, lines)


def stamp_seconds(stamp: str) -> int | None:
    """``"12:34"`` or ``"1:02:03"`` -> seconds; anything else -> ``None``."""
    parts = stamp.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    secs = 0
    for p in parts:
        secs = secs * 60 + int(p)
    return secs


__all__ = [
    "WHISPER_MODEL",
    "Line",
    "PodcastUnavailable",
    "one_at_a_time",
    "read_transcript",
    "stamp_seconds",
    "transcribe",
    "write_atomic",
    "write_transcript",
]
