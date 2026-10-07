"""Timestamped transcripts and the local Whisper pass that writes them.

On disk a transcript is one ``[mm:ss] text`` line per Whisper segment, so it can
be read by a person, diffed, and quoted back with its timestamp.
"""

from __future__ import annotations

import re
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
    "read_transcript",
    "stamp_seconds",
    "transcribe",
    "write_transcript",
]
