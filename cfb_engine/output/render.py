"""Shared rendering helpers: house colors, matplotlib->base64, HTML->PDF, and
text->MP3 narration (neural edge-tts voice, gTTS fallback)."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# House palette (shared with the MLB desk's Morningstar style).
NAVY = "#16324f"
RED = "#c8102e"
GOLD = "#b8860b"
POS = "#2e7d32"
NEG = "#b23b3b"
INK = "#1a1a1a"
MUTE = "#6b7280"


def fig_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def to_pdf(html: str) -> bytes:
    from weasyprint import HTML

    return bytes(HTML(string=html).write_pdf())


def to_mp3(text: str, path: Path) -> bytes:
    """Neural sportscaster voice via edge-tts, gTTS fallback."""
    try:
        import asyncio

        import edge_tts

        async def _go() -> None:
            comm = edge_tts.Communicate(
                text, voice="en-US-ChristopherNeural", rate="+10%", pitch="+2Hz"
            )
            await comm.save(str(path))

        asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001
        logger.warning("edge-tts failed (%s); falling back to gTTS", exc)
        from gtts import gTTS

        gTTS(text=text, lang="en", tld="com", slow=False).save(str(path))
    return path.read_bytes()
