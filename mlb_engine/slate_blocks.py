"""The slate passes: which games each one prices, and what it calls them.

A buy is only made inside WINDOW_HOURS of first pitch (the clock gate), so the
day is priced in passes, each pricing the games starting inside the next
WINDOW_HOURS and emailing them as its own slate article. The pass times are
chosen so the windows tile the day's first pitches (12:05 through 22:10 local):
a game is bought by exactly one pass, and none is refused only because no pass
came back for it. Times are the machine's local clock, as launchd reads them.
"""

from __future__ import annotations

from dataclasses import dataclass

WINDOW_HOURS = 3.0


@dataclass(frozen=True)
class SlateBlock:
    name: str
    run_at: str
    starts: str
    greeting: str


BLOCKS: dict[str, SlateBlock] = {
    b.name: b
    for b in (
        SlateBlock("matinee", "11:55", "first pitch noon to 3pm", "Good afternoon"),
        SlateBlock("afternoon", "14:55", "first pitch 3pm to 6pm", "Good afternoon"),
        SlateBlock("evening", "17:55", "first pitch 6pm to 9pm", "Good evening"),
        SlateBlock("late", "20:55", "first pitch 9pm or later", "Good evening"),
    )
}
BLOCK_NAMES = tuple(BLOCKS)


def block(name: str) -> SlateBlock:
    try:
        return BLOCKS[name]
    except KeyError:
        raise ValueError(f"block must be one of {', '.join(BLOCK_NAMES)}, got {name!r}") from None
