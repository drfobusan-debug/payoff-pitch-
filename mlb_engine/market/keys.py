"""Canonical ``selection`` strings for the ``(matchup, market, selection)`` quote
key. Shared by the pipeline (which emits recommendations) and the market-data
clients (which price them) so quotes line up with the recommendations exactly.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import TypeVar

Key = tuple[str, str, str]
_V = TypeVar("_V")

# Dropped when matching a name. A book writes "Ronald Acuna Jr."; the lineup
# feed writes "Ronald Acuna". Both mean the same hitter. "v" is deliberately
# absent -- too easy to collide with a real surname.
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})
# Kept as significant: digits and the sign/point of a line. Stripping "+1.5"
# and "-1.5" to the same thing would silently price a run line at its mirror.
_KEEP = re.compile(r"[^a-z0-9 .+-]")
# Closed up rather than spaced, because the variant spelling is the apostrophe's
# absence -- O'Neill / ONeill -- not a space in its place.
_APOSTROPHE = re.compile(r"['\u2019\u02bc]")
# A hyphen inside a name is a space -- Ha-Seong / Ha Seong, Crow-Armstrong --
# but a hyphen before a digit is the sign of a run line and must survive.
_NAME_HYPHEN = re.compile(r"-(?!\d)")


def canonical(selection: str) -> str:
    """A selection string with the spelling of a name argued out of it.

    Accents, apostrophes, hyphens and generational suffixes vary between the
    books and the lineup feed for the same player -- Jose Ramirez, Ha-Seong
    Kim, Tyler O'Neill, Vladimir Guerrero Jr. -- and the quote key is matched
    by string equality, so those hitters were never priced at all.
    """
    text = unicodedata.normalize("NFKD", selection).lower()
    text = _APOSTROPHE.sub("", "".join(c for c in text if not unicodedata.combining(c)))
    text = _NAME_HYPHEN.sub(" ", text)
    return " ".join(t for t in _KEEP.sub(" ", text).split() if t.strip(".") not in _SUFFIXES)


def canonical_index(quotes: dict[Key, _V]) -> dict[Key, _V]:
    """Quotes re-keyed on the canonical selection, for use as a fallback.

    Two different players can canonicalize alike -- the Nationals' Luis Garcia
    Jr. and the Astros' Luis Garcia -- so a canonical form claimed by more than
    one selection in the same game is dropped rather than guessed at. An
    ambiguous name goes back to being unpriced, which is the status quo.
    """
    grouped: dict[Key, list[_V]] = defaultdict(list)
    for (matchup, market, selection), value in quotes.items():
        grouped[(matchup, market, canonical(selection))].append(value)
    return {k: v[0] for k, v in grouped.items() if len(v) == 1}


def _pt(point: float) -> str:
    return f"{point:+.1f}"


def game_ml(abbrev: str) -> str:
    return f"{abbrev} ML"


def game_rl(abbrev: str, point: float) -> str:
    return f"{abbrev} {_pt(point)}"


def game_total(over: bool, line: float) -> str:
    return f"{'Over' if over else 'Under'} {line}"


def f5_ml(abbrev: str) -> str:
    return f"{abbrev} F5 ML"


def f5_total(over: bool, line: float) -> str:
    return f"F5 {'Over' if over else 'Under'} {line}"


def f5_rl(abbrev: str, point: float) -> str:
    return f"{abbrev} F5 {_pt(point)}"


def prop_side(side: str) -> str:
    """Selection-string marker for a prop side.

    ``over`` keeps the historic ``o`` spelling, so every selection already held
    in the ledger and the closing captures still matches by string.
    """
    return "u" if side == "under" else "o"


def batter_prop(name: str, stat: str, line: float, side: str = "over") -> str:
    return f"{name} {stat} {prop_side(side)}{line}"


def pitcher_prop(name: str, label: str, line: float, side: str = "over") -> str:
    return f"{name} {label} {prop_side(side)}{line}"


# Engine pitcher stat symbol -> label used in the selection string.
PITCHER_LABEL = {"K": "Ks", "outs": "Outs", "H": "Hits", "BB": "Walks", "ER": "ER"}
