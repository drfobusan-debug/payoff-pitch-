"""Team-name mapping between the odds board, ESPN and the NBA.

Codes are the NBA's tricodes so an archived quote, an ESPN result and an
official injury report key the same way. An unrecognised name maps to ``None``
and the caller drops the game: pricing the wrong team is worse than skipping one.
"""

from __future__ import annotations

import re
import unicodedata

BY_NAME: dict[str, str] = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "los angeles clippers": "LAC",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}

_LEGACY_NAMES: dict[str, str] = {
    "la clippers": "LAC",
    "la lakers": "LAL",
    "philadelphia sixers": "PHI",
}

# ESPN and some feeds spell codes differently from the NBA.
_ALIASES: dict[str, str] = {
    "BRK": "BKN",
    "BK": "BKN",
    "GS": "GSW",
    "NO": "NOP",
    "NOR": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "PHO": "PHX",
    "CHO": "CHA",
}

CODES: frozenset[str] = frozenset(BY_NAME.values())


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace(".", "")).strip()


def code_for(name: str) -> str | None:
    """NBA tricode for a full team name, or ``None``."""
    key = _norm(name)
    return BY_NAME.get(key) or _LEGACY_NAMES.get(key)


def canonical(code: str) -> str:
    """One spelling per franchise code."""
    up = code.strip().upper().replace(".", "")
    return _ALIASES.get(up, up)


__all__ = ["BY_NAME", "CODES", "canonical", "code_for"]
