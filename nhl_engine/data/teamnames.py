"""Team-name mapping between the odds board and the NHL API.

The Odds API says "Montréal Canadiens"; the NHL API says ``MTL``. Codes are the
NHL API's ``abbrev`` so an archived quote and an official result key the same
way. An unrecognised name maps to ``None`` and the caller drops the game: pricing
the wrong team is a worse failure than skipping one.
"""

from __future__ import annotations

import re
import unicodedata

BY_NAME: dict[str, str] = {
    "anaheim ducks": "ANA",
    "boston bruins": "BOS",
    "buffalo sabres": "BUF",
    "calgary flames": "CGY",
    "carolina hurricanes": "CAR",
    "chicago blackhawks": "CHI",
    "colorado avalanche": "COL",
    "columbus blue jackets": "CBJ",
    "dallas stars": "DAL",
    "detroit red wings": "DET",
    "edmonton oilers": "EDM",
    "florida panthers": "FLA",
    "los angeles kings": "LAK",
    "minnesota wild": "MIN",
    "montreal canadiens": "MTL",
    "nashville predators": "NSH",
    "new jersey devils": "NJD",
    "new york islanders": "NYI",
    "new york rangers": "NYR",
    "ottawa senators": "OTT",
    "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT",
    "san jose sharks": "SJS",
    "seattle kraken": "SEA",
    "st louis blues": "STL",
    "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR",
    "utah mammoth": "UTA",
    "vancouver canucks": "VAN",
    "vegas golden knights": "VGK",
    "washington capitals": "WSH",
    "winnipeg jets": "WPG",
}

_LEGACY_NAMES: dict[str, str] = {
    "utah hockey club": "UTA",
    "arizona coyotes": "ARI",
    "phoenix coyotes": "ARI",
    "la kings": "LAK",
    "st louis": "STL",
    "montreal": "MTL",
}

# Codes some feeds spell differently from the NHL API.
_ALIASES: dict[str, str] = {
    "LA": "LAK",
    "SJ": "SJS",
    "TB": "TBL",
    "NJ": "NJD",
    "WAS": "WSH",
    "VEG": "VGK",
    "UTAH": "UTA",
}

CODES: frozenset[str] = frozenset(BY_NAME.values())


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace(".", "")).strip()


def code_for(name: str) -> str | None:
    """NHL API abbreviation for an Odds API team name, or ``None``."""
    key = _norm(name)
    return BY_NAME.get(key) or _LEGACY_NAMES.get(key)


def canonical(code: str) -> str:
    """One spelling per franchise code."""
    up = code.strip().upper().replace(".", "")
    return _ALIASES.get(up, up)


__all__ = ["BY_NAME", "CODES", "canonical", "code_for"]
