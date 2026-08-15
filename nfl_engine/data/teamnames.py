"""Team-name mapping between the odds board and nflverse.

Three vocabularies have to agree before a price can meet a rating:

* The Odds API says "Kansas City Chiefs".
* nflverse says ``KC``.
* nflverse itself is not internally consistent about three franchises -- the
  Rams are ``LA`` in the game file and ``LAR`` in some player feeds, Washington
  is ``WAS`` and ``WSH``, Jacksonville is ``JAX`` and ``JAC``.

:func:`canonical` fixes spelling. :func:`franchise` is a *different* question and
deliberately separate: it maps a relocated club onto its current code so a
rating series can span the move (``OAK`` -> ``LV``), which is right for team
strength and wrong for anything venue-specific, since Oakland's crowd, altitude
and travel were not Las Vegas's.
"""

from __future__ import annotations

import re

# Odds API display name -> nflverse code.
BY_NAME: dict[str, str] = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}

# Historical and alternate names that still appear on odds boards and in
# archived data.
_LEGACY_NAMES: dict[str, str] = {
    "oakland raiders": "OAK",
    "san diego chargers": "SD",
    "st louis rams": "STL",
    "st. louis rams": "STL",
    "washington redskins": "WAS",
    "washington football team": "WAS",
}

# Spelling variants of the same club in the same era.
_ALIASES: dict[str, str] = {
    "LAR": "LA",
    "WSH": "WAS",
    "JAC": "JAX",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "STL",
    "LVR": "LV",
}

# Relocations: historical code -> the franchise's current code.
_RELOCATED: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

TEAMS = frozenset(BY_NAME.values())


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9 .]", "", name.lower()).strip()


def canonical(code: str) -> str:
    """Normalize a team code's spelling, leaving relocations alone."""
    up = code.strip().upper()
    return _ALIASES.get(up, up)


def franchise(code: str) -> str:
    """The current code for the franchise that played as ``code``.

    Use for rating continuity across a relocation; do not use where the venue
    matters.
    """
    return _RELOCATED.get(canonical(code), canonical(code))


def code_for(name: str) -> str | None:
    """nflverse code for an odds-board team name, or ``None`` if unrecognized.

    Returning ``None`` rather than guessing is deliberate: a mis-mapped team
    silently prices the wrong side, which is a worse failure than a missing game.
    """
    key = _norm(name)
    if key in BY_NAME:
        return BY_NAME[key]
    if key in _LEGACY_NAMES:
        return _LEGACY_NAMES[key]
    # Fall back to the nickname alone ("Chiefs"), which some feeds send.
    for full, code in BY_NAME.items():
        if key and full.endswith(key):
            return code
    return None


def is_team(code: str) -> bool:
    return canonical(code) in TEAMS or canonical(code) in _RELOCATED
