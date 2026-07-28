"""Rotowire client: public daily-lineups page + optional authenticated extras.

The daily-lineups page (``rotowire.com/baseball/daily-lineups.php``) is public
and lists each game's expected/confirmed batting order (with position and bat
side) and both probable pitchers -- available *before* MLB posts official
lineups, which is what the pipeline gates on. ``fetch_expected_lineups`` scrapes
it so games can be priced earlier; names are resolved to MLBAM ids downstream.

Umpire assignments and reliever availability remain credential-gated hooks.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

import pandas as pd
import requests

from mlb_engine.config import Credentials

log = logging.getLogger(__name__)

LINEUPS_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
_HEADERS = {"User-Agent": "Mozilla/5.0 (mlb-prediction-engine)"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass(frozen=True)
class RotoBatter:
    name: str
    position: str
    bats: str | None


@dataclass(frozen=True)
class RotoLineup:
    abbrev: str
    nickname: str
    confirmed: bool
    pitcher: str | None
    pitcher_throws: str | None
    batters: list[RotoBatter] = field(default_factory=list)


@dataclass(frozen=True)
class RotoGame:
    away: RotoLineup
    home: RotoLineup


class RotowireClient:
    def __init__(self, creds: Credentials, timeout: int = 25) -> None:
        self.creds = creds
        self.timeout = timeout

    def available(self) -> bool:
        return self.creds.has_rotowire()

    # -- public daily lineups (no auth) -----------------------------------
    def fetch_expected_lineups(self, slate_date: Date | None = None) -> list[RotoGame]:
        """Scrape the public daily-lineups page into per-game lineups.

        Returns ``[]`` on any network/parse failure so the pipeline falls back
        to MLB Stats API lineups.
        """
        try:
            resp = requests.get(LINEUPS_URL, headers=_HEADERS, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Rotowire daily-lineups fetch failed: %s", exc)
            return []
        return parse_daily_lineups(resp.text)

    def load_lineups_csv(self, path: Path) -> pd.DataFrame:
        """Load a Rotowire expected-lineups CSV (fallback for unposted lineups)."""
        return pd.read_csv(path)

    def bullpen_availability(self, team_abbrev: str) -> float | None:
        """Return a team's bullpen availability (0..1, higher = more rested).

        Rotowire publishes reliever usage/availability (recent pitch counts, who
        is unavailable). Wired as the primary source for the bullpen fatigue
        (3-in-4) NPV; returns ``None`` until live access is validated, in which
        case the Statcast workload proxy is used instead.
        """
        if not self.available():
            return None
        log.info("Rotowire bullpen availability pending live access for %s", team_abbrev)
        return None

    def umpire_zone_runs(self, game_pk: int) -> float | None:
        """Return the plate umpire's zone bias for a game as a zone-runs value.

        Rotowire's umpire leaderboard (runs/game vs. the ~9.0 baseline) is the
        primary source for the ``umpire_zone_runs`` human-element input. Returns
        ``None`` until live access is validated so the umpire layer stays neutral.
        """
        if not self.available():
            return None
        log.info("Rotowire umpire assignment pending live access for game %s", game_pk)
        return None


# -- parsing helpers ------------------------------------------------------
_BOX_RE = re.compile(r'lineup__box')
_ABBR_RE = re.compile(r'lineup__abbr">([A-Z0-9]+)<')
_NICK_RE = re.compile(r'lineup__mteam is-(visit|home)">\s*([A-Za-z .\-]+?)\s*<')
_PITCHER_RE = re.compile(
    r'lineup__player-highlight-name">\s*<a [^>]*>([^<]+)</a>\s*'
    r'<span class="lineup__throws">([LRS])',
    re.S,
)
_BATTER_RE = re.compile(
    r'lineup__pos">([A-Z0-9]+)</div>\s*<a title="([^"]+)"[^>]*>[^<]*</a>\s*'
    r'<span class="lineup__bats">([LRS])',
    re.S,
)


def parse_daily_lineups(html: str) -> list[RotoGame]:
    """Parse the daily-lineups HTML into a list of ``RotoGame``."""
    games: list[RotoGame] = []
    starts = [m.start() for m in _BOX_RE.finditer(html)]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        chunk = html[start:end]
        abbrs = _ABBR_RE.findall(chunk)
        nicks = _NICK_RE.findall(chunk)
        if len(abbrs) < 2 or len(nicks) < 2:
            continue
        nick_by_side = {side: name for side, name in nicks}
        vi = chunk.find('lineup__list is-visit')
        hi = chunk.find('lineup__list is-home')
        if vi == -1 or hi == -1:
            continue
        away = _parse_side(chunk[vi:hi], abbrs[0], nick_by_side.get("visit", ""))
        home = _parse_side(chunk[hi:], abbrs[1], nick_by_side.get("home", ""))
        games.append(RotoGame(away=away, home=home))
    return games


def _parse_side(region: str, abbrev: str, nickname: str) -> RotoLineup:
    pm = _PITCHER_RE.search(region)
    pitcher = pm.group(1).strip() if pm else None
    throws = pm.group(2) if pm else None
    confirmed = "is-confirmed" in region
    batters = [
        RotoBatter(name=name.strip(), position=pos, bats=bats)
        for pos, name, bats in _BATTER_RE.findall(region)
    ]
    return RotoLineup(
        abbrev=abbrev,
        nickname=nickname,
        confirmed=confirmed,
        pitcher=pitcher,
        pitcher_throws=throws,
        batters=batters,
    )


def norm_person(s: str) -> str:
    """Normalize a player name for matching: strip accents, punctuation, suffix."""
    text = unicodedata.normalize("NFKD", str(s))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    parts = [p for p in text.split() if p not in _SUFFIXES]
    return " ".join(parts)
