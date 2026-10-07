"""Plate appearances off the Stats API play-by-play: who batted against whom.

A box score says what a hitter did in the game. The power screen's claim is
about one arm -- the starter it rated -- so grading it needs the plate
appearances split by the pitcher on the mound, which only the play-by-play has.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests

from mlb_engine.data import http

BASE = "https://statsapi.mlb.com/api/v1"

log = logging.getLogger(__name__)

HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALKS = frozenset({"walk", "intent_walk"})
STRIKEOUTS = frozenset({"strikeout", "strikeout_double_play", "strikeout_triple_play"})
#: Plate appearances that are not at-bats.
NOT_AT_BATS = WALKS | {
    "hit_by_pitch",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "catcher_interf",
}
#: A play that ends the inning on the bases ends no plate appearance.
_RUNNER_EVENTS = (
    "caught_stealing",
    "pickoff",
    "stolen_base",
    "wild_pitch",
    "passed_ball",
    "balk",
    "other_advance",
    "runner_double_play",
    "game_advisory",
    "other_out",
)


@dataclass(frozen=True)
class PlateAppearance:
    batter: int
    pitcher: int
    event: str
    #: ``top`` (the away side batting, the home starter pitching) or ``bottom``.
    half: str


@dataclass(frozen=True)
class Tally:
    """A batting line over some set of plate appearances."""

    pa: int = 0
    ab: int = 0
    h: int = 0
    tb: int = 0
    hr: int = 0
    bb: int = 0
    k: int = 0

    def __add__(self, other: Tally) -> Tally:
        return Tally(
            self.pa + other.pa,
            self.ab + other.ab,
            self.h + other.h,
            self.tb + other.tb,
            self.hr + other.hr,
            self.bb + other.bb,
            self.k + other.k,
        )

    @property
    def avg(self) -> float | None:
        return self.h / self.ab if self.ab else None

    @property
    def slg(self) -> float | None:
        return self.tb / self.ab if self.ab else None


def is_plate_appearance(event: str) -> bool:
    return bool(event) and not event.startswith(_RUNNER_EVENTS)


def tally(pas: list[PlateAppearance]) -> Tally:
    out = Tally()
    for pa in pas:
        if not is_plate_appearance(pa.event):
            continue
        bases = HITS.get(pa.event, 0)
        out = out + Tally(
            pa=1,
            ab=0 if pa.event in NOT_AT_BATS else 1,
            h=1 if bases else 0,
            tb=bases,
            hr=1 if pa.event == "home_run" else 0,
            bb=1 if pa.event in WALKS else 0,
            k=1 if pa.event in STRIKEOUTS else 0,
        )
    return out


def starters(pas: list[PlateAppearance]) -> dict[str, int]:
    """The first pitcher each side sent out: ``home`` pitches the top half."""
    out: dict[str, int] = {}
    for pa in pas:
        side = "home" if pa.half == "top" else "away"
        out.setdefault(side, pa.pitcher)
    return out


def parse(feed: dict) -> list[PlateAppearance]:
    out: list[PlateAppearance] = []
    for play in feed.get("allPlays", []) or []:
        result = play.get("result", {}) or {}
        matchup = play.get("matchup", {}) or {}
        about = play.get("about", {}) or {}
        batter = (matchup.get("batter", {}) or {}).get("id")
        pitcher = (matchup.get("pitcher", {}) or {}).get("id")
        if not batter or not pitcher or not about.get("isComplete", True):
            continue
        out.append(
            PlateAppearance(
                batter=int(batter),
                pitcher=int(pitcher),
                event=str(result.get("eventType", "") or ""),
                half=str(about.get("halfInning", "") or ""),
            )
        )
    return out


def _cache_path(cache_dir: Path | None, game_pk: int) -> Path | None:
    return None if cache_dir is None else Path(cache_dir) / "plays" / f"{game_pk}.json"


def fetch_plays(
    game_pk: int,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    timeout: int = 20,
    final: bool = True,
) -> list[PlateAppearance]:
    """Every completed plate appearance of a game.

    Cached under ``cache_dir/plays`` only when ``final`` -- a game still being
    played would otherwise be graded off its first innings forever.
    """
    path = _cache_path(cache_dir, game_pk)
    if path is not None and path.exists():
        try:
            return parse(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            log.warning("unreadable cached play-by-play %s; refetching", path)
    s = session or http.session()
    feed = s.get(f"{BASE}/game/{game_pk}/playByPlay", timeout=timeout).json()
    if path is not None and final:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"allPlays": feed.get("allPlays", [])}))
    return parse(feed)
