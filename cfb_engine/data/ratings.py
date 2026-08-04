"""Assemble the team-rating book from all available sources, best first.

Priority (later sources override / blend earlier ones):

1. **CFBD SP+** -- the default spine (adjusted offense/defense in points).
2. **PFF CSV drop-in** (``~/.cfb_engine/pff/*.csv``) -- team offense/defense
   grades exported from a PFF subscription, converted to a points scale and
   blended with SP+ (or used alone when SP+ is unavailable).
3. **Local ratings CSV** (``~/.cfb_engine/ratings.csv``: ``team,off,def``) --
   a direct manual override on the points scale, applied last.

When nothing is available the caller falls back to market-implied ratings.
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from cfb_engine.data.cfbd import RatingBook, TeamRating
from cfb_engine.data.teamnames import norm

log = logging.getLogger(__name__)

# PFF grades run ~0-100 around a ~60 average. Convert a grade to a points offset
# from the league scoring average: (grade - 60) * pts_per_grade.
_PFF_GRADE_PIVOT = 60.0


def _pts_per_grade() -> float:
    raw = os.getenv("CFBE_PFF_PTS_PER_GRADE")
    return float(raw) if raw not in (None, "") else 0.45


def _pff_blend() -> float:
    """Weight on PFF when blending with CFBD (0 = ignore PFF, 1 = PFF only)."""
    raw = os.getenv("CFBE_PFF_BLEND")
    return float(raw) if raw not in (None, "") else 0.5


def _find_col(headers: Sequence[str], *needles: str) -> str | None:
    low = {h.lower().strip(): h for h in headers}
    for key, original in low.items():
        for needle in needles:
            if needle in key:
                return original
    return None


def _read_pff(path: Path, league_avg: float) -> dict[str, TeamRating]:
    """Best-effort parse of a PFF team-grade CSV into points-scale ratings."""
    out: dict[str, TeamRating] = {}
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            team_col = _find_col(headers, "team", "name", "school")
            off_col = _find_col(headers, "off_grade", "offense", "off", "grades_offense")
            def_col = _find_col(headers, "def_grade", "defense", "def", "grades_defense")
            if not team_col or not off_col or not def_col:
                log.warning("PFF CSV %s missing team/offense/defense columns; skipped", path.name)
                return out
            scale = _pts_per_grade()
            for row in reader:
                team = (row.get(team_col) or "").strip()
                if not team:
                    continue
                try:
                    off_grade = float(row[off_col])
                    def_grade = float(row[def_col])
                except (TypeError, ValueError):
                    continue
                offense = league_avg + (off_grade - _PFF_GRADE_PIVOT) * scale
                # Higher defensive grade = fewer points allowed.
                defense = league_avg - (def_grade - _PFF_GRADE_PIVOT) * scale
                out[norm(team)] = TeamRating(team, offense, defense)
    except OSError as exc:
        log.warning("could not read PFF CSV %s: %s", path, exc)
    return out


def _blend(a: TeamRating, b: TeamRating, w_b: float) -> TeamRating:
    return TeamRating(
        a.team,
        (1 - w_b) * a.offense + w_b * b.offense,
        (1 - w_b) * a.defense + w_b * b.defense,
    )


def _read_local(path: Path) -> dict[str, TeamRating]:
    out: dict[str, TeamRating] = {}
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            team_col = _find_col(headers, "team", "name", "school")
            off_col = _find_col(headers, "off")
            def_col = _find_col(headers, "def")
            if not team_col or not off_col or not def_col:
                log.warning("ratings CSV %s missing team/off/def columns; skipped", path.name)
                return out
            for row in reader:
                team = (row.get(team_col) or "").strip()
                if not team:
                    continue
                try:
                    out[norm(team)] = TeamRating(team, float(row[off_col]), float(row[def_col]))
                except (TypeError, ValueError):
                    continue
    except OSError as exc:
        log.warning("could not read ratings CSV %s: %s", path, exc)
    return out


def build_rating_book(base: RatingBook | None, pff_dir: Path, ratings_file: Path) -> RatingBook | None:
    """Overlay PFF and local CSV overrides on the CFBD base (any may be absent)."""
    ratings: dict[str, TeamRating] = dict(base.ratings) if base else {}
    league_avg = base.league_avg if base else 27.5

    pff: dict[str, TeamRating] = {}
    if pff_dir.exists():
        for path in sorted(pff_dir.glob("*.csv")):
            pff.update(_read_pff(path, league_avg))
    if pff:
        if not ratings:  # no CFBD: recentre league average on PFF itself
            league_avg = sum(r.offense for r in pff.values()) / len(pff)
        w = _pff_blend()
        for key, pr in pff.items():
            ratings[key] = _blend(ratings[key], pr, w) if key in ratings else pr

    if ratings_file.exists():
        for key, lr in _read_local(ratings_file).items():
            ratings[key] = lr

    if not ratings:
        return None
    return RatingBook(ratings=ratings, league_avg=league_avg)
