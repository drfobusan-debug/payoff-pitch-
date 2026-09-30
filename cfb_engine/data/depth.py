"""Where an injured player sits on his team's depth chart, inferred from usage.

Nobody publishes a machine-readable FBS depth chart (ESPN's depth-chart API
rejects college football; RotoWire's injury feed carries no rank), so the card
derives one from who has actually played: CFBD's ``/player/usage`` gives every
skill player's share of his team's snaps, and ``/stats/player/season`` (defensive
category) gives every defender's tackles. Ranking a team's players within a
position by that number and reading the rank against how many of that position
start (one QB, three WR, four DB...) says whether the man listed Out was a
starter, the next man up, or a reserve.

Offensive linemen and specialists have no usage stat, so they come back
``None`` and the card prints the name without a role rather than guessing.
Early in a season a player hurt before he took a snap is missing too, so the
caller layers last season's book underneath (:func:`merge_depth`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cfb_engine.data.teamnames import school_key

# How many of each position start in a typical alignment. A rank inside the
# count is a starter; the next ``count`` ranks are the second unit.
STARTERS: dict[str, int] = {
    "QB": 1,
    "RB": 1,
    "FB": 1,
    "WR": 3,
    "TE": 1,
    "DL": 4,
    "DE": 2,
    "DT": 2,
    "EDGE": 2,
    "NT": 1,
    "LB": 3,
    "ILB": 2,
    "OLB": 2,
    "DB": 4,
    "CB": 2,
    "S": 2,
    "SAF": 2,
}
STARTER = "starter"
SECOND = "2nd on depth chart"
RESERVE = "reserve"


@dataclass(frozen=True)
class DepthSlot:
    position: str
    rank: int  # 1 = most used at the position on this team
    starters: int  # how many of this position start

    @property
    def role(self) -> str:
        if self.rank <= self.starters:
            return STARTER
        if self.rank <= 2 * self.starters:
            return SECOND
        return RESERVE


# school_key -> name key -> slot
DepthBook = dict[str, dict[str, DepthSlot]]


def name_key(name: str) -> str:
    """Lower-case letters only; strips suffixes/punctuation so feeds agree."""
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name.lower())
    return re.sub(r"[^a-z]", "", s)


def _rows_to_book(rows: list[tuple[str, str, str, float]]) -> DepthBook:
    """``(team, position, player, load)`` rows -> ranked book."""
    by_group: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for team, pos, player, load in rows:
        by_group.setdefault((school_key(team), pos.upper()), []).append((load, player))
    book: DepthBook = {}
    for (team, pos), players in by_group.items():
        players.sort(key=lambda p: -p[0])
        starters = STARTERS.get(pos, 1)
        for rank, (_, player) in enumerate(players, start=1):
            book.setdefault(team, {}).setdefault(
                name_key(player), DepthSlot(position=pos, rank=rank, starters=starters)
            )
    return book


def parse_usage(data: object) -> list[tuple[str, str, str, float]]:
    """``/player/usage`` rows -> ``(team, position, player, overall usage)``."""
    out: list[tuple[str, str, str, float]] = []
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        team, pos, name = row.get("team"), row.get("position"), row.get("name")
        usage = row.get("usage")
        share = usage.get("overall") if isinstance(usage, dict) else None
        if (
            isinstance(team, str)
            and isinstance(pos, str)
            and isinstance(name, str)
            and isinstance(share, (int, float))
        ):
            out.append((team, pos, name, float(share)))
    return out


def parse_tackles(data: object) -> list[tuple[str, str, str, float]]:
    """``/stats/player/season?category=defensive`` rows -> total tackles per defender."""
    out: list[tuple[str, str, str, float]] = []
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict) or row.get("statType") != "TOT":
            continue
        team, pos, name, stat = (
            row.get("team"),
            row.get("position"),
            row.get("player"),
            row.get("stat"),
        )
        if not (isinstance(team, str) and isinstance(pos, str) and isinstance(name, str)):
            continue
        try:
            tackles = float(str(stat))
        except ValueError:
            continue
        out.append((team, pos, name, tackles))
    return out


def build_depth_book(usage: object, defensive: object) -> DepthBook:
    return _rows_to_book(parse_usage(usage) + parse_tackles(defensive))


def merge_depth(primary: DepthBook, fallback: DepthBook) -> DepthBook:
    """``primary`` with players it lacks filled from ``fallback`` (same team only)."""
    out: DepthBook = {team: dict(players) for team, players in fallback.items()}
    for team, players in primary.items():
        out.setdefault(team, {}).update(players)
    return out


def depth_for(book: DepthBook, team: str, player: str) -> DepthSlot | None:
    return book.get(school_key(team), {}).get(name_key(player))
