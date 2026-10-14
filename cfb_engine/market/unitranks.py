"""Top-25 units, as graded-only ATS and moneyline monitors.

Through 2026-10-13, sides with a top-25 SP+ defense or offense covered about
59% of spreads, but on *current* SP+, which already counts those games. These
ranks are CFBD's garbage-time-free PPA through the week before the slate, so a
game is never graded against a rank it helped set. Each unit and market accrues
as a candidate rule: the side has a top-25 unit and the opponent does not. Nothing
here prices, tiers or refuses a bet.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date as Date

from cfb_engine.data.cfbd import GameMeta
from cfb_engine.data.teamnames import label_matches, school_key

TOP_N = 25

# unit -> (path into a /stats/season/advanced row, higher-is-better)
_UNIT_PATHS: dict[str, tuple[tuple[str, ...], bool]] = {
    "off": (("offense", "ppa"), True),
    "def": (("defense", "ppa"), False),
    "pass_off": (("offense", "passingPlays", "ppa"), True),
    "rush_off": (("offense", "rushingPlays", "ppa"), True),
    "pass_def": (("defense", "passingPlays", "ppa"), False),
    "rush_def": (("defense", "rushingPlays", "ppa"), False),
}
UNITS: tuple[str, ...] = tuple(_UNIT_PATHS)
LABELS: dict[str, str] = {
    "off": "offense",
    "def": "defense",
    "pass_off": "passing offense",
    "rush_off": "rushing offense",
    "pass_def": "pass defense",
    "rush_def": "run defense",
}
MARKETS: dict[str, str] = {"game_ats": "ats", "game_ml": "ml"}

UnitRanks = dict[str, dict[str, int]]


def _value(row: Mapping[str, object], path: tuple[str, ...]) -> float | None:
    node: object = row
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return float(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def rank_teams(rows: list[dict[str, object]]) -> UnitRanks:
    """Team -> unit -> national rank (1 = best); ties share the better rank."""
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        team = row.get("team")
        if not isinstance(team, str) or not team:
            continue
        vals = {u: v for u, (p, _) in _UNIT_PATHS.items() if (v := _value(row, p)) is not None}
        if vals:
            values[team] = vals
    out: UnitRanks = {team: {} for team in values}
    for unit, (_, higher_better) in _UNIT_PATHS.items():
        pool = [v[unit] for v in values.values() if unit in v]
        for team, vals in values.items():
            if unit in vals:
                mine = vals[unit]
                better = sum(1 for v in pool if (v > mine if higher_better else v < mine))
                out[team][unit] = 1 + better
    return out


def prior_week(schedule: list[GameMeta], day: Date) -> int:
    """The last regular-season week finished before ``day``'s slate (0 = none)."""
    target = day.isoformat()
    weeks = [
        m.week
        for m in schedule
        if m.week is not None and m.season_type == "regular" and m.start_date[:10] >= target
    ]
    if weeks:
        return max(min(weeks) - 1, 0)
    played = [m.week for m in schedule if m.week is not None and m.season_type == "regular"]
    return max(played, default=0)


def encode(ranks: Mapping[str, int]) -> str:
    return " ".join(f"{u}:{ranks[u]}" for u in UNITS if u in ranks)


def decode(text: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (text or "").split():
        unit, _, rank = part.partition(":")
        if unit in _UNIT_PATHS and rank.isdigit():
            out[unit] = int(rank)
    return out


def side_team(market: str, selection: str) -> str | None:
    """The team a spread (``Navy -1.5``) or moneyline (``Navy ML``) row backs."""
    name, _, last = selection.strip().rpartition(" ")
    if not name:
        return None
    if market == "game_ml":
        return name if last == "ML" else None
    if market == "game_ats":
        try:
            float(last)
        except ValueError:
            return None
        return name
    return None


def opponent(matchup: str, team: str) -> str | None:
    for sep in (" @ ", " vs "):
        if sep in matchup:
            a, b = (s.strip() for s in matchup.split(sep, 1))
            if a == team:
                return b
            if b == team:
                return a
    return None


def lookup(label: str, ranks: UnitRanks) -> dict[str, int]:
    key = school_key(label)
    for team, units in ranks.items():
        if school_key(team) == key:
            return units
    for team, units in ranks.items():
        if label_matches(label, team):
            return units
    return {}


def stamp(market: str, selection: str, matchup: str, ranks: UnitRanks) -> tuple[str, str] | None:
    """Encoded ranks for the side and its opponent; None off ATS/ML or with no ranks.

    An opponent with no row (an FCS school) stamps empty, which reads as unranked.
    """
    if not ranks or market not in MARKETS:
        return None
    team = side_team(market, selection)
    if team is None:
        return None
    opp = opponent(matchup, team)
    return encode(lookup(team, ranks)), encode(lookup(opp, ranks)) if opp else ""


def rule_name(unit: str, market: str) -> str:
    return f"{unit}_top25_{MARKETS[market]}"


def picks(unit: str, market: str, row_market: str, mine: str | None, opp: str | None) -> bool:
    """A ``market`` row whose side ranks top-25 in ``unit`` against an opponent outside it."""
    if row_market != market or mine is None:
        return False
    return decode(mine).get(unit, TOP_N + 1) <= TOP_N < decode(opp).get(unit, TOP_N + 1)
