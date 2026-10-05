"""Teammate math: split a team quantity across its players, inside their ceilings.

Player props are not independent: the rebounds a center grabs are rebounds the
point guard cannot. So a simulated game first fixes the team's totals (points,
FGA, 3PA, made shots, rebound chances) and then hands each one out by the
players' shares (``docs/nba/master_plan.md`` §4.5a). Shares alone break down
when two high-usage players are out: dividing their vacant usage pro rata can
push a low-usage defender to a shot count he has never taken. Each player
therefore carries a ceiling (his archetype's historical maximum, fitted), and
the split is water-filled: a player at his ceiling is frozen and the rest of
the total keeps flowing to the others by their shares. Whatever nobody can
absorb is returned as ``unabsorbed`` -- the caller turns it into team turnovers
or lower efficiency, never into an impossible player line.

    split = allocate(40.0, {"C": 0.30, "PF": 0.20, "PG": 0.10, ...})
    sum(split.amounts.values()) + split.unabsorbed == 40.0
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

_EPS = 1e-12


@dataclass(frozen=True)
class Allocation:
    amounts: dict[str, float]
    unabsorbed: float
    capped: frozenset[str]


def allocate(
    total: float,
    weights: Mapping[str, float],
    ceilings: Mapping[str, float] | None = None,
) -> Allocation:
    """Split ``total`` by ``weights`` (any scale), no player above his ceiling.

    Players with zero weight or a zero ceiling get nothing; a missing ceiling
    is unbounded. The amounts plus ``unabsorbed`` always equal ``total``.
    """
    if total < 0 or not math.isfinite(total):
        raise ValueError(f"total must be finite and non-negative, got {total}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative")
    caps = dict(ceilings or {})
    amounts = {p: 0.0 for p in weights}
    open_ = {p for p, w in weights.items() if w > 0 and caps.get(p, math.inf) > 0}
    capped: set[str] = {p for p in weights if p not in open_ and caps.get(p, math.inf) <= 0}
    left = total
    while left > _EPS and open_:
        wsum = sum(weights[p] for p in open_)
        full = {p for p in open_ if left * weights[p] / wsum >= caps.get(p, math.inf) - amounts[p]}
        if not full:
            for p in open_:
                amounts[p] += left * weights[p] / wsum
            left = 0.0
            break
        for p in full:
            room = caps[p] - amounts[p]
            amounts[p] = caps[p]
            left -= room
        open_ -= full
        capped |= full
    return Allocation(amounts=amounts, unabsorbed=max(left, 0.0), capped=frozenset(capped))


def vacate(weights: Mapping[str, float], out: set[str]) -> dict[str, float]:
    """The survivors' shares once ``out`` are ruled out (their share is re-split)."""
    return {p: w for p, w in weights.items() if p not in out}


__all__ = ["Allocation", "allocate", "vacate"]
