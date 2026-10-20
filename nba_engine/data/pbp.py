"""Per-half possessions and points from an ESPN summary's play-by-play.

Possessions use the box estimate per half, ``FGA - OREB + TOV + 0.44 * FTA``,
with the play types ESPN uses: a free throw is a shooting play whose type
starts "Free Throw" (technical and flagrant shots are left out, as they end no
possession); an offensive rebound counts only when credited to a player (a
team rebound after a first free throw is not a new possession); a turnover is
any type naming one, plus "Traveling" and "Offensive Charge". Half 1 is
periods 1-2, half 2 is periods 3-4; overtime is left out.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

TURNOVER_TYPES = frozenset({"Traveling", "Offensive Charge"})


@dataclass(frozen=True)
class HalfLine:
    side: str  # home | away
    half: int  # 1 | 2
    possessions: float
    points: int


def _side_by_team(summary: dict) -> dict[str, str]:
    comps = (summary.get("header") or {}).get("competitions") or [{}]
    return {
        str(c["team"]["id"]): c["homeAway"]
        for c in comps[0].get("competitors", [])
        if c.get("team", {}).get("id") and c.get("homeAway") in ("home", "away")
    }


def halves(summary: dict) -> list[HalfLine]:
    """Both teams' regulation halves; empty when the summary has no play-by-play."""
    side_of = _side_by_team(summary)
    plays = summary.get("plays") or []
    if not plays or len(side_of) != 2:
        return []
    count: dict[tuple[str, int, str], float] = defaultdict(float)
    for p in plays:
        period = int((p.get("period") or {}).get("number") or 0)
        side = side_of.get(str((p.get("team") or {}).get("id")))
        if not 1 <= period <= 4 or side is None:
            continue
        half = 1 if period <= 2 else 2
        kind = str((p.get("type") or {}).get("text") or "")
        if p.get("shootingPlay"):
            if not kind.startswith("Free Throw"):
                count[(side, half, "fga")] += 1
            elif "Technical" not in kind and "Flagrant" not in kind:
                count[(side, half, "fta")] += 1
        elif kind == "Offensive Rebound":
            if p.get("participants"):
                count[(side, half, "oreb")] += 1
        elif ("Turnover" in kind and kind != "No Turnover") or kind in TURNOVER_TYPES:
            count[(side, half, "tov")] += 1
        if p.get("scoringPlay"):
            count[(side, half, "pts")] += float(p.get("scoreValue") or 0)
    return [
        HalfLine(
            side=side,
            half=half,
            possessions=count[(side, half, "fga")]
            - count[(side, half, "oreb")]
            + count[(side, half, "tov")]
            + 0.44 * count[(side, half, "fta")],
            points=int(count[(side, half, "pts")]),
        )
        for side in ("away", "home")
        for half in (1, 2)
    ]
