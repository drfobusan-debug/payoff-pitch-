"""Totals Unders when the offenses' yards per play run high, as graded-only rules.

Over FBS-vs-FBS games in weeks 2+ of 2019 and 2021-25 (n=4,126, CFBD lines,
team YPP through the week before each game), Unders landed 57.1% (n=294, final
-2.7 against the total, under the line in all six seasons) when both offenses
sat in the season's top YPP quartile, and 53.6% (n=1,033) in the top quartile of
the matchup index -- each offense's YPP averaged with the YPP its opponent's
defense allows. Both survived a Holm correction over the cuts tried; neither
clears -110 by enough to trust the size. The bottom YPP quartile leaned Over
rather than Under. Nothing here prices, tiers or refuses a bet: each rule
accrues on probation over every graded Under its games produce.
"""

from __future__ import annotations

from itertools import combinations
from statistics import quantiles

from cfb_engine.data.teamnames import label_matches, school_key

BOTH_TOP = "both_top"
MATCHUP_TOP = "matchup_top"
TAGS: tuple[str, ...] = (BOTH_TOP, MATCHUP_TOP)
RULE_NAMES: dict[str, str] = {
    BOTH_TOP: "ypp_both_offenses_top_q_under",
    MATCHUP_TOP: "ypp_matchup_top_q_under",
}
RATIONALES: dict[str, str] = {
    BOTH_TOP: "both offenses rank in the top quartile of yards per play",
    MATCHUP_TOP: "the offenses' YPP against the YPP their opponents allow is top-quartile",
}
LABEL = "YPP Under (graded only, not a bet)"
# Stamped on a judged game that earned no tag, so a blank cell still means "not stamped yet".
NONE = "-"

# team -> (offensive yards per play, yards per play allowed)
TeamYPP = dict[str, tuple[float, float]]


def team_ypp(rows: list[dict[str, object]]) -> TeamYPP:
    """Per-team YPP from CFBD ``/stats/season`` rows (one row per team and stat)."""
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        team, name, val = row.get("team"), row.get("statName"), row.get("statValue")
        if isinstance(team, str) and isinstance(name, str) and isinstance(val, (int, float)):
            stats.setdefault(team, {})[name] = float(val)
    out: TeamYPP = {}
    for team, s in stats.items():
        plays = s.get("rushingAttempts", 0.0) + s.get("passAttempts", 0.0)
        allowed = s.get("rushingAttemptsOpponent", 0.0) + s.get("passAttemptsOpponent", 0.0)
        if plays > 0 and allowed > 0 and "totalYards" in s and "totalYardsOpponent" in s:
            out[team] = (s["totalYards"] / plays, s["totalYardsOpponent"] / allowed)
    return out


def _q3(values: list[float]) -> float:
    return quantiles(values, n=4, method="inclusive")[2]


def matchup_index(home: tuple[float, float], away: tuple[float, float]) -> float:
    return (home[0] + away[1] + away[0] + home[1]) / 4


def cutoffs(book: TeamYPP) -> tuple[float, float] | None:
    """Top-quartile bars: one offense's YPP, and a game's matchup index over all pairings."""
    if len(book) < 4:
        return None
    pairs = [matchup_index(a, b) for a, b in combinations(book.values(), 2)]
    return _q3([v[0] for v in book.values()]), _q3(pairs)


def _lookup(label: str, book: TeamYPP) -> tuple[float, float] | None:
    key = school_key(label)
    for team, vals in book.items():
        if school_key(team) == key:
            return vals
    for team, vals in book.items():
        if label_matches(label, team):
            return vals
    return None


def teams(matchup: str) -> tuple[str, str] | None:
    for sep in (" @ ", " vs "):
        if sep in matchup:
            a, b = (s.strip() for s in matchup.split(sep, 1))
            return a, b
    return None


def tags(matchup: str, book: TeamYPP) -> str | None:
    """Space-separated tags the game earns; :data:`NONE` for none, None when it cannot be judged.

    An FCS side has no row, so the game is not judged rather than read as a miss.
    """
    pair = teams(matchup)
    bars = cutoffs(book)
    if pair is None or bars is None:
        return None
    a, b = (_lookup(t, book) for t in pair)
    if a is None or b is None:
        return None
    off_bar, mm_bar = bars
    out = []
    if min(a[0], b[0]) >= off_bar:
        out.append(BOTH_TOP)
    if matchup_index(a, b) >= mm_bar:
        out.append(MATCHUP_TOP)
    return " ".join(out) or NONE


def stamp(market: str, matchup: str, book: TeamYPP) -> str | None:
    """The tags for a totals row, or None off totals / without a judgement."""
    if market != "game_total" or not book:
        return None
    return tags(matchup, book)


def picks(tag: str, market: str, selection: str, stamped: str | None) -> bool:
    """An Under row whose game carries ``tag``."""
    return (
        market == "game_total" and selection.startswith("Under") and tag in (stamped or "").split()
    )
