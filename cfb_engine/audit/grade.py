"""Grade recommendations against final scores (win / loss / push).

Recommendations carry short school codes (from the odds board) while CFBD
returns full school names; both normalize to the same key, so
:func:`build_result_index` maps a game to its final score and :func:`grade`
reads each market off it.
"""

from __future__ import annotations

from cfb_engine.data.cfbd import GameResult
from cfb_engine.data.teamnames import norm
from cfb_engine.recommendations import Recommendation

WIN, LOSS, PUSH = "win", "loss", "push"

ResultIndex = dict[frozenset[str], GameResult]


def build_result_index(results: list[GameResult]) -> ResultIndex:
    """Index final scores by the unordered pair of normalized team names."""
    index: ResultIndex = {}
    for res in results:
        index[frozenset({norm(res.home), norm(res.away)})] = res
    return index


def same_team(rec_name: str, res_name: str) -> bool:
    """Whether a card label and a CFBD school name are the same school.

    The label a recommendation carries is a *display* label, capped at 14
    characters (:func:`cfb_engine.data.teamnames.short_code`), so ``New Mexico
    State`` reaches the audit as ``New Mexico Sta``. An exact key comparison
    silently drops those games from grading, which is worse than a wrong grade
    because nothing announces it -- hence the prefix.
    """
    left, right = norm(rec_name), norm(res_name)
    return left == right or right.startswith(left)


def result_for(rec: Recommendation, index: ResultIndex) -> GameResult | None:
    if rec.home_abbrev is None or rec.away_abbrev is None:
        return None
    home, away = norm(rec.home_abbrev), norm(rec.away_abbrev)
    exact = index.get(frozenset({home, away}))
    if exact is not None:
        return exact
    # A truncated label matches on prefix, but only if exactly one game answers
    # to it: both teams must agree, and an ambiguous pair is left ungraded
    # rather than graded against a guess.
    found = [
        res
        for res in index.values()
        if (same_team(rec.home_abbrev, res.home) and same_team(rec.away_abbrev, res.away))
        or (same_team(rec.home_abbrev, res.away) and same_team(rec.away_abbrev, res.home))
    ]
    return found[0] if len(found) == 1 else None


def _team_points(rec: Recommendation, res: GameResult) -> tuple[int, int] | None:
    """(picked-team points, opponent points) resolving home/away by name."""
    if rec.team_side is None:
        return None
    home_is_pick = rec.team_side == "home"
    # The rec's home/away may be labeled opposite to CFBD's; align by name.
    if same_team(rec.home_abbrev or "", res.home):
        home_pts, away_pts = res.home_points, res.away_points
    else:
        home_pts, away_pts = res.away_points, res.home_points
    return (home_pts, away_pts) if home_is_pick else (away_pts, home_pts)


def _ou(actual: float, line: float, side: str) -> str:
    if actual == line:
        return PUSH
    over = actual > line
    if side == "over":
        return WIN if over else LOSS
    return WIN if not over else LOSS


def grade(rec: Recommendation, res: GameResult) -> str | None:
    """Return 'win' | 'loss' | 'push', or None if the market is not gradeable."""
    if rec.market == "game_ml":
        pts = _team_points(rec, res)
        if pts is None:
            return None
        team, opp = pts
        if team == opp:
            return PUSH
        return WIN if team > opp else LOSS

    if rec.market == "game_ats" and rec.line is not None:
        pts = _team_points(rec, res)
        if pts is None:
            return None
        team, opp = pts
        adj = (team - opp) + rec.line
        if adj == 0:
            return PUSH
        return WIN if adj > 0 else LOSS

    if rec.market == "game_total" and rec.line is not None and rec.side:
        total = res.home_points + res.away_points
        return _ou(total, rec.line, rec.side)

    return None
