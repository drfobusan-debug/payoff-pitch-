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


def result_for(rec: Recommendation, index: ResultIndex) -> GameResult | None:
    if rec.home_abbrev is None or rec.away_abbrev is None:
        return None
    return index.get(frozenset({norm(rec.home_abbrev), norm(rec.away_abbrev)}))


def _team_points(rec: Recommendation, res: GameResult) -> tuple[int, int] | None:
    """(picked-team points, opponent points) resolving home/away by name."""
    if rec.team_side is None:
        return None
    home_is_pick = rec.team_side == "home"
    # The rec's home/away may be labeled opposite to CFBD's; align by name.
    rec_home = norm(rec.home_abbrev or "")
    if norm(res.home) == rec_home:
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
