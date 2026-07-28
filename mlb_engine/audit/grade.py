"""Grade recommendations against final results (win / loss / push)."""

from __future__ import annotations

from mlb_engine.data.results import GameResult
from mlb_engine.recommendations import Recommendation

WIN, LOSS, PUSH = "win", "loss", "push"


def picked_margin(rec: Recommendation, res: GameResult) -> float | None:
    """Final margin from the picked side's perspective (team_runs - opp_runs).

    Defined only for run-line markets (``game_rl`` / ``f5_rl``); returns ``None``
    for every other market. Positive = the backed team won by that many; used by
    the audit run-line miss matrix to tell one-run-win errors from blowouts.
    """
    if not rec.team_side:
        return None
    if rec.market == "game_rl":
        team = res.home_runs if rec.team_side == "home" else res.away_runs
        opp = res.away_runs if rec.team_side == "home" else res.home_runs
        return float(team - opp)
    if rec.market == "f5_rl":
        team = res.f5_home if rec.team_side == "home" else res.f5_away
        opp = res.f5_away if rec.team_side == "home" else res.f5_home
        return float(team - opp)
    return None


def _ou(actual: float, line: float, side: str) -> str:
    if actual == line:
        return PUSH
    over = actual > line
    if side == "over":
        return WIN if over else LOSS
    return WIN if not over else LOSS


def grade(rec: Recommendation, res: GameResult) -> str | None:
    """Return 'win'|'loss'|'push', or None if not gradeable."""
    cat, market = rec.category, rec.market

    if market == "game_ml" and rec.team_side:
        team = res.home_runs if rec.team_side == "home" else res.away_runs
        opp = res.away_runs if rec.team_side == "home" else res.home_runs
        if team == opp:
            return PUSH
        return WIN if team > opp else LOSS

    if market == "f5_ml":
        if rec.side == "tie":
            return WIN if res.f5_home == res.f5_away else LOSS
        if rec.team_side:
            team = res.f5_home if rec.team_side == "home" else res.f5_away
            opp = res.f5_away if rec.team_side == "home" else res.f5_home
            if team == opp:
                return PUSH
            return WIN if team > opp else LOSS

    if market in ("game_total", "f5_total") and rec.line is not None and rec.side:
        total = (res.home_runs + res.away_runs) if market == "game_total" else (
            res.f5_home + res.f5_away
        )
        return _ou(total, rec.line, rec.side)

    if market in ("game_rl", "f5_rl") and rec.line is not None and rec.team_side:
        if market == "game_rl":
            team = res.home_runs if rec.team_side == "home" else res.away_runs
            opp = res.away_runs if rec.team_side == "home" else res.home_runs
        else:
            team = res.f5_home if rec.team_side == "home" else res.f5_away
            opp = res.f5_away if rec.team_side == "home" else res.f5_home
        adj = (team - opp) + rec.line
        if adj == 0:
            return PUSH
        return WIN if adj > 0 else LOSS

    if cat == "batter" and rec.player_id and rec.stat and rec.line is not None:
        if rec.stat == "HRR":
            b = res.batter(rec.player_id)
            actual = b.get("H", 0) + b.get("R", 0) + b.get("RBI", 0)
        elif rec.stat == "TB":
            b = res.batter(rec.player_id)
            actual = b.get("1B", 0) + 2 * b.get("2B", 0) + 3 * b.get("3B", 0) + 4 * b.get("HR", 0)
        else:
            actual = res.batter(rec.player_id).get(rec.stat, 0)
        return _ou(actual, rec.line, rec.side or "over")

    if cat == "pitcher" and rec.player_id and rec.stat and rec.line is not None:
        actual = res.pitcher(rec.player_id).get(rec.stat, 0)
        return _ou(actual, rec.line, rec.side or "over")

    return None
