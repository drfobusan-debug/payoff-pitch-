"""Settle a priced selection against the official NHL result.

The ``GameResult`` carries per-period scores and how the game was decided, so
every market key the engine prices can be graded from the record of truth:

* ``game_ml`` on the official winner; ``game_ml3`` on the 60-minute score.
* ``game_pl`` on the official final (SO winner credited one goal).
* totals / team totals under the row's ``ot_rule``; ``dual_rule`` reports
  whether ``reg_only`` and ``incl_ot_so`` would grade differently -- the
  reconciliation flag the book-rule expiry leans on.
* ``pN_*`` on that period's goals; a 2-way period ML pushes on a tie.
"""

from __future__ import annotations

import re

from nhl_engine.schemas import GameResult

_PERIOD = re.compile(r"^p([123])_(ml3|ml|pl|total|team_total)$")
WIN, LOSS, PUSH = "win", "loss", "push"


def _ou(actual: int, line: float, side: str) -> str | None:
    if actual == line:
        return PUSH
    if side == "over":
        return WIN if actual > line else LOSS
    if side == "under":
        return WIN if actual < line else LOSS
    return None


def _spread(margin: int, line: float) -> str:
    adj = margin + line
    return PUSH if adj == 0 else (WIN if adj > 0 else LOSS)


def _totals(res: GameResult, ot_rule: str) -> tuple[int, int]:
    """``(home, away)`` goals that count under ``ot_rule``."""
    if ot_rule == "reg_only":
        return res.reg_home, res.reg_away
    if ot_rule == "incl_ot":
        ot_h = sum(p.home for p in res.periods if p.kind == "OT")
        ot_a = sum(p.away for p in res.periods if p.kind == "OT")
        return res.reg_home + ot_h, res.reg_away + ot_a
    return res.final_home, res.final_away


def settle(
    *,
    market: str,
    side: str,
    entity: str,
    line: float | None,
    ot_rule: str,
    res: GameResult,
) -> str | None:
    """``win`` | ``loss`` | ``push``, or ``None`` when the market cannot be graded."""
    if not res.is_final:
        return None
    home, away = res.home, res.away
    if market == "game_ml":
        if side not in (home, away):
            return None
        won = res.final_home > res.final_away
        return WIN if (side == home) == won else LOSS
    if market == "game_ml3":
        rh, ra = res.reg_home, res.reg_away
        if side == "draw":
            return WIN if rh == ra else LOSS
        if side == home:
            return WIN if rh > ra else LOSS
        if side == away:
            return WIN if ra > rh else LOSS
        return None
    if market in ("game_pl", "game_pl_alt"):
        if line is None or side not in (home, away):
            return None
        margin = res.final_home - res.final_away
        return _spread(margin if side == home else -margin, line)
    if market in ("game_total", "game_total_alt"):
        if line is None:
            return None
        th, ta = _totals(res, ot_rule)
        return _ou(th + ta, line, side)
    if market in ("team_total", "team_total_alt"):
        if line is None or entity not in (home, away):
            return None
        th, ta = _totals(res, ot_rule)
        return _ou(th if entity == home else ta, line, side)

    m = _PERIOD.match(market)
    if m is None:
        return None
    ps = res.period(int(m.group(1)))
    if ps is None:
        return None
    gh, ga, kind = ps.home, ps.away, m.group(2)
    if kind == "ml3":
        if side == "draw":
            return WIN if gh == ga else LOSS
        if side in (home, away):
            return WIN if ((gh > ga) if side == home else (ga > gh)) else LOSS
        return None
    if kind == "ml":
        if side not in (home, away):
            return None
        if gh == ga:
            return PUSH
        return WIN if ((gh > ga) if side == home else (ga > gh)) else LOSS
    if kind == "pl":
        if line is None or side not in (home, away):
            return None
        margin = gh - ga
        return _spread(margin if side == home else -margin, line)
    if kind == "total":
        return _ou(gh + ga, line, side) if line is not None else None
    if line is None or entity not in (home, away):
        return None
    return _ou(gh if entity == home else ga, line, side)


def dual_rule(*, market: str, side: str, entity: str, line: float | None, res: GameResult) -> bool:
    """True when regulation and OT/SO-inclusive grading disagree for a totals row."""
    if "total" not in market or market.startswith("p"):
        return False
    a = settle(market=market, side=side, entity=entity, line=line, ot_rule="reg_only", res=res)
    b = settle(market=market, side=side, entity=entity, line=line, ot_rule="incl_ot_so", res=res)
    return a is not None and b is not None and a != b


def pnl(outcome: str | None, american: float, stake: float = 1.0) -> float | None:
    if outcome is None:
        return None
    if outcome == PUSH:
        return 0.0
    if outcome == LOSS:
        return -stake
    return stake * (american / 100.0 if american > 0 else 100.0 / -american)


__all__ = ["LOSS", "PUSH", "WIN", "dual_rule", "pnl", "settle"]
