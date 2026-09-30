"""Market probabilities read off one joint simulation (master plan §4, §5.3).

``price`` maps an archived quote (engine market key, side, entity, line,
``ot_rule``) to the probability the sim gives that side, plus the push
probability where the line is a whole number. Because every market is read
from the same draws, the ML, puck line, totals and period markets of a game are
one consistent picture.

Settlement conventions carried here (book exceptions live in ``book_rules``):

* ``game_ml`` includes OT/SO; ``game_ml3`` is the regulation 3-way.
* ``game_pl`` grades on the official final, which credits the shootout winner
  one goal -- so at -1.5 a SO win never covers and a +1.5 dog never loses.
* totals / team totals follow ``ot_rule``: ``incl_ot_so`` counts the SO goal,
  ``incl_ot`` counts an OT goal but not the SO goal, ``reg_only`` neither.
  A ``book_rule`` still unresolved prices as ``incl_ot_so`` (the official
  final); the pipeline stamps the row ``settlement_unverified``.
* period markets grade on that period's goals only; a 2-way period ML pushes
  on a drawn period (probability returned conditional on no push).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from nhl_engine.models.periods import SimResult

_PERIOD = re.compile(r"^p([123])_(ml3|ml|pl|total|team_total)$")


@dataclass(frozen=True)
class Prob:
    win: float  # probability the side wins, conditional on no push
    push: float = 0.0

    @property
    def lose(self) -> float:
        return 1.0 - self.win


class UnpricedMarket(ValueError):
    """The market key/side combination has no reading in the sim."""


def _cond(win: np.ndarray, push: np.ndarray | None = None) -> Prob:
    p_push = float(push.mean()) if push is not None else 0.0
    p_win = float(win.mean())
    if p_push >= 1.0:
        return Prob(0.5, 1.0)
    return Prob(p_win / (1.0 - p_push), p_push)


def _totals(sim: SimResult, ot_rule: str) -> tuple[np.ndarray, np.ndarray]:
    if ot_rule == "reg_only":
        return sim.reg_h, sim.reg_a
    if ot_rule == "incl_ot":
        return sim.ot_goals_h, sim.ot_goals_a
    return sim.final_h, sim.final_a


def _over(total: np.ndarray, line: float, side: str) -> Prob:
    push = total == line
    if side == "over":
        return _cond(total > line, push)
    if side == "under":
        return _cond(total < line, push)
    raise UnpricedMarket(side)


def _spread(margin: np.ndarray, line: float) -> Prob:
    """P(margin + line > 0) for the side carrying ``line`` (e.g. -1.5 favourite)."""
    adj = margin + line
    return _cond(adj > 0, adj == 0)


def moneyline(sim: SimResult, home: bool) -> Prob:
    return Prob(float(sim.home_win.mean()) if home else 1.0 - float(sim.home_win.mean()))


def moneyline3(sim: SimResult, side: str, home: str, away: str) -> Prob:
    rh, ra = sim.reg_h, sim.reg_a
    if side == home:
        return Prob(float((rh > ra).mean()))
    if side == away:
        return Prob(float((ra > rh).mean()))
    if side == "draw":
        return Prob(float((rh == ra).mean()))
    raise UnpricedMarket(side)


def period_moneyline3(sim: SimResult, p: int, side: str, home: str, away: str) -> Prob:
    gh, ga = sim.ph[:, p - 1], sim.pa[:, p - 1]
    if side == home:
        return Prob(float((gh > ga).mean()))
    if side == away:
        return Prob(float((ga > gh).mean()))
    if side == "draw":
        return Prob(float((gh == ga).mean()))
    raise UnpricedMarket(side)


def price(
    sim: SimResult,
    *,
    market: str,
    side: str,
    entity: str,
    line: float | None,
    ot_rule: str,
    home: str,
    away: str,
) -> Prob:
    """Sim probability for one archived quote. Raises ``UnpricedMarket`` if unknown."""
    if market == "game_ml":
        if side not in (home, away):
            raise UnpricedMarket(side)
        return moneyline(sim, side == home)
    if market == "game_ml3":
        return moneyline3(sim, side, home, away)
    if market in ("game_pl", "game_pl_alt"):
        if line is None or side not in (home, away):
            raise UnpricedMarket(f"{market} {side} {line}")
        margin = sim.final_h - sim.final_a
        return _spread(margin if side == home else -margin, line)
    if market in ("game_total", "game_total_alt"):
        if line is None:
            raise UnpricedMarket(market)
        th, ta = _totals(sim, ot_rule)
        return _over(th + ta, line, side)
    if market in ("team_total", "team_total_alt"):
        if line is None or entity not in (home, away):
            raise UnpricedMarket(f"{market} {entity} {line}")
        th, ta = _totals(sim, ot_rule)
        return _over(th if entity == home else ta, line, side)

    m = _PERIOD.match(market)
    if m is None:
        raise UnpricedMarket(market)
    p, kind = int(m.group(1)), m.group(2)
    gh, ga = sim.ph[:, p - 1], sim.pa[:, p - 1]
    if kind == "ml3":
        return period_moneyline3(sim, p, side, home, away)
    if kind == "ml":
        if side not in (home, away):
            raise UnpricedMarket(side)
        win = (gh > ga) if side == home else (ga > gh)
        return _cond(win, gh == ga)
    if kind == "pl":
        if line is None or side not in (home, away):
            raise UnpricedMarket(f"{market} {side} {line}")
        margin = gh - ga
        return _spread(margin if side == home else -margin, line)
    if kind == "total":
        if line is None:
            raise UnpricedMarket(market)
        return _over(gh + ga, line, side)
    if line is None or entity not in (home, away):
        raise UnpricedMarket(f"{market} {entity} {line}")
    return _over(gh if entity == home else ga, line, side)


def summary(sim: SimResult) -> dict[str, float]:
    """Headline numbers for the card / brief."""
    th, ta = sim.final_h, sim.final_a
    return {
        "home_win": float(sim.home_win.mean()),
        "home_reg_win": float((sim.reg_h > sim.reg_a).mean()),
        "draw_reg": float(sim.went_ot.mean()),
        "went_so": float(sim.went_so.mean()),
        "exp_home": float(th.mean()),
        "exp_away": float(ta.mean()),
        "exp_total": float((th + ta).mean()),
        "sd_total": float((th + ta).std()),
        "exp_margin": float((th - ta).mean()),
        "home_cover_m15": float((th - ta > 1.5).mean()),
        "away_cover_p15": float((ta - th > -1.5).mean()),
        "p1_exp_total": float((sim.ph[:, 0] + sim.pa[:, 0]).mean()),
        "p2_exp_total": float((sim.ph[:, 1] + sim.pa[:, 1]).mean()),
        "p3_exp_total": float((sim.ph[:, 2] + sim.pa[:, 2]).mean()),
    }


__all__ = ["Prob", "UnpricedMarket", "moneyline", "moneyline3", "price", "summary"]
