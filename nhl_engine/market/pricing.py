"""EV, tier and gate stamps for one priced selection (master plan §5.9).

Nothing is filtered. Every selection the sim can read gets a ``Priced`` row with
its model probability, edge over the devigged consensus, EV at the best board
price, a tier, and the list of gates it failed -- an empty list is
``pass_gate``. Probation grading (``audit/``) uses the failed rows too, which
is how a gate earns the right to be loosened.

EV per unit at decimal odds ``d`` with push probability ``q`` and win
probability ``p`` (conditional on no push)::

    EV = (1 - q) x (p x (d - 1) - (1 - p))

Game markets (``GateParams.anchored_markets``) are priced off the sim blended
toward the devigged consensus (``anchor``); ``Priced.sim_prob`` keeps the sim's
own number for grading.

Tiers follow the CFB engine: a buy must clear the EV floor *at the best price*
and is then ranked on edge, with a ceiling -- past ``max_edge`` a disagreement
with the market reads as a model error, not a bigger bet.

Gates (all stamped, none silently applied):

``min_edge`` ``max_edge`` ``min_ev`` ``max_buy_odds`` ``stale_quote``
``one_way_quote`` ``settlement_unverified`` ``goalie_unconfirmed``
``probation`` (market not yet a live buy market).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine_common.odds import american_to_prob, prob_to_american
from nhl_engine.config import GateParams
from nhl_engine.market.board import Selection
from nhl_engine.models.markets import Prob

GOALIE_MARKETS = frozenset(
    {"game_ml", "game_ml3", "game_pl", "game_pl_alt", "team_total", "team_total_alt"}
)


class Tier(str, Enum):
    STRONG = "Strong buy"
    MODERATE = "Moderate buy"
    PASS = "Pass"


PROP_PREFIXES = ("sk_", "g_", "ags", "fgs", "lgs")


def is_prop(market: str) -> bool:
    return market.startswith(PROP_PREFIXES)


def family(market: str) -> str:
    if is_prop(market):
        return "prop"
    if market.startswith("p"):
        return "period"
    if "total" in market:
        return "total"
    if "pl" in market:
        return "pl"
    return "ml"


def anchor(model: Prob, consensus: float, weight: float) -> Prob:
    """``(1 - w) x model + w x consensus`` on the no-push win probability."""
    w = min(max(weight, 0.0), 1.0)
    return Prob((1.0 - w) * model.win + w * consensus, model.push)


def ev_per_unit(p: Prob, american: float) -> float:
    dec = 1.0 / american_to_prob(american)
    return (1.0 - p.push) * (p.win * (dec - 1.0) - (1.0 - p.win))


@dataclass(frozen=True)
class Priced:
    sel: Selection
    ot_rule: str
    model: Prob
    edge: float
    ev: float
    tier: Tier
    gates: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # The sim's own probability when ``model`` is market-anchored.
    sim_prob: float | None = None

    @property
    def pass_gate(self) -> bool:
        return not self.gates

    @property
    def is_buy(self) -> bool:
        return self.pass_gate and self.tier != Tier.PASS

    @property
    def model_american(self) -> float:
        p = min(max(self.model.win, 1e-4), 1 - 1e-4)
        return prob_to_american(p)

    def kelly(self) -> float:
        dec = 1.0 / american_to_prob(self.sel.best_american)
        b = dec - 1.0
        return max(0.0, (self.model.win * b - (1.0 - self.model.win)) / b)


def stamp(
    sel: Selection,
    model: Prob,
    *,
    ot_rule: str,
    thr: GateParams,
    quote_age_minutes: float,
    goalie_confirmed: bool,
    settlement_gate: str | None,
    sim_prob: float | None = None,
) -> Priced:
    edge = model.win - sel.consensus
    ev = ev_per_unit(model, sel.best_american)
    gates: list[str] = []
    reasons = [
        f"model {model.win:.3f} vs consensus {sel.consensus:.3f} edge {edge:+.3f} EV {ev:+.3f}"
    ]

    if ev <= thr.min_ev:
        gates.append("min_ev")
    if edge < thr.min_edge:
        gates.append("min_edge")
    elif edge > thr.max_edge:
        gates.append("max_edge")
    cap = thr.max_buy_odds.get(family(sel.market), 150.0)
    if sel.best_american > cap:
        gates.append("max_buy_odds")
    if quote_age_minutes > thr.max_quote_age_minutes:
        gates.append("stale_quote")
    if sel.one_way:
        gates.append("one_way_quote")
    if settlement_gate:
        gates.append(settlement_gate)
    if sel.market in GOALIE_MARKETS or (sel.market.startswith("p") and "total" not in sel.market):
        if not goalie_confirmed:
            gates.append("goalie_unconfirmed")
    if is_prop(sel.market):
        gates.append("research_only")
        if sel.market == "g_saves" and not goalie_confirmed:
            gates.append("goalie_unconfirmed")
    if sel.market not in thr.live_markets:
        gates.append("probation")

    if ev <= thr.min_ev or edge < thr.min_edge or edge > thr.max_edge:
        tier = Tier.PASS
    elif edge >= thr.min_edge + thr.strong_edge_gap:
        tier = Tier.STRONG
    else:
        tier = Tier.MODERATE
    if gates:
        reasons.append("gates: " + ",".join(gates))
    return Priced(sel, ot_rule, model, edge, ev, tier, gates, reasons, sim_prob)


__all__ = [
    "GOALIE_MARKETS",
    "PROP_PREFIXES",
    "Priced",
    "Tier",
    "anchor",
    "ev_per_unit",
    "family",
    "is_prop",
    "stamp",
]
