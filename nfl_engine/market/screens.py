"""Vetoes, and the tier that survives them.

Every screen here can only *remove* a bet, never create or promote one, and each
returns a short machine-readable name that is persisted on the row it rejected --
a screen nobody can grade is a superstition, and the MLB engine only found out
which of its gates removed winners once the rejections were on the ledger.

What the evidence says a screen has to do in this sport:

*The forecast is not the edge.* Phase 3, 3,450 games: the rating lost to the
closing spread and explained none of its residual (t +0.25). Priced through the
simulator off the market's own number, the moneyline came out at Brier 0.2126
against the closing moneyline's 0.2122 -- a dead heat -- and backing the
simulator's disagreement lost 2.5-4.0% flat over 3,017 games at every threshold
tested. So a model-versus-market disagreement is not evidence of a bet, and
``no_execution_edge`` demands a price better than the *de-vigged consensus on the
same line* before anything else is considered.

*Our own disagreement is a warning.* The same run put mean |sim - market| at
2.2pp with a maximum of 8.8pp. Beyond ``MAX_DISAGREEMENT`` the simulator is
almost certainly the one that is wrong, so ``model_disagrees`` removes the bet
even when the price beats consensus -- if our number is off, so is the fair number
we think we are beating.

*Long prices are where a small probability error is expensive.* Flat-betting
every closing moneyline 2006-2025 by price band (n=10,562 sides): +300 and longer
returned -10.78% (t -1.77) and -200..-150 returned -7.68% (t -3.47), with no
monotone pattern in between. Neither is a strong result on its own -- the honest
reading is that no band is profitable and the tails are worst -- but a 2pp error
on a +400 price costs five times what it costs at -150, so ``longshot`` caps the
price at ``MAX_AMERICAN``.

*An unpaired price is not a price.* Without the other side of the same line at
the same book there is no hold to remove, so ``unpaired`` and ``thin_market``
refuse to invent one. Assuming -110 is exactly the mistake the MLB engine paid
for on total bases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from nfl_engine.market.ev import MONEYLINE, PricedBet

# Minimum execution edge: the price must beat the consensus fair number by this
# much EV per unit. The median two-way hold on a closing NFL moneyline is 2.47%,
# so a book inside ~1.5c of consensus is inside the noise of which books happened
# to be in the sample rather than genuinely off the market.
MIN_EXECUTION_EV = 0.015
# Beyond this the simulator's departure from the consensus reads as our error.
# Mean |disagreement| against the closing moneyline was 0.022, max 0.088.
MAX_DISAGREEMENT = 0.060
# Price ceiling on the moneyline.
MAX_AMERICAN = 300.0
# Two independently paired books, or the "consensus" is one book's opinion.
MIN_PAIRED_BOOKS = 2
# Strong tier: double the minimum execution edge.
STRONG_EXECUTION_EV = 0.030


class Tier(str, Enum):
    STRONG = "Strong buy"
    MODERATE = "Moderate buy"
    PASS = "Pass"


@dataclass(frozen=True)
class Thresholds:
    min_execution_ev: float = MIN_EXECUTION_EV
    max_disagreement: float = MAX_DISAGREEMENT
    max_american: float = MAX_AMERICAN
    min_paired_books: int = MIN_PAIRED_BOOKS
    strong_execution_ev: float = STRONG_EXECUTION_EV


def screen(bet: PricedBet, thresholds: Thresholds | None = None) -> tuple[str, ...]:
    """Every veto this bet trips, in the order they are worth reading.

    All of them are reported rather than short-circuiting on the first, because a
    row rejected by three screens is a different animal from one that just missed
    the EV floor, and only the full list makes that visible later.
    """
    thr = thresholds or Thresholds()
    reasons: list[str] = []

    if bet.fair is None or bet.fair.paired_books == 0:
        reasons.append("unpaired")
    elif bet.fair.paired_books < thr.min_paired_books:
        reasons.append("thin_market")

    fair_ev = bet.ev_fair
    if fair_ev is None:
        # No trustworthy consensus: the only support left is the model, and phase
        # 3 says that is not support.
        reasons.append("model_only")
    elif fair_ev <= thr.min_execution_ev:
        reasons.append("no_execution_edge")

    if bet.ev_model <= 0.0:
        reasons.append("model_negative")

    disagreement = bet.edge_vs_fair
    if disagreement is not None and abs(disagreement) > thr.max_disagreement:
        reasons.append("model_disagrees")

    if bet.market == MONEYLINE and bet.american > thr.max_american:
        reasons.append("longshot")

    if bet.push_prob >= 1.0:
        reasons.append("certain_push")

    return tuple(reasons)


def apply_screens(
    bets: list[PricedBet], thresholds: Thresholds | None = None
) -> list[PricedBet]:
    """Stamp every bet with its vetoes. Nothing is dropped: rejections are rows."""
    return [replace(bet, screens=screen(bet, thresholds)) for bet in bets]


def tier_of(bet: PricedBet, thresholds: Thresholds | None = None) -> Tier:
    """Rank a surviving bet by execution edge, the only edge that is not a claim.

    Deliberately *not* ranked on ``ev_model``: EV rises with the price for a fixed
    edge, so an EV ranking quietly sorts by longshot, which is the ordering the
    MLB engine had to take its own best-bets list off in #150.
    """
    thr = thresholds or Thresholds()
    if bet.screens:
        return Tier.PASS
    fair_ev = bet.ev_fair
    if fair_ev is None:
        return Tier.PASS
    return Tier.STRONG if fair_ev >= thr.strong_execution_ev else Tier.MODERATE


def buys(bets: list[PricedBet], thresholds: Thresholds | None = None) -> list[PricedBet]:
    """Screened survivors, best execution edge first."""
    kept = [bet for bet in apply_screens(bets, thresholds) if bet.is_bet()]
    return sorted(kept, key=lambda b: -(b.ev_fair or 0.0))
