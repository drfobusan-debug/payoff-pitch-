"""How the card is ordered -- one ordering, in one place, and price-aware.

The order is not cosmetic. It decides which bet is read first, which five make
the narration's headline plays, and which rows the Excel gradient paints
brightest, so whatever it ranks on is what gets bet biggest.

Three candidate rankings, and the arithmetic matters:

* **EV.** ``EV = (decimal - 1) x p - q``, so at a fixed edge the EV rises with
  the price. Under flat stakes that is the correct ordering *if the model's
  probabilities are right*; the MLB engine measured it inverted (AUC 0.33 on
  moneylines), because a long price multiplies model error as surely as it
  multiplies edge. The Excel workbook ranked buys on EV until this module
  existed, which is why the workbook's brightest rows were its longest prices.
* **Edge.** Price-blind: 3 points of edge at -140 and at +450 rank equal.
  Honest, and what the tiers already rank on, but it declines to use the one
  thing known for certain about a bet -- what it pays.
* **Kelly.** ``f* = EV / (decimal - 1) = (p - breakeven) / (1 - breakeven)``, the
  growth-optimal fraction. Same edge, and it prefers the shorter price: the
  divisor shrinks as the break-even rises. That is the price-length penalty EV
  lacks and edge ignores, without a fitted constant anywhere in it.

Kelly is the default, and this is a *reading order*, not a screen: no bet is
added or removed by it, and ``CFBE_CARD_ORDER=edge`` or ``=ev`` restores either
of the others. Passes and fades are ranked on how far the model is *below* the
market instead, since a rejected side has no stake for Kelly to size.
"""

from __future__ import annotations

import os

from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation

KELLY = "kelly"
EDGE = "edge"
EV = "ev"
MODES = (KELLY, EDGE, EV)

_TIER_RANK = {Tier.STRONG: 0, Tier.MODERATE: 1, Tier.PASS: 2}


def order_mode() -> str:
    """``CFBE_CARD_ORDER``, falling back to Kelly on anything unrecognised."""
    mode = os.getenv("CFBE_CARD_ORDER", KELLY).strip().lower()
    return mode if mode in MODES else KELLY


def conviction(rec: Recommendation, mode: str | None = None) -> float:
    """How strongly a row is held, on the configured ranking.

    A Pass is scored on the model's disagreement *against* the side, so the
    fades sheet leads with what the model most dislikes rather than with
    whichever refused row happens to carry the longest price.
    """
    if rec.tier == Tier.PASS:
        if rec.fair_prob is not None:
            return max(0.0, rec.fair_prob - rec.model_prob)
        return max(0.0, -(rec.ev or 0.0))
    chosen = mode or order_mode()
    if chosen == EDGE:
        return rec.edge or 0.0
    if chosen == EV:
        return rec.ev or 0.0
    kelly = rec.kelly
    # An unpriced buy cannot be Kelly-sized; fall back to its edge rather than
    # sinking it to the bottom of a sheet it qualified for.
    return kelly if kelly is not None else (rec.edge or 0.0)


def sort_key(rec: Recommendation, mode: str | None = None) -> tuple[int, float]:
    """Tier first, then conviction descending."""
    return _TIER_RANK.get(rec.tier, 3), -conviction(rec, mode)


def order_recs(recs: list[Recommendation], mode: str | None = None) -> list[Recommendation]:
    return sorted(recs, key=lambda r: sort_key(r, mode))


def order_buys(recs: list[Recommendation], mode: str | None = None) -> list[Recommendation]:
    """The buys only, best first -- the card's and the narration's running order."""
    return order_recs([r for r in recs if r.tier != Tier.PASS], mode)
