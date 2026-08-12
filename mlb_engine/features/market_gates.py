"""Hard per-market screens that price and probability alone cannot express.

These run after the EV tiers and are deliberately blind to EV: they exist for
the markets where the graded card shows the model's edge estimate is itself the
thing that is wrong, so letting a big EV override them would reinstate exactly
the bets they remove.

Each returns ``(keep_buy, reason)`` and is neutral on missing inputs -- an
unpriced selection is already a Pass, and a gate that fires on absent data
would silently turn a data hole into a betting decision.
"""

from __future__ import annotations

import math


def price_band_allows(
    price: float | None,
    min_odds: float,
    max_odds: float,
    label: str = "price-band",
) -> tuple[bool, str]:
    """Whether a buy's price sits inside the band that has historically paid.

    Long prices are where a small absolute probability error becomes a large
    relative one: the model's edge on a +900 home run is a fraction of a point
    dressed up as a big number, and the card bears that out (three winners in 62
    bets at +500 or longer). Short prices fail the other way -- the payout stops
    covering the base rate.
    """
    if price is None:
        return True, ""
    if min_odds <= price <= max_odds:
        return True, ""
    bound = f"{min_odds:+.0f}..{max_odds:+.0f}" if math.isfinite(max_odds) else f"{min_odds:+.0f} or better"
    return False, f"{label}: PASS ({price:+.0f} outside {bound})"


def price_ceiling_allows(
    price: float | None, refuse_at: float, label: str = "price-ceiling"
) -> tuple[bool, str]:
    """Whether a buy is short enough in price to be worth taking at all.

    Unlike ``price_band_allows`` the ceiling is exclusive, because the rule it
    encodes is "at this price or longer, don't": road moneyline underdogs won
    28.6% of 77 bets and lost a third of stake, in both halves of the window.
    """
    if price is None:
        return True, ""
    if price < refuse_at:
        return True, ""
    return False, f"{label}: PASS ({price:+.0f} at or beyond {refuse_at:+.0f})"


def prob_floor_allows(
    prob: float | None, floor: float, label: str = "prob-floor"
) -> tuple[bool, str]:
    """Whether a buy clears a minimum model probability, whatever its EV.

    A conviction floor rather than a value test. It applies where cheap tickets
    were the whole loss: RBI overs below 40% cost 20.5 of the 21.5 units that
    market lost, while everything above it was roughly flat.
    """
    if prob is None or floor <= 0.0:
        return True, ""
    if prob >= floor:
        return True, ""
    return False, f"{label}: PASS (model {prob:.3f} under {floor:.2f})"
