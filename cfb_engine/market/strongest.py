"""The "strongest buy" flag: a shadow price at a near-market blend (graded only).

The 2021-25 era refit (``scripts/cfb/era_refit.py``) found the rating's
disagreement with the close carries ~1% +/- 8% of real signal, against the 65%
the shipped ``CFBE_MARKET_BLEND`` 0.35 gives it. The flag re-prices every buy
with the margin blended ``CFBE_SHADOW_BLEND`` (default 0.9) toward the market
and the rest, bye and travel points left out, and marks the buys that still
clear the same EV/edge gates. The engine's tiers, gates and stakes are
untouched; the flag only labels a subset of its buys so they can be graded.
"""

from __future__ import annotations

from dataclasses import replace

from cfb_engine.config import FeatureParams

LABEL = "Strongest Buy"
CELL = "STRONGEST"
WATCH_CELL = "WATCH (no bet)"
WATCH_LABEL = "Near-market watch (graded only)"
WATCH_NOTE = (
    "Rows the engine passes on at its own price that would clear the buy gates with "
    "the margin blended 90% to the market and rest, bye and travel left out. "
    "Tracked to grade the near-market price on its own; these are not bets and carry no stake."
)
CARD_NOTE = (
    "<b>BOLD CAPS</b> = strongest buy: still clears the buy gates with the margin "
    "blended 90% to the market and rest, bye and travel left out. Graded only; "
    "the stake is the same as any other buy."
)


def without_situational(params: FeatureParams) -> FeatureParams:
    """``params`` with the rest, bye and travel points zeroed (neutral-site HFA kept)."""
    return replace(params, rest_pts_per_day=0.0, bye_bonus_pts=0.0, travel_pts_per_1000mi=0.0)
