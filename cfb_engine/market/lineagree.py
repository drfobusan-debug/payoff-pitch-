"""Totals where the line has already moved toward the model's side.

A research rule, graded on probation and printed on the card and workbook, but
never a bet: no tier, gate or stake reads it. The 2026 ledger put the model's
totals side at 59.0% (n=39, +14.6%, se 15.5) when the line had moved at least a
point of no-vig probability its way before the bet, against 41.8% (n=55) when it
had moved away -- the right direction, inside two standard errors, and from six
slates. Every totals game the rule picks is graded, not only buys, so the sample
reaches the probation bar within a season.

Movement is read from the opener capture (``open_drift``) and falls back to the
day's first board (``drift``) for a game the opener did not see.
"""

from __future__ import annotations

RULE_NAME = "totals_line_agrees_1pt"
LABEL = "Line agrees (graded only, not a bet)"
# One no-vig probability point toward the model's side.
AGREE_FLOOR = 0.01


def movement(open_drift: float | None, drift: float | None) -> float | None:
    """Probability points the market moved toward this side before the bet."""
    return open_drift if open_drift is not None else drift


def line_agrees(
    market: str,
    model_prob: float,
    fair_prob: float | None,
    open_drift: float | None,
    drift: float | None,
) -> bool:
    """True on the model's side of a total whose line has come to it."""
    if market != "game_total" or fair_prob is None or model_prob <= fair_prob:
        return False
    moved = movement(open_drift, drift)
    return moved is not None and moved >= AGREE_FLOOR
