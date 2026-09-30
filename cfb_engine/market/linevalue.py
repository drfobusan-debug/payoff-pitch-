"""Line movement in the one unit every market can be compared in.

The sibling MLB engine measures market movement in no-vig probability points,
because a baseball prop moves by *price*: the line is fixed at 1.5 total bases
and only the number beside it changes. Football does not work that way. A spread
re-centres itself -- when the market decides Alabama is half a point better it
moves -7 to -7.5 and leaves the price at -110 -- so the no-vig probability of a
side sits near 0.500 all week regardless of how far the market has actually
travelled. Measured in MLB's unit, a full point of spread movement looks like no
movement at all.

So the two axes are measured separately and then converted into one number:

* **the handicap** (ATS, totals), in points, signed from the perspective of the
  side being bet -- a home favourite laying 7 when the close is 7.5 gained half
  a point, an away dog taking 7 when the close is 7.5 lost half of one;
* **the price** (all three markets, and the only axis a moneyline has), in no-vig
  probability points.

Points are converted to probability with the local slope of the scoring
distribution, ``1 / (sd * sqrt(2*pi))`` -- the derivative of the normal CDF at
its centre, which is where a main-line bet sits by construction. At the engine's
margin SD of 16 that is 2.5 points of probability per point of spread, and at the
total SD of 13 it is 3.1 per point. The slope falls away in the tails, so this
overstates movement on a line bet far from the middle; main lines are what the
board quotes and what gets bet, and for those the centre is the right place to
linearise.

One subtraction, two uses, and the thing worth being careful about is that they
are the *same* measurement over different intervals rather than two signals:

* between the bet and the close it is closing-line value -- the number we hold
  against the number the market settled on (:mod:`cfb_engine.audit.clv`);
* between the first board and now it is pre-bet drift -- where the market's
  opinion has travelled while we were deciding (:mod:`cfb_engine.market.drift`).

Both are positive when the market has moved *toward* the side, and on the
handicap axis those are one quantity, not two: moving toward a side is exactly
what makes that side's number worse, so the earlier number is better by the
amount the market moved. A spread going -7 to -7.5 means the market came to
Alabama, and it means -7 was the better number to have held; the away +7 that
became +7.5 lost opinion and gained a point of value. Treating the two as
opposites is the mistake that leaves a movement signal pointing backwards.

MLB's ledger says this axis predicts, and that it does not predict monotonically
(small adverse moves were fine, large ones were not, and moves already in our
favour were the worst of all). So the drift is recorded on every priced side and
is not allowed to refuse a bet until the CFB ledger has graded rows to say which
tail, if any, is real -- see :mod:`cfb_engine.market.drift`.
"""

from __future__ import annotations

import math

#: Markets whose handicap moves (a moneyline has no number to move).
LINE_MARKETS = frozenset({"game_ats", "game_total"})


def prob_per_point(sd: float) -> float:
    """No-vig probability points gained per point of line, at the distribution's centre."""
    if sd <= 0:
        return 0.0
    return 1.0 / (sd * math.sqrt(2.0 * math.pi))


def value_points(
    market: str, side: str | None, held: float | None, other: float | None
) -> float | None:
    """Points of line value ``held`` has over ``other`` for this side (positive = better).

    Both arguments are that *side's own* handicap, which is the convention
    :mod:`cfb_engine.market.keys` already writes into a selection (the home spread
    for the home side, its negation for the away side, the total for both over and
    under). Equivalently: points the market moved toward the side going from
    ``held`` to ``other``.
    """
    if market not in LINE_MARKETS or held is None or other is None:
        return None
    if market == "game_ats":
        # Laying fewer points, or taking more, is the better number -- and both
        # fall out of the same subtraction once each side carries its own sign.
        return round(held - other, 2)
    if side == "over":
        return round(other - held, 2)
    if side == "under":
        return round(held - other, 2)
    return None


def line_drift_points(
    market: str, side: str | None, from_line: float | None, to_line: float | None
) -> float | None:
    """Points the market's own expectation moved *toward* this side over an interval.

    The same subtraction as :func:`value_points` read over time instead of against
    the close, and deliberately *not* negated: a market that moves toward a side
    is a market making that side's number worse, so the amount it moved and the
    advantage of the earlier number are one number.
    """
    return value_points(market, side, from_line, to_line)


def drift_probability(
    market: str,
    side: str | None,
    *,
    from_prob: float | None,
    to_prob: float | None,
    from_line: float | None = None,
    to_line: float | None = None,
    margin_sd: float,
    total_sd: float,
) -> float | None:
    """Total movement toward this side since ``from_*``, in no-vig probability points.

    Both axes are summed: the handicap (converted at the local slope) and the
    price at that handicap. On a main-line ATS/total bet the price axis is small
    and the handicap carries the signal; on a moneyline the handicap does not
    exist and the price is all there is.
    """
    total = 0.0
    seen = False
    pts = line_drift_points(market, side, from_line, to_line)
    if pts is not None:
        sd = margin_sd if market == "game_ats" else total_sd
        total += pts * prob_per_point(sd)
        seen = True
    if from_prob is not None and to_prob is not None:
        total += to_prob - from_prob
        seen = True
    return round(total, 4) if seen else None
