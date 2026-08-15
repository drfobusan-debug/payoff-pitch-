"""The order the buys are listed in, and why it is not the edge.

The slate report listed buys "strongest first", meaning by edge -- our
disagreement with the devigged price. On 823 graded buys that ordering is
uninformative inside the band we allow and actively wrong outside it:

    edge 2-4%    n=134   win 47.0%   ROI  -9.4%
    edge 4-6%    n=233   win 48.9%   ROI  -8.3%
    edge 6-8%    n=192   win 47.9%   ROI  -6.5%
    edge 8%+     n=212   win 41.5%   ROI -14.7%   (capped, not bet)

Flat, then a cliff. So sorting by edge put the rows nearest the ceiling -- the
ones the cap exists to distrust -- at the top of the page and called them the
strongest. Expected value is worse still, because EV rises with the payout:

    ev 0-3%      n= 98   win 55.1%   ROI  -5.0%
    ev 6-10%     n=183   win 45.9%   ROI -15.2%
    ev 20%+      n=128   win 39.1%   ROI  -6.7%

What does order the record is the price:

    -167 or shorter   n= 83   win 63.9%   ROI  -3.9%
    -167 to +100      n=408   win 50.5%   ROI  -9.7%
    +100 to +160      n=231   win 38.5%   ROI -16.6%
    +160 to +300      n= 86   win 32.6%   ROI  -8.5%

which is the same finding as #52 and #120 from a different direction: the model
is over-confident, and over-confidence is worth the most money on the longest
prices. The +300-and-up cell is 15 bets and is ignored.

So buys are listed by conviction tier, then shortest price first, and edge is
shown as data rather than used as the ranking. None of this is a claim that
short prices are profitable -- every cell above loses -- only that the order a
reader's eye takes should not be the one the ledger says is worst.
"""

from __future__ import annotations

# Beyond this the price cells are too thin to order on (15 graded bets), so
# everything longer sorts together at the end.
LONGSHOT_AMERICAN = 300.0


def decimal_odds(american: float | None) -> float:
    """Decimal price, with an unpriced row sorted to the end."""
    if american is None:
        return float("inf")
    if american >= LONGSHOT_AMERICAN:
        return 1.0 + LONGSHOT_AMERICAN / 100.0
    return 1.0 + american / 100.0 if american > 0 else 1.0 + 100.0 / -american


def bet_sort_key(
    *, strong: bool, american: float | None, edge: float | None
) -> tuple[int, float, float]:
    """Conviction first, then the shortest price, then the edge as a tiebreak."""
    return (0 if strong else 1, decimal_odds(american), -(edge or 0.0))
