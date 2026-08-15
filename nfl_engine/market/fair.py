"""Stripping the hold off a two-way price, and forming a consensus fair number.

An EV number is only as good as the probability it is compared against, so the
de-vig is not a detail. Four methods, all measured against every closing
moneyline nflverse carries (n=5,281 non-push games, 2006-2025, median hold 2.47%)
by ``scripts/nfl/devig_study.py``:

    method         Brier     log loss
    proportional  0.21112    0.60946
    additive      0.21110    0.60932
    power         0.21112    0.60931
    shin          0.21110    0.60932

Indistinguishable, which is the honest headline -- at a 2.5% hold there is not
enough room between them for an aggregate score to separate them. The reason to
pick one is *where* each is wrong, which the same study measures as fair minus
realised by the favourite's booked price:

    favourite     n      realised   prop     addi     power    shin
    0.5-0.6     1,289     0.5407   +0.0051  +0.0064  +0.0070  +0.0064
    0.6-0.7     1,750     0.6131   +0.0181  +0.0216  +0.0233  +0.0216
    0.7-0.8     1,382     0.7279   -0.0008  +0.0054  +0.0086  +0.0054
    0.8-0.9       725     0.8303   -0.0123  -0.0037  +0.0010  -0.0037
    0.9-1.0       135     0.9185   -0.0231  -0.0109  -0.0033  -0.0109

**Proportional de-vig has a slope and power does not.** Dividing by the booked
sum takes the vig off in proportion to each side's price, so it lifts the
longshot and shaves the favourite -- a 2.3pp understatement of teams priced above
-900, monotone from +1.8pp in the middle. That is the favourite-longshot bias
being *created by the arithmetic*, and on the moneyline it is the difference
between calling a heavy favourite a buy and a fade. ``power`` is within 1pp of
realised in four of five buckets, so it is the default; the rest stay available
because the choice should be re-measured when there is a prop board to measure it
on, where the hold is 6-8% and the methods diverge much further.

One thing the table says about the market rather than the method: every method
overstates a modest favourite (0.6-0.7) by around 2pp on 1,750 games, realised
0.6131 against roughly 0.632 booked. That is a 1.8-sigma hint of a real bias, not
a signal to bet, and it is recorded here so the screens layer has somewhere to
start looking.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nfl_engine.market.board import BOOK_WEIGHTS, MarketQuote

DEFAULT_METHOD = "power"
# Below this the two prices are effectively already fair and every method agrees.
MIN_HOLD = 1e-9
BISECT_STEPS = 60


def hold(prices: Sequence[float]) -> float:
    """Booked overround: how much more than 1.0 the implied probabilities sum to."""
    return sum(prices) - 1.0


def _bisect(fn, low: float, high: float) -> float | None:
    """Plain bisection; scipy is not a dependency of this package."""
    f_low, f_high = fn(low), fn(high)
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        return None
    for _ in range(BISECT_STEPS):
        mid = (low + high) / 2.0
        f_mid = fn(mid)
        if f_mid == 0.0:
            return mid
        if f_low * f_mid < 0.0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def devig_proportional(implied: Sequence[float]) -> list[float]:
    total = sum(implied)
    if total <= 0:
        return list(implied)
    return [p / total for p in implied]


def devig_additive(implied: Sequence[float]) -> list[float]:
    """Take the same probability off every side, then clamp away from zero."""
    excess = hold(implied) / len(implied)
    out = [max(p - excess, 1e-6) for p in implied]
    total = sum(out)
    return [p / total for p in out]


def devig_power(implied: Sequence[float]) -> list[float]:
    """Raise every implied probability to the common exponent that sums to one."""
    if abs(hold(implied)) < MIN_HOLD:
        return list(implied)
    if min(implied) <= 0.0:
        return devig_proportional(implied)

    def excess(k: float) -> float:
        return sum(p**k for p in implied) - 1.0

    k = _bisect(excess, 0.2, 5.0)
    if k is None:
        return devig_proportional(implied)
    out = [p**k for p in implied]
    total = sum(out)
    return [p / total for p in out]


def devig_shin(implied: Sequence[float]) -> list[float]:
    """Shin's model: the hold is the book's defence against informed money."""
    total = sum(implied)
    if total <= 1.0 + MIN_HOLD:
        return devig_proportional(implied)

    def value(z: float, p: float) -> float:
        root = math.sqrt(z * z + 4.0 * (1.0 - z) * p * p / total)
        return (root - z) / (2.0 * (1.0 - z))

    def excess(z: float) -> float:
        return sum(value(z, p) for p in implied) - 1.0

    z = _bisect(excess, 1e-9, 0.4)
    if z is None:
        return devig_proportional(implied)
    out = [value(z, p) for p in implied]
    denom = sum(out)
    return [p / denom for p in out] if denom > 0 else devig_proportional(implied)


METHODS = {
    "proportional": devig_proportional,
    "additive": devig_additive,
    "power": devig_power,
    "shin": devig_shin,
}


def devig(implied: Sequence[float], method: str = DEFAULT_METHOD) -> list[float]:
    if method not in METHODS:
        raise ValueError(f"unknown de-vig method {method!r}; have {sorted(METHODS)}")
    return METHODS[method](list(implied))


@dataclass(frozen=True)
class FairPrice:
    """A consensus fair probability, and how much of the board it came from."""

    prob: float
    books: int
    paired_books: int
    median_hold: float
    method: str

    def is_trustworthy(self, *, min_paired: int = 2) -> bool:
        """Two independently paired books, or the number is one book's opinion."""
        return self.paired_books >= min_paired and 0.0 < self.prob < 1.0


def _book_weight(book: str) -> float:
    return BOOK_WEIGHTS.get(book.lower(), 1.0)


def fair_from_quotes(
    quotes: Iterable[MarketQuote], *, method: str = DEFAULT_METHOD
) -> FairPrice | None:
    """Weighted-average fair probability over every *paired* quote on a side.

    Unpaired quotes are counted but never priced: without the other side of the
    same line at the same book there is no hold to remove, and assuming one is
    how the MLB engine ended up grading total bases against a -110 that never
    existed. A side with no paired quote returns a :class:`FairPrice` that
    ``is_trustworthy`` rejects rather than a fabricated number.
    """
    quotes = list(quotes)
    if not quotes:
        return None
    weights: list[float] = []
    probs: list[float] = []
    holds: list[float] = []
    for quote in quotes:
        if not quote.paired or quote.opposite_american is None:
            continue
        implied = [
            1.0 / quote.decimal,
            1.0 / _decimal(quote.opposite_american),
        ]
        fair = devig(implied, method)[0]
        weights.append(_book_weight(quote.book))
        probs.append(fair)
        holds.append(hold(implied))
    if not probs:
        return FairPrice(
            prob=0.0,
            books=len(quotes),
            paired_books=0,
            median_hold=float("nan"),
            method=method,
        )
    total = sum(weights)
    blended = sum(p * w for p, w in zip(probs, weights, strict=True)) / total
    return FairPrice(
        prob=blended,
        books=len(quotes),
        paired_books=len(probs),
        median_hold=_median(holds),
        method=method,
    )


def _decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american >= 0 else 100.0 / abs(american))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
