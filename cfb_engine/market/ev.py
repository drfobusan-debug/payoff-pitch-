"""Expected-value calculation against market prices."""

from __future__ import annotations

from dataclasses import dataclass

from cfb_engine.market.odds import american_to_decimal, american_to_prob

# Pinnacle is the sharpest book widely carried by The Odds API; weight its
# line heavier than the recreational books when forming the consensus fair
# price. Any book not listed defaults to 1.0.
BOOK_WEIGHTS = {"pinnacle": 2.0, "lowvig": 1.5, "betonlineag": 1.2}
_DEFAULT_WEIGHT = 1.0


@dataclass
class MarketQuote:
    book: str
    american: float
    # Same book, same line, other side (under / opposing team). Required to strip
    # the vig: one side's implied probability carries roughly half the book's
    # overround.
    opposite_american: float | None = None

    @property
    def no_vig_prob(self) -> float:
        """The book's fair probability for this side, vig removed if possible."""
        p = american_to_prob(self.american)
        if self.opposite_american is None:
            return p
        q = american_to_prob(self.opposite_american)
        total = p + q
        return p / total if total > 0 else p

    @property
    def devigged(self) -> bool:
        return self.opposite_american is not None


@dataclass
class EVResult:
    model_prob: float
    best_quote: MarketQuote
    decimal: float
    ev: float  # EV per $1 staked
    fair_prob: float  # market no-vig implied
    edge: float  # model_prob - fair_prob
    # Share of consensus weight whose vig was actually removed. Below 1.0 the
    # edge is understated, so a thin-edge guard is stricter than it looks.
    devig_coverage: float = 1.0


def anchor_to_market(model_prob: float, fair_prob: float, weight: float) -> float:
    """Shrink the model toward the market's no-vig price by ``weight``.

    ``weight`` 0 leaves the model untouched, 1 bets the market itself. Because
    both screening criteria are affine in the probability, a weight ``w`` scales
    the measured edge to ``(1 - w) * (model - fair)`` -- arithmetically identical
    to demanding ``edge >= threshold / (1 - w)``. It keeps the model's biggest
    disagreements with the market and drops the small ones.
    """
    w = min(max(weight, 0.0), 1.0)
    return (1.0 - w) * model_prob + w * fair_prob


def ev_per_dollar(model_prob: float, american: float) -> float:
    dec = american_to_decimal(american)
    return model_prob * (dec - 1.0) - (1.0 - model_prob)


def _weight(book: str) -> float:
    return BOOK_WEIGHTS.get(book, _DEFAULT_WEIGHT)


def evaluate(model_prob: float, quotes: list[MarketQuote]) -> EVResult:
    """Compute EV against the best available price, edge vs the consensus.

    EV uses the best (highest-payout) price across books so we line-shop the
    actual bet. The no-vig fair probability is a book-weighted consensus
    (Pinnacle heavier -- see ``BOOK_WEIGHTS``) so the thin-edge guard defers to
    the sharper market.
    """
    best = max(quotes, key=lambda q: american_to_decimal(q.american))
    dec = american_to_decimal(best.american)
    ev = model_prob * (dec - 1.0) - (1.0 - model_prob)

    wsum = sum(_weight(q.book) for q in quotes)
    fair = sum(_weight(q.book) * q.no_vig_prob for q in quotes) / wsum
    covered = sum(_weight(q.book) for q in quotes if q.devigged) / wsum

    return EVResult(
        model_prob=model_prob,
        best_quote=best,
        decimal=dec,
        ev=ev,
        fair_prob=fair,
        edge=model_prob - fair,
        devig_coverage=covered,
    )
