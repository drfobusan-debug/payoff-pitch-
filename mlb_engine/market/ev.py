"""Expected-value calculation against market prices, with VSIN handle/bets."""

from __future__ import annotations

from dataclasses import dataclass

from mlb_engine.market.odds import american_to_decimal, american_to_prob

# Circa is a low-limit sharp market; weight its line/split heavier than the
# recreational books when forming the consensus fair price and sharp signal.
BOOK_WEIGHTS = {"circa": 2.0, "draftkings": 1.0}
_DEFAULT_WEIGHT = 1.0


@dataclass
class MarketQuote:
    book: str  # "draftkings" | "circa"
    american: float
    handle_pct: float | None = None  # VSIN: % of money on this side
    bets_pct: float | None = None  # VSIN: % of tickets on this side
    # Same book, same line, other side (under / opposing team). Required to strip
    # the vig: one side's implied probability carries roughly half the book's
    # overround, which is ~3.4 points on a player prop.
    opposite_american: float | None = None

    @property
    def sharp_divergence(self) -> float | None:
        """handle% - bets%. Positive => money outweighs tickets (sharp side)."""
        if self.handle_pct is None or self.bets_pct is None:
            return None
        return self.handle_pct - self.bets_pct

    @property
    def no_vig_prob(self) -> float:
        """The book's fair probability for this side, vig removed if possible.

        Two-sided normalisation (``p / (p + q)``) is the only way to recover it.
        Without the other side this falls back to the raw implied probability,
        which overstates the market by about half the hold.
        """
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
    sharp_divergence: float | None
    # Share of consensus weight whose vig was actually removed. Below 1.0 the
    # edge is understated, so a thin-edge guard is stricter than it looks.
    devig_coverage: float = 1.0


def anchor_to_market(model_prob: float, fair_prob: float, weight: float) -> float:
    """Shrink the model toward the market's no-vig price by ``weight``.

    ``weight`` 0 leaves the model untouched, 1 bets the market itself. Rationale:
    retro-pricing nine slates showed the devigged market is the better forecaster
    in every market we bet (Brier .2347 vs .2408), so the market is the better
    prior and the model should have to earn its departures from it.

    Note what this does to selection, because it is not what it sounds like.
    Both screening criteria are affine in the probability, so shrinking by
    ``weight`` scales the measured edge: ``edge = (1 - weight) * (model - fair)``.
    Against a fixed threshold that is arithmetically identical to *raising* the
    edge requirement to ``threshold / (1 - weight)`` -- it keeps the model's
    largest disagreements with the market and drops the small ones. It makes the
    model pay a bigger toll to disagree; it does not make it defer.
    """
    w = min(max(weight, 0.0), 1.0)
    return (1.0 - w) * model_prob + w * fair_prob


def ev_per_dollar(model_prob: float, american: float) -> float:
    dec = american_to_decimal(american)
    return model_prob * (dec - 1.0) - (1.0 - model_prob)


def _weight(book: str) -> float:
    return BOOK_WEIGHTS.get(book, _DEFAULT_WEIGHT)


def evaluate(model_prob: float, quotes: list[MarketQuote]) -> EVResult:
    """Compute EV against the best available price, edge vs the sharp consensus.

    EV uses the best (highest-payout) price across books so we line-shop the
    actual bet. The no-vig fair probability and the handle/bets divergence are
    book-weighted consensus values (Circa heavier -- see ``BOOK_WEIGHTS``) so
    thin-edge guards and the sharp signal defer to the sharper market.
    """
    best = max(quotes, key=lambda q: american_to_decimal(q.american))
    dec = american_to_decimal(best.american)
    ev = model_prob * (dec - 1.0) - (1.0 - model_prob)

    wsum = sum(_weight(q.book) for q in quotes)
    fair = sum(_weight(q.book) * q.no_vig_prob for q in quotes) / wsum
    covered = sum(_weight(q.book) for q in quotes if q.devigged) / wsum

    weighted_divs = [
        (_weight(q.book), d) for q in quotes if (d := q.sharp_divergence) is not None
    ]
    divergence: float | None = None
    if weighted_divs:
        dw = sum(w for w, _ in weighted_divs)
        divergence = sum(w * d for w, d in weighted_divs) / dw

    return EVResult(
        model_prob=model_prob,
        best_quote=best,
        decimal=dec,
        ev=ev,
        fair_prob=fair,
        edge=model_prob - fair,
        sharp_divergence=divergence,
        devig_coverage=covered,
    )
