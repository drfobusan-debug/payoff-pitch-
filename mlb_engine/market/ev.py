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

    @property
    def sharp_divergence(self) -> float | None:
        """handle% - bets%. Positive => money outweighs tickets (sharp side)."""
        if self.handle_pct is None or self.bets_pct is None:
            return None
        return self.handle_pct - self.bets_pct


@dataclass
class EVResult:
    model_prob: float
    best_quote: MarketQuote
    decimal: float
    ev: float  # EV per $1 staked
    fair_prob: float  # market no-vig implied (single-side approx)
    edge: float  # model_prob - fair_prob
    sharp_divergence: float | None


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
    fair = sum(_weight(q.book) * american_to_prob(q.american) for q in quotes) / wsum

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
    )
