"""Expected-value calculation against market prices, with VSIN handle/bets."""

from __future__ import annotations

from dataclasses import dataclass

from mlb_engine.market.odds import american_to_decimal, american_to_prob


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


def evaluate(model_prob: float, quotes: list[MarketQuote]) -> EVResult:
    """Pick the best-priced quote for our side and compute EV & edge."""
    best = max(quotes, key=lambda q: american_to_decimal(q.american))
    dec = american_to_decimal(best.american)
    ev = model_prob * (dec - 1.0) - (1.0 - model_prob)
    fair = american_to_prob(best.american)
    return EVResult(
        model_prob=model_prob,
        best_quote=best,
        decimal=dec,
        ev=ev,
        fair_prob=fair,
        edge=model_prob - fair,
        sharp_divergence=best.sharp_divergence,
    )
