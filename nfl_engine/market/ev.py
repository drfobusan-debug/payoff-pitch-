"""Turning one game's board into priced bets.

The phase-3 walk-forward settled what this layer is allowed to claim. Over 3,450
games the opponent-adjusted rating lost to the closing spread (margin MAE 10.282
against 9.905), its disagreement with the line explained none of the line's
residual (t +0.25), and the blend curve was monotone to ``MARKET_WEIGHT = 1.0``.
So **the mean is the market's**, and this module does not look for a bet by
disagreeing about who is better. It looks for two things that survive that
verdict:

*Execution.* The same game is quoted by a dozen books. A price better than the
de-vigged consensus on the identical line is an edge in arithmetic, not in
opinion, and it is the only edge here that does not depend on a forecast being
right.

*Shape.* Anchored to the market's own number, the possession simulator prices
every rung of the ladder coherently -- P(margin = 3) is 13.95% against a realised
14.83% and a normal's 5.36%. That is what lets it say what -3 to -2.5 is worth
when a book hangs the half point at a price that does not reflect it.

Both edges are reported separately and never merged, because they fail
differently: ``ev_fair`` is arbitrage-like and survives a bad model, ``ev_model``
is a claim about the distribution and dies with it. A bet needs the first; the
second is a veto, not a licence (see :mod:`nfl_engine.screens`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nfl_engine.market.board import OVER, UNDER, GameOdds, MarketQuote
from nfl_engine.market.fair import DEFAULT_METHOD, FairPrice, fair_from_quotes
from nfl_engine.models.distribution import MarketProb, ScoreDistribution

MONEYLINE, SPREAD, TOTAL = "moneyline", "spread", "total"


@dataclass(frozen=True)
class PricedBet:
    """One side of one line at one book, with both edges kept apart."""

    matchup: str
    market: str
    side: str
    line: float | None
    book: str
    american: float
    decimal: float
    # The same book's price on the other side of the same line. Persisted so the
    # fade can be graded later without re-fetching a board that no longer exists,
    # and never fabricated when the quote was unpaired.
    opposite_american: float | None
    model_prob: float
    push_prob: float
    fair: FairPrice | None
    screens: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fair_prob(self) -> float | None:
        if self.fair is None or not self.fair.is_trustworthy():
            return None
        return self.fair.prob

    @property
    def ev_model(self) -> float:
        """EV per unit staked at this price, on the simulator's probability."""
        return ev(self.model_prob, self.decimal)

    @property
    def ev_fair(self) -> float | None:
        """EV per unit staked on the consensus probability: the execution edge."""
        prob = self.fair_prob
        return None if prob is None else ev(prob, self.decimal)

    @property
    def edge_vs_fair(self) -> float | None:
        """Simulator minus consensus. Large means *we* are probably wrong."""
        prob = self.fair_prob
        return None if prob is None else self.model_prob - prob

    def is_bet(self) -> bool:
        return not self.screens

    def label(self) -> str:
        if self.market == MONEYLINE:
            return f"{self.side} ML"
        if self.market == SPREAD:
            return f"{self.side} {self.line:+g}"
        return f"{self.side} {self.line:g}"


def ev(prob: float, decimal: float) -> float:
    """Expected profit per unit staked, pushes already conditioned out."""
    return prob * (decimal - 1.0) - (1.0 - prob)


def implied_break_even(decimal: float) -> float:
    return 1.0 / decimal


def price_game(
    odds: GameOdds,
    distribution: ScoreDistribution,
    *,
    home: str,
    away: str,
    method: str = DEFAULT_METHOD,
) -> list[PricedBet]:
    """Price every quote on the board off one simulated distribution.

    Every rung is priced, not just the main line: the ladder is the product in
    this sport, and a rung nobody else quotes is exactly where the shape can be
    worth something. Screens are applied by the caller so that a rejected bet
    still lands on the ledger with its reason.
    """
    bets: list[PricedBet] = []

    for side, quotes in odds.ml.items():
        prob = distribution.moneyline(home=(side == home))
        bets.extend(_bets_for(odds, MONEYLINE, side, None, quotes, prob, method))

    for home_point, sides in odds.spreads.items():
        for side, quotes in sides.items():
            if side == home:
                point, prob = home_point, distribution.spread(home_point)
            elif side == away:
                point = -home_point
                prob = distribution.spread(point, home=False)
            else:
                continue
            bets.extend(_bets_for(odds, SPREAD, side, point, quotes, prob, method))

    for line, sides in odds.totals.items():
        for side, quotes in sides.items():
            if side not in (OVER, UNDER):
                continue
            prob = distribution.total(line, over=(side == OVER))
            bets.extend(_bets_for(odds, TOTAL, side, line, quotes, prob, method))

    return bets


def _bets_for(
    odds: GameOdds,
    market: str,
    side: str,
    line: float | None,
    quotes: list[MarketQuote],
    prob: MarketProb,
    method: str,
) -> list[PricedBet]:
    """One :class:`PricedBet` per book, sharing the line's consensus fair price."""
    fair = fair_from_quotes(quotes, method=method)
    return [
        PricedBet(
            matchup=odds.matchup,
            market=market,
            side=side,
            line=line,
            book=quote.book,
            american=quote.american,
            decimal=quote.decimal,
            opposite_american=quote.opposite_american,
            model_prob=prob.conditional,
            push_prob=prob.push,
            fair=fair,
        )
        for quote in quotes
    ]


def best_by_line(bets: list[PricedBet]) -> list[PricedBet]:
    """Keep only the best-priced book on each market/side/line.

    Betting the same line at two books is one position, and pooling both would
    double-count it in every PPV/NPV and ROI figure downstream.
    """
    best: dict[tuple[str, str, float | None], PricedBet] = {}
    for bet in bets:
        key = (bet.market, bet.side, bet.line)
        current = best.get(key)
        if current is None or bet.decimal > current.decimal:
            best[key] = bet
    return list(best.values())
