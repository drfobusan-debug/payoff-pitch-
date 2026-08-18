"""What the board is offering: best numbers, key-number crossings, and middles
that actually clear the vig.

Three things come out of a scan, and none of them is an opinion about who wins:

* **Best number and price per side**, with the book holding it, against the
  quote-weighted consensus. Half a point on a key number is worth more than any
  rating improvement measured in this repo, and it is the only edge available
  without predicting anything.
* **Crossings**, where the best available number buys more win probability than
  its worse price costs. Both sides of that trade are priced in the same unit --
  probability -- so "+3.5 at -120 vs +2.5 at -110" resolves to a number instead
  of an instinct.
* **Middles**, priced against the empirical distribution rather than a normal
  curve, so the ones that only look free do not print. Most do not clear: two
  legs at -112 need the window to hit ~5.7% of the time, and a total landing on
  52-53 does so under 4%.

Every probability is conditioned on the consensus number for that game (see
:mod:`lineshop.distribution`), and carries the sample it was counted from.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from lineshop import distribution as dist
from lineshop.feed import OVER, UNDER, Game, Quote, restrict
from nfl_engine.market.odds import american_to_decimal, american_to_prob

ABOVE, BELOW = "above", "below"
H2H, SPREADS, TOTALS = "h2h", "spreads", "totals"
# Margins that carry a disproportionate share of results. Only used to label a
# crossing -- the probability is always counted, never assumed from this list.
KEY_MARGINS = {"cfb": (3, 7, 10, 14, 1, 4, 6), "nfl": (3, 7, 6, 4, 10, 14, 1)}
# A game nobody has priced yet is not a disagreement, it is an empty board.
MIN_BOOKS = 3
# Below this the "edge" is inside the noise of both the sample and the board.
MIN_CROSSING_EDGE = 0.01
MIN_MIDDLE_EV = 0.0


@dataclass(frozen=True)
class Offer:
    """The best version of one side available anywhere on the board."""

    side: str
    market: str
    point: float | None
    american: int
    books: tuple[str, ...]

    @property
    def decimal(self) -> float:
        return american_to_decimal(self.american)

    def label(self) -> str:
        if self.market == H2H:
            return f"{self.side} ML {self.american:+d}"
        if self.market == TOTALS:
            return f"{self.side} {self.point:g} ({self.american:+d})"
        return f"{self.side} {self.point:+g} ({self.american:+d})"


@dataclass(frozen=True)
class Crossing:
    """A better number that is worth more than the worse price attached to it."""

    matchup: str
    market: str
    side: str
    best: Offer
    consensus_point: float
    consensus_american: int
    prob_gain: float  # win probability bought by the extra number
    price_cost: float  # break-even probability given up for the worse price
    keys: tuple[int, ...]
    sample: int

    @property
    def edge(self) -> float:
        return self.prob_gain - self.price_cost


@dataclass(frozen=True)
class Middle:
    matchup: str
    market: str
    low: Offer  # the leg that wins when the result goes high
    high: Offer  # the leg that wins when the result goes low
    p_middle: float
    p_push: float  # one leg pushes, the other wins -- a free half of the middle
    ev: float  # per unit of total stake (both legs)
    sample: int
    thin: bool


@dataclass(frozen=True)
class GameScan:
    matchup: str
    sport: str
    commence: str
    books: int
    consensus_spread: float | None  # home handicap
    consensus_total: float | None
    best: dict[tuple[str, str], Offer]
    crossings: tuple[Crossing, ...]
    middles: tuple[Middle, ...]


# -- consensus and best offers ------------------------------------------------
def consensus_point(game: Game, market: str, side: str | None = None) -> float | None:
    """Quote-weighted median number, on the home axis for spreads."""
    points: list[float] = []
    for (m, s), quotes in game.quotes.items():
        if m != market or (side is not None and s != side):
            continue
        for quote in quotes:
            if quote.point is None:
                continue
            flip = market == SPREADS and side is None and s != game.home
            points.append(-quote.point if flip else quote.point)
    return statistics.median(points) if points else None


def consensus_price(game: Game, market: str, side: str, point: float | None) -> int | None:
    """Median price for ``side`` at the consensus number -- what the shopper is
    giving up by taking a better rung somewhere else."""
    prices = [
        q.american
        for q in game.get(market, side)
        if point is None or (q.point is not None and abs(q.point - point) < 1e-9)
    ]
    return round(statistics.median(prices)) if prices else None


def best_offer(game: Game, market: str, side: str) -> Offer | None:
    """Best number first, price second.

    Number before price is deliberate and is the one place this ordering is
    right: a rung crossing 3 is worth several cents of vig, and the crossing
    check below re-prices the trade anyway, so nothing is lost by leading with
    the number.
    """
    quotes = game.get(market, side)
    if not quotes:
        return None
    ranked = sorted(quotes, key=lambda q: offer_rank(market, side, game, q), reverse=True)
    top = offer_rank(market, side, game, ranked[0])
    books = tuple(sorted({q.book for q in quotes if offer_rank(market, side, game, q) == top}))
    return Offer(
        side=side,
        market=market,
        point=ranked[0].point,
        american=ranked[0].american,
        books=books,
    )


def offer_rank(market: str, side: str, game: Game, quote: Quote) -> tuple[float, float]:
    if market == H2H or quote.point is None:
        return (0.0, float(quote.american))
    # An Over wants the lowest line, everything else the highest one it can get.
    point = -quote.point if side == OVER else quote.point
    return (point, float(quote.american))


# -- probability of a side ----------------------------------------------------
def _axis(market: str) -> str:
    return dist.TOTAL if market == TOTALS else dist.MARGIN


def _threshold(game: Game, market: str, side: str, point: float | None) -> tuple[str, float]:
    """The side expressed as a comparison against the result axis.

    Margin is home score minus away score, so a home handicap of -3.5 wins above
    +3.5 and an away +3.5 wins below +3.5; a moneyline is the same question at
    zero, which is why it needs no special case.
    """
    if market == TOTALS:
        return (ABOVE if side == OVER else BELOW, float(point or 0.0))
    handicap = float(point or 0.0)
    if side == game.home:
        return (ABOVE, -handicap)
    return (BELOW, handicap)


def side_probability(
    sport: str, game: Game, market: str, side: str, point: float | None, line: float
) -> dist.Estimate:
    """P(side wins), counting a push as half -- the unit a price compares to."""
    kind, threshold = _threshold(game, market, side, point)
    axis = _axis(market)
    above = dist.p_above(sport, axis, line, threshold)
    push = dist.p_at(sport, axis, line, threshold)
    p = above.p if kind == ABOVE else 1.0 - above.p - push.p
    return dist.Estimate(p + 0.5 * push.p, above.n, above.tolerance)


def _conditioning_line(game: Game, market: str) -> float | None:
    """The number the historical sample is drawn around."""
    if market == TOTALS:
        return consensus_point(game, TOTALS)
    handicap = consensus_point(game, SPREADS)
    # The distribution's margin axis is "home favoured by", the negative of the
    # handicap a book hangs on the home team.
    return None if handicap is None else -handicap


# -- crossings ----------------------------------------------------------------
def crossings(sport: str, game: Game, shop: Game | None = None) -> list[Crossing]:
    """``game`` sets the consensus; ``shop`` is what the operator can bet.

    Keeping them separate matters once the scan is pointed at real accounts: the
    number to beat is the market's, but the number available is whatever the
    four books you hold are showing.
    """
    shop = shop or game
    out: list[Crossing] = []
    for market in (SPREADS, TOTALS):
        line = _conditioning_line(game, market)
        if line is None:
            continue
        for side in sorted(set(shop.sides(market))):
            best = best_offer(shop, market, side)
            base = consensus_point(game, market, side)
            if best is None or best.point is None or base is None or best.point == base:
                continue
            base_price = consensus_price(game, market, side, base)
            if base_price is None:
                continue
            better = side_probability(sport, game, market, side, best.point, line)
            worse = side_probability(sport, game, market, side, base, line)
            gain = better.p - worse.p
            cost = american_to_prob(best.american) - american_to_prob(base_price)
            if gain - cost < MIN_CROSSING_EDGE:
                continue
            out.append(
                Crossing(
                    matchup=game.matchup,
                    market=market,
                    side=side,
                    best=best,
                    consensus_point=base,
                    consensus_american=base_price,
                    prob_gain=gain,
                    price_cost=cost,
                    keys=_keys_crossed(sport, game, market, side, base, best.point),
                    sample=better.n,
                )
            )
    return sorted(out, key=lambda c: c.edge, reverse=True)


def _keys_crossed(
    sport: str, game: Game, market: str, side: str, base: float, best: float
) -> tuple[int, ...]:
    if market != SPREADS:
        return ()
    _, low = _threshold(game, market, side, base)
    _, high = _threshold(game, market, side, best)
    lo, hi = sorted((abs(low), abs(high)))
    return tuple(sorted(k for k in KEY_MARGINS.get(sport, ()) if lo <= k <= hi))


# -- middles ------------------------------------------------------------------
def middles(sport: str, game: Game, shop: Game | None = None) -> list[Middle]:
    """Every two-legged window the board is currently offering, priced.

    Both legs are taken at the best number available anywhere, which is the only
    way a middle exists at all: one book's Over and another book's Under.
    """
    shop = shop or game
    out: list[Middle] = []
    for market, pair in ((SPREADS, (game.away, game.home)), (TOTALS, (OVER, UNDER))):
        line = _conditioning_line(game, market)
        if line is None:
            continue
        legs = [best_offer(shop, market, side) for side in pair]
        if any(leg is None or leg.point is None for leg in legs):
            continue
        above = next(
            leg
            for leg in legs
            if leg is not None and _threshold(game, market, leg.side, leg.point)[0] == ABOVE
        )
        below = next(
            leg
            for leg in legs
            if leg is not None and _threshold(game, market, leg.side, leg.point)[0] == BELOW
        )
        t_above = _threshold(game, market, above.side, above.point)[1]
        t_below = _threshold(game, market, below.side, below.point)[1]
        if t_below <= t_above:
            continue  # no window: the two best numbers do not overlap
        priced = middle_ev(sport, _axis(market), line, t_above, t_below, above, below)
        if priced is None or priced.ev < MIN_MIDDLE_EV:
            continue
        out.append(
            Middle(
                matchup=game.matchup,
                market=market,
                low=above,
                high=below,
                p_middle=priced.p_middle,
                p_push=priced.p_push,
                ev=priced.ev,
                sample=priced.sample,
                thin=priced.thin,
            )
        )
    return out


@dataclass(frozen=True)
class _Priced:
    p_middle: float
    p_push: float
    ev: float
    sample: int
    thin: bool


def middle_ev(
    sport: str, axis: str, line: float, t_above: float, t_below: float, above: Offer, below: Offer
) -> _Priced | None:
    """EV per unit of total stake for one unit on each leg.

    Outside the window exactly one leg wins, so the loss is the vig on one bet
    rather than both -- which is why a middle can be worth taking at a hit rate
    in the single digits, and why the exact hit rate has to be counted rather
    than eyeballed.
    """
    inside = dist.p_between(sport, axis, line, t_above, t_below)
    if not inside.n:
        return None
    push_low = dist.p_at(sport, axis, line, t_above).p
    push_high = dist.p_at(sport, axis, line, t_below).p
    p_high_only = dist.p_above(sport, axis, line, t_below).p  # above-leg wins alone
    p_low_only = max(0.0, 1.0 - inside.p - push_low - push_high - p_high_only)
    d_a, d_b = above.decimal, below.decimal
    ev = (
        inside.p * (d_a + d_b - 2.0)
        + push_low * (d_b - 1.0)
        + push_high * (d_a - 1.0)
        + p_high_only * (d_a - 2.0)
        + p_low_only * (d_b - 2.0)
    )
    return _Priced(inside.p, push_low + push_high, ev / 2.0, inside.n, inside.thin)


# -- the scan -----------------------------------------------------------------
def scan_game(sport: str, game: Game, shop: Game | None = None) -> GameScan | None:
    if len(game.books) < MIN_BOOKS:
        return None
    shop = shop or game
    best: dict[tuple[str, str], Offer] = {}
    for market, side in shop.quotes:
        offer = best_offer(shop, market, side)
        if offer is not None:
            best[(market, side)] = offer
    return GameScan(
        matchup=game.matchup,
        sport=sport,
        commence=game.commence,
        books=len(game.books),
        consensus_spread=consensus_point(game, SPREADS),
        consensus_total=consensus_point(game, TOTALS),
        best=best,
        crossings=tuple(crossings(sport, game, shop)),
        middles=tuple(middles(sport, game, shop)),
    )


def scan(sport: str, games: list[Game], books: tuple[str, ...] = ()) -> list[GameScan]:
    """Scan the board, optionally as seen from the accounts in ``books``."""
    shops = {g.game_id: g for g in restrict(games, books)} if books else {}
    out = [scan_game(sport, game, shops.get(game.game_id) if books else None) for game in games]
    return [s for s in out if s is not None]
