"""Stage 9: turn the screened hitters and the arms they face into bets.

The screen ranks. It does not price, and a ranking that never reaches a ticket is
a points column, so this stage takes the engine's own priced slate --  the same
Monte Carlo and the same board the daily card runs off -- and reports, for every
name the screen kept:

* what the simulator projects him to do, at the resolution the board prices it;
* every side of his props that survived the EV screen, with the book and the
  price it survived at.

Nothing here re-models anything. The probabilities are the pipeline's, the tiers
are :mod:`mlb_engine.market.tiers`, and the point of the stage is that the
screen's output and the engine's output are read side by side: a hitter the
screen loves and the market has priced correctly produces no bet, and that is
information the screen cannot give on its own.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from mlb_engine.market.tiers import Tier
from mlb_engine.output.power_board import DISPLAY_ONLY
from mlb_engine.output.power_screen import ScreenResult
from mlb_engine.recommendations import Recommendation

#: Batter stats reported, in the order the card reads them.
BATTER_STATS: tuple[str, ...] = ("H", "1B", "2B", "HR", "R", "RBI", "HRR", "TB")
#: Starter stats reported, in the order the card reads them.
PITCHER_STATS: tuple[str, ...] = ("K", "BB", "H", "ER", "outs")

BUY_TIERS = (Tier.STRONG, Tier.MODERATE)


@dataclass(frozen=True)
class PricedSide:
    """One side of one prop that the engine bought, at the price it bought it."""

    market: str
    selection: str
    stat: str
    side: str
    line: float
    #: The probability the EV screen actually bet on: ``bet_prob`` when the
    #: market anchor moved the model, ``model_prob`` when it did not. The tier
    #: and the EV on this row were decided on that number, so printing the raw
    #: model beside them would show a probability nothing on the row was priced
    #: from -- and on the graded ledger the anchored number is the better of the
    #: two (Brier 0.240 against 0.244 over the recorded screen rows).
    prob: float
    fair: float
    odds: float
    book: str
    ev: float
    edge: float
    tier: Tier
    reasons: tuple[str, ...] = ()

    @property
    def is_buy(self) -> bool:
        """A buy, unless the market is one the card only displays.

        Same rule as :mod:`mlb_engine.output.power_board`: a display-only market
        is priced and shown and never bought, because the audit ledger does not
        record it and a bet nothing grades cannot be judged.
        """
        return self.tier in BUY_TIERS and self.stat not in DISPLAY_ONLY


@dataclass
class PlayerBets:
    """One player's projection and his surviving prices."""

    name: str
    mlbam_id: int = 0
    #: ``stat -> the highest priced threshold the model clears at even money``.
    #: A floor on the median at the board's own resolution: outs are quoted at
    #: 15.5 and 17.5, so a median of 16.4 outs reads as 16.
    median: dict[str, float] = field(default_factory=dict)
    #: ``stat -> P(stat > lowest priced line)``, which for a 0.5 line is the
    #: probability he records the stat at all.
    reach: dict[str, float] = field(default_factory=dict)
    #: ``stat -> the bound on a median the board's lines do not reach down to``.
    #: A starter quoted at 4.5 strikeouts and under it is *below five*, which is
    #: not the same claim as zero.
    under: dict[str, float] = field(default_factory=dict)
    sides: tuple[PricedSide, ...] = ()

    @property
    def buys(self) -> tuple[PricedSide, ...]:
        return tuple(s for s in self.sides if s.is_buy)


@dataclass
class BetCard:
    """The priced end of the screen: the hitters, the arms, and the tickets."""

    hitters: tuple[PlayerBets, ...] = ()
    arms: tuple[PlayerBets, ...] = ()

    @property
    def batter_buys(self) -> tuple[PricedSide, ...]:
        return _by_ev(self.hitters)

    @property
    def pitcher_buys(self) -> tuple[PricedSide, ...]:
        return _by_ev(self.arms)

    @property
    def priced_sides(self) -> int:
        return sum(len(p.sides) for p in self.hitters + self.arms)


def _by_ev(players: Iterable[PlayerBets]) -> tuple[PricedSide, ...]:
    buys = [s for p in players for s in p.buys]
    buys.sort(key=lambda s: -s.ev)
    return tuple(buys)


def median_at(overs: dict[float, float]) -> float:
    """The highest priced threshold the model clears at even money.

    Prop lines come at halves, so an over probability above 0.5 at 1.5 means the
    median is at least 2. Counted from the bottom and stopped at the first line
    the model does not clear, because the answer wanted is the median rather
    than the count of lines a monotone series happens to clear.

    NaN when the board's lowest line is already above the median -- a starter
    whose strikeouts are quoted at 4.5 and 5.5 and who clears neither is under
    five, not zero, and the board never priced the difference. :func:`below`
    carries that bound.
    """
    if not overs:
        return math.nan
    low = min(overs)
    if overs[low] <= 0.5:
        return 0.0 if low < 1.0 else math.nan
    best = 0.0
    for line in sorted(overs):
        if overs[line] <= 0.5:
            break
        best = math.ceil(line)
    return best


def below(overs: dict[float, float]) -> float:
    """The bound on an unresolved median: the lowest line the model missed."""
    if not overs:
        return math.nan
    low = min(overs)
    return math.ceil(low) if low >= 1.0 and overs[low] <= 0.5 else math.nan


def _ev(side: PricedSide) -> float:
    return -9.9 if math.isnan(side.ev) else side.ev


def _best_quote(sides: Iterable[PricedSide]) -> tuple[PricedSide, ...]:
    """One row per market, line and side: the best price the board offered.

    Several books hang the same bet, and printing each of them turns one
    recommendation into four. Line shopping is a different page.
    """
    best: dict[tuple[str, float, str], PricedSide] = {}
    for s in sides:
        key = (s.stat, s.line, s.side)
        if key not in best or _ev(s) > _ev(best[key]):
            best[key] = s
    return tuple(best.values())


def _matches(rec: Recommendation, name: str, mlbam_id: int) -> bool:
    """Is this row about this player? The id is authoritative, the name a fallback.

    Same rule as :mod:`mlb_engine.output.power_board`: a prop row carries the
    book's spelling in ``selection``, and surnames are shared often enough that a
    prefix test alone would cross-price two players.
    """
    if rec.player_id is not None and mlbam_id:
        return rec.player_id == mlbam_id
    return rec.selection.startswith(f"{name} ")


def _side(rec: Recommendation) -> PricedSide | None:
    if rec.stat is None or rec.side is None or rec.line is None:
        return None
    return PricedSide(
        market=rec.market,
        selection=rec.selection,
        stat=rec.stat,
        side=rec.side,
        line=rec.line,
        prob=rec.model_prob if rec.bet_prob is None else rec.bet_prob,
        fair=rec.fair_prob if rec.fair_prob is not None else math.nan,
        odds=rec.market_american if rec.market_american is not None else math.nan,
        book=rec.book or "",
        ev=rec.ev if rec.ev is not None else math.nan,
        edge=rec.edge if rec.edge is not None else math.nan,
        tier=rec.tier,
        reasons=tuple(rec.reasons),
    )


def _player(
    name: str,
    mlbam_id: int,
    recs: Sequence[Recommendation],
    stats: Sequence[str],
) -> PlayerBets:
    mine = [r for r in recs if _matches(r, name, mlbam_id)]
    sides = _best_quote(s for s in (_side(r) for r in mine) if s is not None)
    out = PlayerBets(name=name, mlbam_id=mlbam_id, sides=sides)
    for stat in stats:
        overs = {s.line: s.prob for s in sides if s.stat == stat and s.side == "over"}
        if not overs:
            continue
        out.median[stat] = median_at(overs)
        out.reach[stat] = overs[min(overs)]
        bound = below(overs)
        if not math.isnan(bound):
            out.under[stat] = bound
    return out


def price_card(
    recs: Sequence[Recommendation],
    hitters: Sequence[tuple[str, int]],
    arms: Sequence[tuple[str, int]],
) -> BetCard:
    """The bet card for these names, in the order given.

    ``recs`` is a priced slate from :meth:`mlb_engine.pipeline.Pipeline.run`.
    Names the slate never priced come back with an empty projection rather than
    being dropped, so the card says which of the screen's hitters the board had
    no market for -- that is a reason there is no bet, and a different reason
    from the market being right.
    """
    batter = [r for r in recs if r.category == "batter"]
    pitcher = [r for r in recs if r.category == "pitcher"]
    return BetCard(
        hitters=tuple(_player(n, i, batter, BATTER_STATS) for n, i in hitters),
        arms=tuple(_player(n, i, pitcher, PITCHER_STATS) for n, i in arms),
    )


def build(result: ScreenResult, recs: Sequence[Recommendation]) -> BetCard:
    """The card for a screen result: its hitters in composite order, and the arms.

    The hitters are read off the composite ranking so the priced table and the
    ranking table are the same list in the same order; when the halves never ran,
    the screened pool stands in. The arms are the starters whose lineups were
    screened -- the screen's own reason for looking at them is that they are
    fadeable, and a fadeable arm's props are the other half of that read.
    """
    ids = {v.line.name: v.line.mlbam_id for s in result.sections for v in s.hitters}
    order = [f.name for f in result.final] or list(ids)
    arms = [(s.starter.name, s.starter.mlbam_id) for s in result.sections]
    return price_card(recs, [(n, ids.get(n, 0)) for n in order], arms)
