"""What the prices did -- the money record, next to the classification record.

The audit's PPV/NPV answers one question: at the model's 0.5 boundary, does it
sort winners from losers? That is a *classification* score, and it is blind to
the two things that decide whether a bet earned anything.

1. **The price.** A market can hit 58% on the buy side and still lose, because
   the prices it took were -150 and needed 60%. Reported as "+8 points of lift"
   it reads like an edge; in the ledger it is negative ROI. This is not a
   hypothetical -- it is what the MLB engine paid to learn, and the reason the
   CFB scorecard already grades a tier against its break-even.
2. **What was actually bet.** PPV counts every favored row, including the ones
   the tiers refused. A market the engine passes on entirely still scores.

So this module reports, per market, only the rows that were priced and bought:
how often they won, the win rate their prices demanded, the units they returned,
and the closing number they got. Nothing here is a rate the base rate can
inflate, and nothing here counts a row that was never a bet.

Two football-specific columns sit beside the probability CLV. A spread or total
is shopped in *points*, and on a main line the number carries movement the price
does not (-3 to -3.5 at the same -110 is a real loss of value that ``clv``
cannot see), so ``clv_pts`` is summarised separately. And one-way rows -- no
second side quoted, so nothing to devig -- are counted apart rather than
dropped: they were real bets that returned real units, but their edge was
measured against a vigged number, so pooling them makes both groups unreadable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from cfb_engine.audit.grade import PUSH, WIN
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier

BUY_TIERS = frozenset({Tier.STRONG.value, Tier.MODERATE.value})
# A priced market needs this many graded bets before its ROI is printed as a
# number about the market rather than a number about ten coin flips. Lower than
# the MLB engine's 15 would be false precision; higher would print nothing for
# most of a 12-game season, since college football offers ~800 games a year
# against baseball's nightly prop board.
MIN_PRICED = 15
# ROI this far below zero, on a market whose PPV lift is positive, is the
# price-versus-classification contradiction worth naming in prose.
CONTRADICTION_ROI = -0.02
# The engine-wide row's key. Findings skip it: it is the sum of the market rows,
# so reading it as one more market double-counts every bet in the total.
ENGINE_KEY = "ALL"


@dataclass
class PricedStat:
    """Realized performance of the bets actually placed in one market."""

    key: str
    label: str
    n: int  # decided, priced buys
    wins: int
    breakeven: float  # mean win rate the prices demanded
    units: float
    n_one_way: int  # of ``n``, rows with no second side quoted
    units_one_way: float
    n_clv: int
    clv: float  # mean closing movement in no-vig probability points
    beat_close: int
    n_clv_pts: int
    clv_pts: float  # mean points of line value against the close
    beat_number: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def shortfall(self) -> float:
        """Points of win rate above (positive) or below what the price demanded."""
        return self.win_rate - self.breakeven

    @property
    def roi(self) -> float:
        return self.units / self.n if self.n else float("nan")

    @property
    def two_sided(self) -> int:
        return self.n - self.n_one_way

    @property
    def clv_rate(self) -> float:
        return self.beat_close / self.n_clv if self.n_clv else float("nan")

    @property
    def number_rate(self) -> float:
        return self.beat_number / self.n_clv_pts if self.n_clv_pts else float("nan")


def priced_buys(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Decided buys carrying a real price -- the only rows with a P/L.

    A pass has no stake, and an unpriced row was graded at an assumed -110 that
    nobody actually offered.
    """
    return [e for e in entries if e.tier in BUY_TIERS and e.odds is not None and e.result != PUSH]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _stat(key: str, label: str, rows: list[LedgerEntry]) -> PricedStat:
    # priced_buys filtered on a real price, so every row has one.
    be = sum(1.0 / american_to_decimal(e.odds) for e in rows if e.odds is not None)
    one_way = [e for e in rows if e.under_odds is None]
    clv_rows = [e for e in rows if e.clv is not None]
    pts_rows = [e for e in rows if e.clv_pts is not None]
    return PricedStat(
        key=key,
        label=label,
        n=len(rows),
        wins=sum(1 for e in rows if e.result == WIN),
        breakeven=be / len(rows) if rows else float("nan"),
        units=round(sum(e.pnl for e in rows), 3),
        n_one_way=len(one_way),
        units_one_way=round(sum(e.pnl for e in one_way), 3),
        n_clv=len(clv_rows),
        clv=_mean([e.clv or 0.0 for e in clv_rows]),
        beat_close=sum(1 for e in clv_rows if (e.clv or 0.0) > 0),
        n_clv_pts=len(pts_rows),
        clv_pts=_mean([e.clv_pts or 0.0 for e in pts_rows]),
        beat_number=sum(1 for e in pts_rows if (e.clv_pts or 0.0) > 0),
    )


def engine_priced_stat(entries: list[LedgerEntry], label: str = "Every buy") -> PricedStat:
    return _stat(ENGINE_KEY, label, priced_buys(entries))


def priced_stats(
    entries: list[LedgerEntry],
    labeller: Callable[[str], str] | None = None,
) -> list[PricedStat]:
    """Priced performance per market, biggest sample first."""
    groups: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in priced_buys(entries):
        groups[e.market].append(e)
    out = [_stat(key, labeller(key) if labeller else key, rows) for key, rows in groups.items()]
    out.sort(key=lambda s: s.n, reverse=True)
    return out


def contradictions(
    stats: list[PricedStat],
    ppv_lift: dict[str, float],
    *,
    min_n: int = MIN_PRICED,
) -> list[tuple[PricedStat, float]]:
    """Markets the classification report praises and the ledger charges for.

    A positive PPV lift with a negative ROI is not a paradox and not a sign the
    engine is broken at picking sides: it means the side selection is right and
    the price already knows. Naming these is the whole reason the money table
    sits next to the PPV table -- read alone, the lift column is an argument for
    betting more of exactly the markets that cost the most.
    """
    out: list[tuple[PricedStat, float]] = []
    for s in stats:
        lift = ppv_lift.get(s.key)
        if lift is None or s.n < min_n:
            continue
        if lift > 0 and s.roi <= CONTRADICTION_ROI:
            out.append((s, lift))
    out.sort(key=lambda pair: pair[0].roi)
    return out


def priced_findings(stats: list[PricedStat], *, min_n: int = MIN_PRICED) -> list[str]:
    """Plain-language read on the markets whose prices are doing the damage."""
    out: list[str] = []
    markets = [s for s in stats if s.key != ENGINE_KEY]
    graded = [s for s in markets if s.n >= min_n]
    for s in sorted(graded, key=lambda s: s.roi)[:3]:
        if s.roi >= 0:
            break
        out.append(
            f"**{s.label} buys returned {s.roi * 100:+.1f}%** ({s.units:+.1f}u on "
            f"{s.n} bets): {s.win_rate * 100:.1f}% won against the "
            f"{s.breakeven * 100:.1f}% the prices demanded, "
            f"{abs(s.shortfall) * 100:.1f} points short."
        )
    best = max(graded, key=lambda s: s.roi, default=None)
    if best is not None and best.roi > 0:
        out.append(
            f"**{best.label} is the one market clearing its own price**: "
            f"{best.win_rate * 100:.1f}% against a {best.breakeven * 100:.1f}% "
            f"break-even, {best.units:+.1f}u on {best.n} bets "
            f"({best.roi * 100:+.1f}%)."
        )
    # Counted over every market, not just the printable ones: this sentence is
    # read against the total row, and a thin market's one-way bets are in that
    # total whether or not the market earned a line of its own.
    one_way = [s for s in markets if s.n_one_way]
    if one_way:
        n = sum(s.n_one_way for s in one_way)
        units = sum(s.units_one_way for s in one_way)
        out.append(
            f"**{n} of these bets were one-way quotes** ({units:+.1f}u) -- no second "
            "side was posted, so their edge was measured against a vigged number "
            "and the devigged columns above do not apply to them."
        )
    return out
