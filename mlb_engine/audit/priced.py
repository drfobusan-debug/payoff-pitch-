"""What the prices did — the money record, next to the classification record.

The nightly audit's PPV/NPV answers one question: at the model's 0.5 boundary,
does it sort winners from losers? That is a *classification* score, and it is
free of the two things that decide whether a bet earns anything.

1. **The price.** A market can hit 70% on the buy side and lose money, because
   the prices it hit were -250 and needed 71.4%. Reported as "+42 points of lift"
   it reads like an edge; in the ledger it is -27% ROI. Total bases did exactly
   that.
2. **What was actually bet.** PPV counts every favored row, including the ones
   the tiers refused. A market the engine passes on entirely still scores.

So this module reports, per market, only the rows that were priced and bought:
how often they won, the win rate their prices demanded, the units they returned,
and the closing line they got. Nothing here is a rate the base rate can inflate,
and nothing here counts a row that was never a bet.

One-way rows (no second side quoted, so nothing to devig) are counted separately
rather than dropped: they were real bets and they lost real units, but their
"edge" was measured against a vigged number, so pooling them with the two-sided
rows makes both unreadable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from mlb_engine.audit.grade import PUSH, WIN
from mlb_engine.audit.ledger import ENGINE, LedgerEntry
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier

BUY_TIERS = frozenset({Tier.STRONG.value, Tier.MODERATE.value})
# A priced market needs this many graded bets before its ROI is printed as a
# number about the market rather than a number about ten coin flips.
MIN_PRICED = 15
# ROI this far below zero, on a market whose PPV lift is positive, is the
# price-versus-classification contradiction worth naming in prose.
CONTRADICTION_ROI = -0.02


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
    clv: float  # mean closing-line movement in probability points
    beat_close: int

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


def priced_buys(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Our own decided buys that carry a real price — the only rows with a P/L.

    A pass has no stake, an outside model's row is not ours, and an unpriced row
    was graded at an assumed -110 that nobody offered.
    """
    return [
        e
        for e in entries
        if e.source == ENGINE
        and e.tier in BUY_TIERS
        and e.odds is not None
        and e.result != PUSH
    ]


def _stat(key: str, label: str, rows: list[LedgerEntry]) -> PricedStat:
    be = 0.0
    for e in rows:
        assert e.odds is not None  # priced_buys filtered on it
        be += 1.0 / american_to_decimal(e.odds)
    one_way = [e for e in rows if e.under_odds is None]
    clv_rows = [e for e in rows if e.clv is not None]
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
        clv=(sum(e.clv or 0.0 for e in clv_rows) / len(clv_rows)) if clv_rows else float("nan"),
        beat_close=sum(1 for e in clv_rows if (e.clv or 0.0) > 0),
    )


def engine_priced_stat(entries: list[LedgerEntry], label: str = "Every buy") -> PricedStat:
    return _stat("ALL", label, priced_buys(entries))


def priced_stats(
    entries: list[LedgerEntry],
    labeller: Callable[[str], str] | None = None,
) -> list[PricedStat]:
    """Priced performance per market, biggest sample first."""
    groups: dict[str, list[LedgerEntry]] = defaultdict(list)
    for e in priced_buys(entries):
        groups[e.market].append(e)
    out = [
        _stat(key, labeller(key) if labeller else key, rows)
        for key, rows in groups.items()
    ]
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
    engine is broken at picking: it means the side selection is right and the
    price already knows. Naming these is the whole reason the money table sits
    next to the PPV table -- read alone, the lift column is an argument for
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
    graded = [s for s in stats if s.n >= min_n]
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
    one_way = [s for s in stats if s.n_one_way]
    if one_way:
        n = sum(s.n_one_way for s in one_way)
        units = sum(s.units_one_way for s in one_way)
        out.append(
            f"**{n} of these bets were one-way quotes** ({units:+.1f}u) — no second "
            "side was posted, so their edge was measured against a vigged number "
            "and the devigged columns above do not apply to them."
        )
    return out
