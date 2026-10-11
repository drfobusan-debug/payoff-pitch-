"""Which buys belong to a kind of bet the ledger has already proven.

Display only: no tier, gate or stake reads it. A kind of bet is a market
(spread, moneyline, total) or a candidate upgrade from
:mod:`cfb_engine.audit.probation`. It earns the star once it has the probation
volume (100 graded bets) and is winning by more than one standard error in both
halves of the window -- the same three tests that would shut a losing market,
read the other way.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.probation import (
    CANDIDATE_UPGRADES,
    PROMOTE,
    WATCHING,
    CandidateUpgrade,
    Probation,
    market_probation,
    upgrade_probation,
)
from cfb_engine.market.tiers import Tier
from cfb_engine.recommendations import Recommendation

STAR = "★"
LEGEND = (
    f"{STAR} = this kind of bet has 100+ graded bets and is winning by more than one standard"
    " error in both halves of the ledger. Display only; stakes are unchanged."
)


def passed(p: Probation) -> bool:
    """A market verdict that is judged and winning on all three tests."""
    return (
        p.status != WATCHING and p.roi > max(p.se, 0.0) and p.first_half > 0 and p.second_half > 0
    )


@dataclass(frozen=True)
class Proven:
    markets: frozenset[str] = frozenset()
    upgrades: tuple[CandidateUpgrade, ...] = ()

    @classmethod
    def from_ledger(cls, entries: list[LedgerEntry]) -> Proven:
        markets = frozenset(p.name for p in market_probation(entries) if passed(p))
        promoted = {p.name for p in upgrade_probation(entries) if p.status == PROMOTE}
        return cls(markets, tuple(u for u in CANDIDATE_UPGRADES if u.name in promoted))

    def starred(self, rec: Recommendation) -> bool:
        if rec.tier == Tier.PASS:
            return False
        return rec.market in self.markets or any(u.promotes(rec) for u in self.upgrades)

    def mark(self, rec: Recommendation) -> str:
        return f"{STAR} " if self.starred(rec) else ""

    def any_starred(self, recs: list[Recommendation]) -> bool:
        return any(self.starred(r) for r in recs)
