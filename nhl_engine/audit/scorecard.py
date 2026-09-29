"""Aggregate graded rows: by market, by tier, by gate (master plan §7 probation ledger).

A market leaves probation when its *buy* rows (``pass_gate`` and a buy tier)
number at least ``PROBATION_N`` with positive return and a Brier no worse than
the devigged consensus. The scorecard reports those three numbers per market
so the decision is read off the ledger, never argued from a slate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nhl_engine.audit.ledger import LedgerRow

PROBATION_N = 100


@dataclass
class Bucket:
    n: int = 0
    wins: int = 0
    pushes: int = 0
    pnl: float = 0.0
    clv_sum: float = 0.0
    clv_n: int = 0
    brier_model: float = 0.0
    brier_market: float = 0.0
    dual_rule: int = 0

    def add(self, r: LedgerRow) -> None:
        if r.outcome is None:
            return
        self.n += 1
        if r.outcome == "push":
            self.pushes += 1
        else:
            won = 1.0 if r.outcome == "win" else 0.0
            self.wins += int(won)
            self.brier_model += (r.model_prob - won) ** 2
            self.brier_market += (r.consensus - won) ** 2
        self.pnl += r.pnl or 0.0
        if r.clv is not None:
            self.clv_sum += r.clv
            self.clv_n += 1
        if r.dual_rule_differs:
            self.dual_rule += 1

    @property
    def decided(self) -> int:
        return self.n - self.pushes

    def as_dict(self) -> dict[str, float | int]:
        d = self.decided
        return {
            "n": self.n,
            "wins": self.wins,
            "pushes": self.pushes,
            "roi": self.pnl / self.n if self.n else 0.0,
            "pnl": self.pnl,
            "clv": self.clv_sum / self.clv_n if self.clv_n else 0.0,
            "brier_model": self.brier_model / d if d else 0.0,
            "brier_market": self.brier_market / d if d else 0.0,
            "dual_rule_differs": self.dual_rule,
        }


@dataclass
class Scorecard:
    by_market: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    buys_by_market: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_tier: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))
    by_gate: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))

    def add(self, r: LedgerRow) -> None:
        self.by_market[r.market].add(r)
        if r.is_buy:
            self.buys_by_market[r.market].add(r)
            self.by_tier[r.tier].add(r)
        for g in r.gates or ["pass_gate"]:
            self.by_gate[g].add(r)

    def probation(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for mk, b in sorted(self.buys_by_market.items()):
            d = b.as_dict()
            cleared = (
                b.n >= PROBATION_N
                and float(d["roi"]) > 0
                and float(d["brier_model"]) <= float(d["brier_market"])
            )
            out[mk] = {**d, "cleared": cleared, "needed": max(0, PROBATION_N - b.n)}
        return out

    def render(self) -> str:
        lines = ["market           n   W-L-P     roi     clv   brier(model/mkt)  dual"]
        for mk, b in sorted(self.by_market.items()):
            d = b.as_dict()
            losses = b.decided - b.wins
            lines.append(
                f"{mk:<15s} {b.n:4d}  {b.wins:3d}-{losses:3d}-{b.pushes:2d} {d['roi']:+7.3f} {d['clv']:+7.3f}"
                f"   {d['brier_model']:.3f}/{d['brier_market']:.3f}   {b.dual_rule:3d}"
            )
        lines.append("")
        lines.append("buys by tier")
        for t, b in sorted(self.by_tier.items()):
            d = b.as_dict()
            lines.append(f"  {t:<14s} {b.n:4d}  roi {d['roi']:+.3f}  clv {d['clv']:+.3f}")
        lines.append("")
        lines.append("probation (buys per market; needs 100, roi>0, brier<=market)")
        for mk, p in self.probation().items():
            status = "CLEARED" if p["cleared"] else f"{p['needed']} to go"
            lines.append(f"  {mk:<15s} {p['n']!s:>4s}  roi {p['roi']:+.3f}  {status}")
        return "\n".join(lines) + "\n"


def scorecard(rows: list[LedgerRow]) -> Scorecard:
    sc = Scorecard()
    for r in rows:
        sc.add(r)
    return sc


__all__ = ["PROBATION_N", "Bucket", "Scorecard", "scorecard"]
