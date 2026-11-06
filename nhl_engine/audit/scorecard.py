"""Aggregate graded rows: by market, by tier, by gate (master plan §7 probation ledger).

A market leaves probation when its *buy* rows (``pass_gate`` and a buy tier)
number at least ``PROBATION_N`` with positive return and a Brier no worse than
the devigged consensus. The scorecard reports those three numbers per market
so the decision is read off the ledger, never argued from a slate.

Rows priced at or after puck drop are left out, and CLV counts only closes
taken within ``STALE_CLOSE_MIN`` of puck drop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from engine_common import closecheck, promotion
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
    pnls: list[float] = field(default_factory=list)
    clvs: list[float] = field(default_factory=list)
    close_rois: list[float] = field(default_factory=list)
    graded: list[closecheck.Graded] = field(default_factory=list)

    def add(self, r: LedgerRow) -> None:
        if r.outcome is None:
            return
        self.n += 1
        self.pnls.append(r.pnl or 0.0)
        if r.outcome == "push":
            self.pushes += 1
        else:
            won = 1.0 if r.outcome == "win" else 0.0
            self.wins += int(won)
            self.brier_model += (r.model_prob - won) ** 2
            self.brier_market += (r.consensus - won) ** 2
        self.pnl += r.pnl or 0.0
        if r.clv is not None and r.close_fresh:
            self.clv_sum += r.clv
            self.clv_n += 1
            self.clvs.append(r.clv)
        if r.close_consensus is not None and r.close_fresh:
            self.close_rois.append(promotion.close_roi(r.close_consensus, r.american))
            if r.outcome != "push":
                self.graded.append(
                    closecheck.Graded(
                        r.close_consensus, r.american, r.outcome == "win", r.pnl or 0.0
                    )
                )
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
    # Rows the unanchored sim alone would have bought (graded-only shadow).
    sim_buys_by_market: dict[str, Bucket] = field(default_factory=lambda: defaultdict(Bucket))

    def add(self, r: LedgerRow) -> None:
        self.by_market[r.market].add(r)
        if r.is_buy:
            self.buys_by_market[r.market].add(r)
            self.by_tier[r.tier].add(r)
        if r.sim_buy:
            self.sim_buys_by_market[r.market].add(r)
        for g in r.gates or ["pass_gate"]:
            self.by_gate[g].add(r)

    def probation(self) -> dict[str, dict[str, object]]:
        return self._probation(self.buys_by_market)

    def sim_probation(self) -> dict[str, dict[str, object]]:
        return self._probation(self.sim_buys_by_market)

    @staticmethod
    def _probation(buckets: dict[str, Bucket]) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for mk, b in sorted(buckets.items()):
            d = b.as_dict()
            p = promotion.read(
                mk,
                b.pnls,
                b.close_rois,
                float(d["brier_model"]) if b.decided else None,
                float(d["brier_market"]) if b.decided else None,
            )
            out[mk] = {
                **d,
                "cleared": p.cleared,
                "needed": p.needed,
                "status": p.status,
                "stake": p.stake,
                "close_n": p.close_n,
                "close_roi": p.close_roi,
                "close_roi_lo": p.close_roi_lo,
                "halves": p.halves,
                "why_not": p.why_not,
            }
        return out

    def promotions(self) -> list[promotion.Promotion]:
        """The live buy markets read against the shared bar, for the card."""
        out = []
        for mk, b in sorted(self.buys_by_market.items()):
            d = b.as_dict()
            out.append(
                promotion.read(
                    mk,
                    b.pnls,
                    b.close_rois,
                    float(d["brier_model"]) if b.decided else None,
                    float(d["brier_market"]) if b.decided else None,
                )
            )
        return out

    def closecheck(self) -> list[closecheck.BucketRead]:
        """The devigged close against results, by price range, over all buys."""
        return closecheck.read(g for b in self.buys_by_market.values() for g in b.graded)

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
        lines.append(
            "probation (buys per market; needs 100, roi>=0 both halves, close-implied ROI"
            " lower bound>0, brier<=market -> Lean (probation), 0.5u)"
        )
        for mk, p in self.probation().items():
            status = p["status"] if p["cleared"] else f"paper: {p['why_not']}"
            lines.append(
                f"  {mk:<15s} {p['n']!s:>4s}  roi {p['roi']:+.3f}  clv {p['clv']:+.4f}"
                f"  close_roi {p['close_roi']:+.4f} (lo {p['close_roi_lo']:+.4f}, n={p['close_n']})"
                f"  {status}"
            )
        lines.append("")
        lines.append("sim would-be buys (unanchored sim, graded-only; same bar)")
        for mk, p in self.sim_probation().items():
            status = p["status"] if p["cleared"] else f"paper: {p['why_not']}"
            lines.append(f"  {mk:<15s} {p['n']!s:>4s}  roi {p['roi']:+.3f}  {status}")
        checks = self.closecheck()
        if checks:
            lines.append("")
            lines.append(closecheck.TITLE)
            lines.append(
                "  "
                + "  ".join(
                    f"{h:<16s}" if i == 0 else f"{h:>12s}" for i, h in enumerate(closecheck.HEAD)
                )
            )
            for c in checks:
                cells = closecheck.row(c)
                lines.append(
                    "  "
                    + "  ".join(f"{c:<16s}" if i == 0 else f"{c:>12s}" for i, c in enumerate(cells))
                )
        return "\n".join(lines) + "\n"


def scorecard(rows: list[LedgerRow]) -> Scorecard:
    """Rows are read in slate-date order so the halves split old from new."""
    sc = Scorecard()
    for r in rows:
        if not r.priced_late:
            sc.add(r)
    return sc


__all__ = ["PROBATION_N", "Bucket", "Scorecard", "scorecard"]
