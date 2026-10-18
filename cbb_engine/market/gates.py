"""§3.7 gates and tiers for the CBB card. Every refusal is a named string on the row."""

from __future__ import annotations

from dataclasses import dataclass

from engine_common.odds import american_to_prob


@dataclass(frozen=True)
class Thresholds:
    min_ev: float = 0.0
    min_edge: float = 0.02
    strong_edge_gap: float = 0.03
    max_edge: float = 0.12
    min_paired_books: int = 3
    k_sampling_spread: float = 1.0


def ev_of(p: float, american: float) -> float:
    dec = 1 + (american / 100 if american > 0 else 100 / -american)
    return p * dec - 1


def tier(p: float, american: float, p_fair: float, thr: Thresholds, *, spread: float = 0.0) -> tuple[str, float, float, list[str]]:
    """Return (tier, edge, ev, gates). Edge is p − p_fair; the price has to pay too."""
    edge = p - p_fair
    ev = ev_of(p, american)
    gates: list[str] = []
    if ev <= thr.min_ev or edge < thr.min_edge:
        return "Pass", edge, ev, gates
    if edge > thr.max_edge:
        gates.append("max_edge")
        return "Pass", edge, ev, gates
    if spread and edge <= thr.k_sampling_spread * spread:
        return "Pass", edge, ev, gates
    return ("Strong buy" if edge >= thr.min_edge + thr.strong_edge_gap else "Moderate buy"), edge, ev, gates


def market_gates(*, paired_books: int, one_way: bool, exec_quote_present: bool, pending: bool, thr: Thresholds) -> list[str]:
    out: list[str] = []
    if paired_books < thr.min_paired_books:
        out.append("thin_market")
    if one_way:
        out.append("one_way_quote")
    if not exec_quote_present:
        out.append("stale_quote")
    if pending:
        out.append("pending")
    return out


def implied(american: float) -> float:
    return american_to_prob(american)


__all__ = ["Thresholds", "ev_of", "implied", "market_gates", "tier"]
