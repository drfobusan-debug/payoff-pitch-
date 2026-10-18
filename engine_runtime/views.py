"""The audit views over the ledger grain. The views are the analysis; a PDF renders them.

Each function returns rows as dicts so a report, a test and a notebook read the
same numbers. Interval reporting is the caller's (slate-day clustered
bootstrap in the sport's scorecard); these views only aggregate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import mean

from engine_runtime.ledger import Row


def v_clv(rows: Iterable[Row]) -> dict[str, object]:
    """Live bet-of-record rows: consensus and execution CLV, fill-verified and all."""
    live = [r for r in rows if r.status == "live" and r.bet_of_record and r.is_buy]
    filled = [r for r in live if r.placed == "Y"]

    def cents(xs: list[Row], attr: str) -> float | None:
        vals = [getattr(x, attr) for x in xs if getattr(x, attr) is not None]
        return round(mean(vals), 4) if vals else None

    return {
        "n": len(live),
        "n_filled": len(filled),
        "n_unknown": sum(1 for r in live if r.placed == "unknown"),
        "clv_consensus": cents(live, "clv_consensus"),
        "clv_exec": cents(live, "clv_exec"),
        "clv_consensus_filled": cents(filled, "clv_consensus"),
        "clv_exec_filled": cents(filled, "clv_exec"),
    }


def v_shape(rows: Iterable[Row]) -> list[dict[str, object]]:
    """Predicted vs realised push mass by number on live totals."""
    by_line: dict[float, list[Row]] = defaultdict(list)
    for r in rows:
        if r.status == "live" and r.market.endswith("total") and r.line is not None and r.outcome:
            by_line[r.line].append(r)
    out: list[dict[str, object]] = []
    for line, xs in sorted(by_line.items()):
        pred = [x.push_mass for x in xs if x.push_mass is not None]
        out.append(
            {
                "line": line,
                "n": len(xs),
                "pred_push": round(mean(pred), 4) if pred else None,
                "real_push": round(sum(1 for x in xs if x.outcome == "push") / len(xs), 4),
            }
        )
    return out


def v_rung(rows: Iterable[Row], rungs: list[str]) -> list[dict[str, object]]:
    """Mean incremental edge each named rung added over the rung beneath (live + shadow)."""
    out = []
    for rung in rungs:
        vals = [r.edge_by_rung[rung] for r in rows if rung in r.edge_by_rung]
        out.append(
            {"rung": rung, "n": len(vals), "mean_increment": round(mean(vals), 5) if vals else None}
        )
    return out


def v_gate(rows: Iterable[Row]) -> list[dict[str, object]]:
    """Refused-row CLV by gate x market x status, the counterfactual the gate is graded on."""
    groups: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for r in rows:
        for g in r.gates:
            groups[(g, r.market, r.status)].append(r)
    out = []
    for (gate, market, status), xs in sorted(groups.items()):
        vals = [x.clv_consensus for x in xs if x.clv_consensus is not None]
        out.append(
            {
                "gate": gate,
                "market": market,
                "status": status,
                "n": len(xs),
                "clv_consensus": round(mean(vals), 4) if vals else None,
            }
        )
    return out


def v_prior(rows: Iterable[Row]) -> list[dict[str, object]]:
    """Mean stored log-score per futures-prior variant on the rows that carry them."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for name, score in r.prior_variants.items():
            acc[name].append(score)
    return [
        {"variant": k, "n": len(v), "mean_score": round(mean(v), 5)} for k, v in sorted(acc.items())
    ]


__all__ = ["v_clv", "v_gate", "v_prior", "v_rung", "v_shape"]
