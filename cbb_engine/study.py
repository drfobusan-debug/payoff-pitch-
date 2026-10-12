"""Phase 0b/3 studies computed from the tape, no credits: gate funnel, audit cut 10, time-travel leakage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from statistics import mean, pstdev

from engine_runtime import ledger, quotes

GATE_ORDER = (
    "starter_unknown", "one_way_quote", "settlement_unverified", "ml_sharp", "drift", "pending",
    "thin_market", "stale_quote", "width_unfit", "anchor_miss", "se_cap", "halves_sum",
    "team_totals_sum", "monotone_alt", "win_vs_cover", "exposure_cap",
)


def funnel(rows: Iterable[ledger.Row]) -> list[dict[str, object]]:
    """Games surviving each gate in GATE_ORDER, by market x quality bucket (best row per event)."""
    best: dict[tuple[str, str, str], ledger.Row] = {}
    for r in rows:
        if not r.bet_of_record:
            continue
        k = (r.market, r.quality_bucket, r.event_id)
        if k not in best or len(r.gates) < len(best[k].gates):
            best[k] = r
    groups: dict[tuple[str, str], list[ledger.Row]] = defaultdict(list)
    for (market, bucket, _), r in best.items():
        groups[(market, bucket)].append(r)
    out = []
    for (market, bucket), xs in sorted(groups.items()):
        alive = list(xs)
        steps: dict[str, int] = {"entered": len(alive)}
        for g in GATE_ORDER:
            alive = [r for r in alive if g not in r.gates]
            steps[g] = len(alive)
        steps["buys"] = sum(1 for r in alive if r.is_buy)
        out.append({"market": market, "quality_bucket": bucket, **steps})
    return out


def implied(american: float) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def subset_15(shards: Iterable[quotes.Quote]) -> set[str]:
    """Featured pass ids a 15-minute clock would have produced: greedy, >=15 min apart from the first."""
    stamps = sorted({(q.captured_at, q.pass_id) for q in shards if q.pass_kind == "featured"})
    keep: set[str] = set()
    last: datetime | None = None
    for ts, pid in stamps:
        t = _ts(ts)
        if last is None or (t - last).total_seconds() >= 15 * 60:
            keep.add(pid)
            last = t
    return keep


def featured_value(rows: Iterable[ledger.Row], shards: list[quotes.Quote]) -> list[dict[str, object]]:
    """Cut 10: price taken on a 10-min featured pass vs the same quote at the last 15-min-subset pass.

    Delta is in implied-probability points (bet price minus counterfactual price, so + means
    the extra pass bought a better number); against one close both CLV columns move by exactly
    this delta. `mde_80` is the per-market effect detectable at 80% power / 5% two-sided
    with the n in hand — a result inside it is "underpowered", not a verdict.
    """
    keep = subset_15(shards)
    by_pass: dict[str, list[quotes.Quote]] = defaultdict(list)
    for q in shards:
        by_pass[q.pass_id].append(q)
    times = sorted((qs[0].captured_at, pid) for pid, qs in by_pass.items() if pid in keep)
    deltas: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not (r.bet_of_record and r.is_buy and r.status == "live" and r.quote_shard in by_pass):
            continue
        t0 = by_pass[r.quote_shard][0].captured_at
        prior = [pid for ts, pid in times if ts <= t0]
        if not prior or r.american is None:
            continue
        cf = next((q for q in by_pass[prior[-1]] if (q.event_id, q.market, q.side, q.line, q.book) == (r.event_id, r.market, r.side, r.line, r.book)), None)
        if cf is None:
            continue
        deltas[r.market].append((implied(cf.american) - implied(r.american)) * 100)
    out = []
    for market, d in sorted(deltas.items()):
        sd = pstdev(d) if len(d) > 1 else None
        out.append({"market": market, "n": len(d), "delta_pts": round(mean(d), 3), "sd": round(sd, 3) if sd is not None else None, "mde_80": round(2.8 * sd / len(d) ** 0.5, 3) if sd else None})
    return out


def time_travel(as_of: str, stamps: Iterable[tuple[str, str]]) -> list[str]:
    """Integrity check: every (source, ISO stamp) an as-of replay used must be <= `as_of` (a date)."""
    cutoff = datetime.fromisoformat(as_of + "T23:59:59+00:00")
    return [f"{src} {stamp}" for src, stamp in stamps if _ts(stamp) > cutoff]


__all__ = ["GATE_ORDER", "featured_value", "funnel", "subset_15", "time_travel"]
