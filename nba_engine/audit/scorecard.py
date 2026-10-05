"""What the graded ledger proves, by the cut each decision is judged on (plan §7, §11).

Scope is every graded row priced at an execution book. PPV and NPV are scored
against the base rate of that scope, not against zero (MLB #105): a 52% hit
rate is no skill on a market whose two sides each win half the time. A gate's
row reads its refusals as bets, so its win rate and ROI are the false negatives
it cost. CLV uses only rows with a real pre-tip close at the book; pulled and
moved rows report ``pre_pull_clv`` in its own columns. Intervals bootstrap
whole games, since one game's rows are not independent draws.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from engine_common.odds import american_to_prob
from nba_engine.audit.ledger import CLOSE, MOVED, PULLED, LedgerRow
from nba_engine.audit.settle import ABSENT, LOSS, PUSH, VOID, WIN
from nba_engine.data.oddsapi import parse_utc


@dataclass(frozen=True)
class Metrics:
    label: str
    rows: int  # every graded row in scope, both classes
    n: int  # positive rows graded win/loss/push
    wins: int
    losses: int
    pushes: int
    voids: int
    win_pct: float
    required_win_pct: float  # mean break-even of the prices taken
    base_rate: float
    ppv: float
    npv: float
    ppv_lift: float
    npv_lift: float
    sensitivity: float
    specificity: float
    units: float
    roi: float
    roi_lo: float
    roi_hi: float
    clv_n: int
    mean_clv: float
    clv_beat_pct: float
    mean_clv_ev: float
    clv_ev_lo: float
    clv_ev_hi: float
    pulled: int
    moved: int
    pre_pull_n: int
    mean_pre_pull_clv: float
    brier_model: float | None  # on positive rows that carry a model probability
    brier_market: float | None  # the same rows, at the consensus fair


def _safe(num: float, den: float) -> float:
    return round(num / den, 4) if den else 0.0


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 5) if xs else 0.0


def clustered_ci(
    rows: list[LedgerRow],
    value: Callable[[LedgerRow], float | None],
    *,
    draws: int = 400,
    seed: int = 1,
) -> tuple[float, float]:
    """95% interval of the mean of ``value``, resampling whole games."""
    by_game: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = value(r)
        if v is not None:
            by_game[r.event_id].append(v)
    games = sorted(by_game)
    if len(games) < 2:
        return 0.0, 0.0
    rnd = random.Random(seed)
    sims: list[float] = []
    for _ in range(draws):
        vals = [v for g in rnd.choices(games, k=len(games)) for v in by_game[g]]
        sims.append(sum(vals) / len(vals))
    sims.sort()
    return round(sims[int(0.025 * draws)], 5), round(sims[min(draws - 1, int(0.975 * draws))], 5)


def in_scope(rows: Iterable[LedgerRow]) -> list[LedgerRow]:
    return [r for r in rows if r.american is not None and r.outcome in (WIN, LOSS, PUSH, VOID)]


def metrics(
    rows: Iterable[LedgerRow],
    is_positive: Callable[[LedgerRow], bool],
    label: str,
    *,
    draws: int = 400,
) -> Metrics:
    scope = in_scope(rows)
    decided = [r for r in scope if r.outcome in (WIN, LOSS)]
    pos = [r for r in scope if is_positive(r)]
    tp = sum(1 for r in decided if is_positive(r) and r.outcome == WIN)
    fp = sum(1 for r in decided if is_positive(r) and r.outcome == LOSS)
    fn = sum(1 for r in decided if not is_positive(r) and r.outcome == WIN)
    tn = sum(1 for r in decided if not is_positive(r) and r.outcome == LOSS)
    staked = [r for r in pos if r.outcome in (WIN, LOSS, PUSH)]
    units = sum(r.pnl or 0.0 for r in staked)
    breakeven = [american_to_prob(r.american) for r in staked if r.american is not None]
    base = _safe(tp + fn, len(decided))
    ppv, npv = _safe(tp, tp + fp), _safe(tn, tn + fn)
    roi_lo, roi_hi = clustered_ci(staked, lambda r: r.pnl, draws=draws)
    closed = [r for r in staked if r.close_status == CLOSE and r.clv is not None]
    with_ev = [r for r in closed if r.clv_ev is not None]
    ev_lo, ev_hi = clustered_ci(with_ev, lambda r: r.clv_ev, draws=draws)
    pre = [r.pre_pull_clv for r in staked if r.pre_pull_clv is not None]
    modelled = [r for r in decided if is_positive(r) and r.model_prob is not None and r.fair]
    return Metrics(
        label=label,
        rows=len(scope),
        n=len(staked),
        wins=tp,
        losses=fp,
        pushes=sum(1 for r in staked if r.outcome == PUSH),
        voids=sum(1 for r in pos if r.outcome == VOID),
        win_pct=ppv,
        required_win_pct=_safe(sum(breakeven), len(breakeven)),
        base_rate=base,
        ppv=ppv,
        npv=npv,
        ppv_lift=round(ppv - base, 4) if tp + fp else 0.0,
        npv_lift=round(npv - (1.0 - base), 4) if tn + fn else 0.0,
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        units=round(units, 3),
        roi=_safe(units, len(staked)),
        roi_lo=roi_lo,
        roi_hi=roi_hi,
        clv_n=len(closed),
        mean_clv=_mean([r.clv for r in closed if r.clv is not None]),
        clv_beat_pct=_safe(sum(1 for r in closed if (r.clv or 0.0) > 0), len(closed)),
        mean_clv_ev=_mean([r.clv_ev for r in with_ev if r.clv_ev is not None]),
        clv_ev_lo=ev_lo,
        clv_ev_hi=ev_hi,
        pulled=sum(1 for r in staked if r.close_status == PULLED),
        moved=sum(1 for r in staked if r.close_status == MOVED),
        pre_pull_n=len(pre),
        mean_pre_pull_clv=_mean(pre),
        brier_model=_brier(modelled, lambda r: r.model_prob),
        brier_market=_brier(modelled, lambda r: r.fair),
    )


def _brier(rows: list[LedgerRow], prob: Callable[[LedgerRow], float | None]) -> float | None:
    vals = [
        (p - (1.0 if r.outcome == WIN else 0.0)) ** 2 for r in rows if (p := prob(r)) is not None
    ]
    return round(sum(vals) / len(vals), 5) if vals else None


def _buy(r: LedgerRow) -> bool:
    return r.is_buy


def _all(_: LedgerRow) -> bool:
    return True


def _tier_is(tier: str) -> Callable[[LedgerRow], bool]:
    return lambda r: r.tier == tier


def _refused_by(gate: str) -> Callable[[LedgerRow], bool]:
    return lambda r: gate in r.gates.split(";")


def tables(rows: Iterable[LedgerRow], *, draws: int = 400) -> dict[str, list[Metrics]]:
    """Every audit cut: buys, tiers, markets, gates, execution book, one/two-book, board."""
    rows = list(rows)
    out: dict[str, list[Metrics]] = {"buys": [metrics(rows, _buy, "buys", draws=draws)]}
    out["tier"] = [
        metrics(rows, _tier_is(t), f"tier:{t}", draws=draws) for t in sorted({r.tier for r in rows})
    ]
    markets = sorted({r.market for r in rows})
    out["market"] = [
        metrics([r for r in rows if r.market == m], _buy, f"buys:{m}", draws=draws) for m in markets
    ]
    gates = sorted({g for r in rows for g in r.gates.split(";") if g})
    out["gate"] = [metrics(rows, _refused_by(g), f"refused:{g}", draws=draws) for g in gates]
    out["book"] = [
        metrics([r for r in rows if r.book == b], _buy, f"buys@{b}", draws=draws)
        for b in sorted({r.book for r in rows if r.book})
    ]
    out["exec_books"] = [
        metrics([r for r in rows if r.exec_books == k], _buy, f"buys:{k}-book", draws=draws)
        for k in (1, 2)
    ]
    out["board"] = [
        metrics([r for r in rows if r.market == m], _all, f"board:{m}", draws=draws)
        for m in markets
    ]
    return out


def calibration(
    rows: Iterable[LedgerRow],
    prob: Callable[[LedgerRow], float | None],
    *,
    edges: tuple[float, ...] = (0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0),
) -> list[tuple[str, int, float, float]]:
    """(bin, n, mean predicted, hit rate) over decided rows."""
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for r in in_scope(rows):
        p = prob(r)
        if p is None or r.outcome not in (WIN, LOSS):
            continue
        i = next(k for k in range(len(edges) - 1) if p < edges[k + 1] or k == len(edges) - 2)
        bins[i].append((p, int(r.outcome == WIN)))
    out = []
    for i in sorted(bins):
        vals = bins[i]
        out.append(
            (
                f"{edges[i]:.2f}-{edges[i + 1]:.2f}",
                len(vals),
                round(sum(p for p, _ in vals) / len(vals), 4),
                round(sum(h for _, h in vals) / len(vals), 4),
            )
        )
    return out


@dataclass(frozen=True)
class Integrity:
    rows: int
    positions: int
    duplicates: int  # rows sharing a position on one slate: must be 0
    priced_after_tip: int  # look-ahead: must be 0
    close_before_price: int  # a close older than the price taken: must be 0
    ungraded: int  # no final yet, or a player name the box does not carry
    voids: int
    voids_by_reason: dict[str, int]
    no_exec_book: int
    no_close: int
    pulled: int
    moved: int
    paper: int
    by_outcome: dict[str, int]


def absent_names(rows: Iterable[LedgerRow], top: int = 15) -> list[tuple[str, int]]:
    """Players voided as missing from the box, by games: a regular here is a misspelling."""
    games = {(r.entity, r.event_id) for r in rows if r.void_reason == ABSENT}
    return Counter(e for e, _ in games).most_common(top)


def integrity(rows: Iterable[LedgerRow]) -> Integrity:
    rows = list(rows)
    seen = Counter((r.slate_date, *r.position) for r in rows)
    after_tip = 0
    for r in rows:
        tip, at = parse_utc(r.tip_utc), parse_utc(r.priced_at)
        if tip is not None and at is not None and at >= tip:
            after_tip += 1
    priced = [r for r in rows if r.american is not None]
    return Integrity(
        rows=len(rows),
        positions=len(seen),
        duplicates=sum(n - 1 for n in seen.values()),
        priced_after_tip=after_tip,
        close_before_price=sum(
            1 for r in priced if r.close_captured_at and r.close_captured_at < r.priced_at
        ),
        ungraded=sum(1 for r in rows if not r.outcome),
        voids=sum(1 for r in rows if r.outcome == VOID),
        voids_by_reason=dict(Counter(r.void_reason for r in rows if r.outcome == VOID)),
        no_exec_book=len(rows) - len(priced),
        no_close=sum(1 for r in priced if not r.close_status),
        pulled=sum(1 for r in rows if r.close_status == PULLED),
        moved=sum(1 for r in rows if r.close_status == MOVED),
        paper=sum(1 for r in rows if r.mode == "paper"),
        by_outcome=dict(Counter(r.outcome or "ungraded" for r in rows)),
    )


__all__ = [
    "Integrity",
    "Metrics",
    "calibration",
    "clustered_ci",
    "in_scope",
    "integrity",
    "metrics",
    "tables",
]
