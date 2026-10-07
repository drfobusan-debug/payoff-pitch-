"""Weekly ledger audit: the past week and the whole ledger, read the same way.

Every segment carries its record, units, ROI with a 95% range, the rate its
prices demanded, how often it beat the close and how many bets its own ROI
would need before the range excludes zero -- so a hot or cold week is shown
against the sample it rests on, not argued from the record alone. The
accuracy table puts the model's probabilities beside the market's at bet time
and at the close on the same graded rows.

Nothing here is a model input; it is rebuilt from ``ledger.csv`` each run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta

from cfb_engine.audit.grade import LOSS, PUSH, WIN
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.market.tiers import Tier

Z = 1.96
BUY = frozenset({Tier.STRONG.value, Tier.MODERATE.value})
MIN_SEGMENT = 10  # smallest segment a finding will name
MIN_VERDICT = 30  # fewest decided bets that can be called beyond noise
_DEFAULT_DECIMAL = 1.91
ATS_BANDS: tuple[tuple[str, float, float], ...] = (
    ("ATS spread 0-3.5", 0.0, 3.5),
    ("ATS spread 4-7.5", 3.51, 7.5),
    ("ATS spread 8-14.5", 7.51, 14.5),
    ("ATS spread 15+", 14.51, 1e9),
)
EV_BANDS: tuple[tuple[str, float, float], ...] = (
    ("EV up to 3%", -1e9, 0.03),
    ("EV 3-5%", 0.03, 0.05),
    ("EV 5-8%", 0.05, 0.08),
    ("EV over 8%", 0.08, 1e9),
)


@dataclass(frozen=True)
class Segment:
    label: str
    n: int  # decided bets (pushes excluded)
    wins: int
    losses: int
    pushes: int
    units: float
    roi: float
    roi_lo: float
    roi_hi: float
    breakeven: float  # mean win rate the prices demanded
    clv_n: int
    beat_close: float  # share of priced-at-close bets that beat it
    clv_mean: float  # mean probability points gained on the close
    close_ev: float  # mean EV of the price taken, judged at the closing no-vig
    need: int | None  # decided bets this ROI needs for its range to exclude zero

    @property
    def record(self) -> str:
        base = f"{self.wins}-{self.losses}"
        return f"{base}-{self.pushes}" if self.pushes else base

    @property
    def win_pct(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def verdict(self) -> str:
        if self.n < 2:
            return "no sample"
        if self.n >= MIN_VERDICT and self.roi_lo > 0:
            return "winning beyond noise"
        if self.n >= MIN_VERDICT and self.roi_hi < 0:
            return "losing beyond noise"
        if self.need is None:
            return "underpowered"
        return f"underpowered (needs ~{self.need:,})"


@dataclass(frozen=True)
class Accuracy:
    """Brier scores (lower is better) on the same graded, non-push rows."""

    n: int
    model: float
    market: float
    n_close: int
    model_at_close: float
    close: float


@dataclass
class ScopeAudit:
    title: str
    start: Date | None
    end: Date
    slates: list[str]
    buys: Segment
    by_tier: list[Segment]
    by_market: list[Segment]
    splits: list[Segment]
    ev_bands: list[Segment]
    clv_split: list[Segment]
    passes: Segment
    accuracy: Accuracy
    periods: list[Segment]  # per slate (week) or per week (ledger)
    findings: list[str] = field(default_factory=list)


@dataclass
class WeeklyAudit:
    start: Date
    end: Date
    week: ScopeAudit
    ledger: ScopeAudit


def _decimal(e: LedgerEntry) -> float:
    return american_to_decimal(e.odds) if e.odds is not None else _DEFAULT_DECIMAL


def segment(label: str, rows: list[LedgerEntry]) -> Segment:
    decided = [e for e in rows if e.result in (WIN, LOSS)]
    pushes = sum(e.result == PUSH for e in rows)
    n = len(decided)
    pnl = [e.pnl for e in decided]
    units = sum(pnl)
    roi = units / n if n else 0.0
    breakeven = sum(1.0 / _decimal(e) for e in decided) / n if n else 0.0
    if n >= 2:
        # Never narrower than a coin at the prices' own break-even: a 7-0 run has
        # no spread of its own, and must not read as certainty.
        floor = math.sqrt((1.0 - breakeven) / breakeven) if 0 < breakeven < 1 else 0.0
        sd = max(math.sqrt(sum((p - roi) ** 2 for p in pnl) / (n - 1)), floor)
        half = Z * sd / math.sqrt(n)
        need = math.ceil((Z * sd / roi) ** 2) if roi else None
    else:
        sd, half, need = 0.0, 0.0, None
    priced = [e for e in rows if e.clv is not None and e.result in (WIN, LOSS, PUSH)]
    close_ev = [e.clv_ev for e in priced if e.clv_ev is not None]
    return Segment(
        label=label,
        n=n,
        wins=sum(e.result == WIN for e in decided),
        losses=sum(e.result == LOSS for e in decided),
        pushes=pushes,
        units=round(units, 3),
        roi=roi,
        roi_lo=roi - half,
        roi_hi=roi + half,
        breakeven=breakeven,
        clv_n=len(priced),
        beat_close=sum((e.clv or 0.0) > 0 for e in priced) / len(priced) if priced else 0.0,
        clv_mean=sum(e.clv or 0.0 for e in priced) / len(priced) if priced else 0.0,
        close_ev=sum(close_ev) / len(close_ev) if close_ev else 0.0,
        need=need,
    )


def _where(label: str, rows: list[LedgerEntry], keep: Callable[[LedgerEntry], bool]) -> Segment:
    return segment(label, [e for e in rows if keep(e)])


def _abs_line_between(lo: float, hi: float) -> Callable[[LedgerEntry], bool]:
    return lambda e: lo <= abs(e.line or 0.0) <= hi


def _ev_between(lo: float, hi: float) -> Callable[[LedgerEntry], bool]:
    return lambda e: lo < (e.ev or 0.0) <= hi


def _field_is(name: str, value: str) -> Callable[[LedgerEntry], bool]:
    if name == "tier":
        return lambda e: e.tier == value
    return lambda e: e.category == value


def _brier(pairs: list[tuple[float, float]]) -> float:
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def accuracy(rows: list[LedgerEntry]) -> Accuracy:
    decided = [e for e in rows if e.result in (WIN, LOSS)]
    priced = [e for e in decided if e.fair_prob is not None]
    closed = [e for e in decided if e.close_prob is not None]

    def won(e: LedgerEntry) -> float:
        return 1.0 if e.result == WIN else 0.0

    return Accuracy(
        n=len(priced),
        model=_brier([(e.model_prob, won(e)) for e in priced]),
        market=_brier([(e.fair_prob or 0.0, won(e)) for e in priced]),
        n_close=len(closed),
        model_at_close=_brier([(e.model_prob, won(e)) for e in closed]),
        close=_brier([(e.close_prob or 0.0, won(e)) for e in closed]),
    )


def _side(e: LedgerEntry) -> str:
    return e.selection.split(" ", 1)[0].lower()


def _splits(buys: list[LedgerEntry]) -> list[Segment]:
    tot = [e for e in buys if e.market == "game_total"]
    ats = [e for e in buys if e.market == "game_ats" and e.line is not None]
    ml = [e for e in buys if e.market == "game_ml" and e.odds is not None]
    out = [
        _where("Totals: Over", tot, lambda e: _side(e) == "over"),
        _where("Totals: Under", tot, lambda e: _side(e) == "under"),
        _where("ATS: favorite", ats, lambda e: (e.line or 0.0) < 0),
        _where("ATS: underdog", ats, lambda e: (e.line or 0.0) > 0),
    ]
    for label, lo, hi in ATS_BANDS:
        out.append(_where(label, ats, _abs_line_between(lo, hi)))
    out += [
        _where("ML: favorite", ml, lambda e: (e.odds or 0.0) < 0),
        _where("ML: underdog", ml, lambda e: (e.odds or 0.0) > 0),
    ]
    return [s for s in out if s.n or s.pushes]


def _ev_bands(buys: list[LedgerEntry]) -> list[Segment]:
    rows = [e for e in buys if e.ev is not None]
    out = [_where(label, rows, _ev_between(lo, hi)) for label, lo, hi in EV_BANDS]
    return [s for s in out if s.n or s.pushes]


def _clv_split(buys: list[LedgerEntry]) -> list[Segment]:
    priced = [e for e in buys if e.clv is not None]
    out = [
        _where("Beat the close", priced, lambda e: (e.clv or 0.0) > 0),
        _where("Matched or lost to the close", priced, lambda e: (e.clv or 0.0) <= 0),
    ]
    return [s for s in out if s.n or s.pushes]


def _pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def findings(scope: ScopeAudit) -> list[str]:
    b, acc = scope.buys, scope.accuracy
    out: list[str] = []
    if b.n < 2:
        return ["Too few graded buys to say anything."]
    if b.verdict == "winning beyond noise":
        out.append(
            f"Buys are {b.record}, {b.units:+.1f}u, ROI {_pct(b.roi)} "
            f"(95% range {_pct(b.roi_lo)} to {_pct(b.roi_hi)}): winning beyond noise."
        )
    elif b.verdict == "losing beyond noise":
        out.append(
            f"Buys are {b.record}, {b.units:+.1f}u, ROI {_pct(b.roi)} "
            f"(95% range {_pct(b.roi_lo)} to {_pct(b.roi_hi)}): losing beyond noise."
        )
    else:
        need = f"; at this ROI it needs about {b.need:,} bets to be proven" if b.need else ""
        out.append(
            f"Buys are {b.record}, {b.units:+.1f}u, ROI {_pct(b.roi)}, but the 95% range "
            f"runs {_pct(b.roi_lo)} to {_pct(b.roi_hi)}: not proven ({b.n} bets{need})."
        )
    if acc.n:
        who = "more" if acc.market < acc.model else "less"
        out.append(
            f"The bet-time market was {who} accurate than the model "
            f"(Brier {acc.market:.4f} vs {acc.model:.4f} on {acc.n} graded rows; lower is better)."
        )
    if acc.n_close:
        who = "more" if acc.close < acc.model_at_close else "less"
        out.append(
            f"The closing line was {who} accurate than the model "
            f"(Brier {acc.close:.4f} vs {acc.model_at_close:.4f} on {acc.n_close} rows)."
        )
    if b.clv_n:
        tail = ""
        if b.close_ev < 0 < b.units:
            tail = ", so the profit is running ahead of what the prices justify (likely variance)"
        elif b.close_ev > 0:
            tail = ", which is the sign of a real edge if it holds"
        out.append(
            f"Buys beat the close {b.beat_close:.0%} of the time by {b.clv_mean * 100:+.2f} "
            f"probability points on average; judged at the closing price the average buy was "
            f"worth {_pct(b.close_ev)}{tail}."
        )
    named = [s for s in scope.by_market + scope.splits if s.n >= MIN_SEGMENT]
    if named:
        worst = min(named, key=lambda s: s.roi)
        best = max(named, key=lambda s: s.roi)
        if worst.roi < 0:
            out.append(
                f"Weakest segment: {worst.label} {worst.record}, {worst.units:+.1f}u "
                f"({_pct(worst.roi)}), {worst.verdict}."
            )
        if best.roi > 0 and best.label != worst.label:
            luck = (
                f"; it is the best of {len(named)} segments checked, so part of that is "
                "selection luck"
                if best.verdict == "winning beyond noise"
                else ""
            )
            out.append(
                f"Strongest segment: {best.label} {best.record}, {best.units:+.1f}u "
                f"({_pct(best.roi)}), {best.verdict}{luck}."
            )
    return out


def _week_label(start: Date, end: Date) -> str:
    return f"{start:%b} {start.day}-{end:%b} {end.day}"


def _periods_by_slate(rows: list[LedgerEntry]) -> list[Segment]:
    dates = sorted({e.date for e in rows})
    return [segment(d, [e for e in rows if e.date == d]) for d in dates]


def _periods_by_week(rows: list[LedgerEntry], end: Date) -> list[Segment]:
    if not rows:
        return []
    first = min(Date.fromisoformat(e.date) for e in rows)
    out: list[Segment] = []
    hi = end
    while hi >= first:
        lo = hi - timedelta(days=6)
        span = [e for e in rows if lo <= Date.fromisoformat(e.date) <= hi]
        if span:
            out.append(segment(_week_label(lo, hi), span))
        hi = lo - timedelta(days=1)
    return out[::-1]


def scope_audit(
    title: str, rows: list[LedgerEntry], start: Date | None, end: Date, *, weekly: bool
) -> ScopeAudit:
    buys = [e for e in rows if e.tier in BUY]
    markets = sorted({e.category for e in buys})
    scope = ScopeAudit(
        title=title,
        start=start,
        end=end,
        slates=sorted({e.date for e in rows}),
        buys=segment("All buys", buys),
        by_tier=[
            _where(t.value, buys, _field_is("tier", t.value)) for t in (Tier.STRONG, Tier.MODERATE)
        ],
        by_market=[_where(m, buys, _field_is("category", m)) for m in markets],
        splits=_splits(buys),
        ev_bands=_ev_bands(buys),
        clv_split=_clv_split(buys),
        passes=_where("Pass rows (not bet)", rows, lambda e: e.tier == Tier.PASS.value),
        accuracy=accuracy(rows),
        periods=_periods_by_slate(buys) if weekly else _periods_by_week(buys, end),
    )
    scope.findings = findings(scope)
    return scope


def weekly_audit(entries: list[LedgerEntry], end: Date, days: int = 7) -> WeeklyAudit:
    """The ``days`` slates ending ``end`` (inclusive), and the ledger through ``end``."""
    start = end - timedelta(days=days - 1)
    to_date = [e for e in entries if Date.fromisoformat(e.date) <= end]
    week = [e for e in to_date if Date.fromisoformat(e.date) >= start]
    return WeeklyAudit(
        start=start,
        end=end,
        week=scope_audit(f"This week ({_week_label(start, end)})", week, start, end, weekly=True),
        ledger=scope_audit("Ledger to date", to_date, None, end, weekly=False),
    )
