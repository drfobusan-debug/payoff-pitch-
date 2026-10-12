"""Weekly ledger audit: the past week and the whole ledger, read the same way.

Every segment carries its record, units, ROI with a 95% range, the rate its
prices demanded, how often it beat the close and how many bets its own ROI
would need before the range excludes zero -- so a hot or cold week is shown
against the sample it rests on, not argued from the record alone. The price is
read first: EV at the close beside the juice the same prices carried at bet
time, then a game-level Brier per market (model vs bet-time market vs close),
and only then the record.

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
    bet_ev: float  # mean EV of the same prices at the bet-time no-vig: the juice paid
    clv_pts_n: int
    clv_pts: float  # mean points gained on the closing spread/total
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
class GameBrier:
    """One market scored once per game, on the favorite (ML, spread) or the over.

    Lower is better. ``market`` is the bet-time no-vig; ``close`` is on the
    subset of games with a captured close, beside the model on that subset.
    """

    label: str
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
    briers: list[GameBrier]
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


def _slate_sd(decided: list[LedgerEntry], roi: float) -> float:
    """Per-bet SD implied by resampling whole slates (0 under two slates).

    A Saturday's buys share news, weather and a market mood, so they win and
    lose together; treating them as independent understates the range.
    """
    by_slate: dict[str, float] = {}
    for e in decided:
        by_slate[e.date] = by_slate.get(e.date, 0.0) + e.pnl - roi
    g = len(by_slate)
    if g < 2:
        return 0.0
    var_sum = g / (g - 1) * sum(v * v for v in by_slate.values())
    return math.sqrt(var_sum / len(decided))


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
        sd = max(
            math.sqrt(sum((p - roi) ** 2 for p in pnl) / (n - 1)), floor, _slate_sd(decided, roi)
        )
        half = Z * sd / math.sqrt(n)
        need = math.ceil((Z * sd / roi) ** 2) if roi else None
    else:
        sd, half, need = 0.0, 0.0, None
    priced = [e for e in rows if e.clv is not None and e.result in (WIN, LOSS, PUSH)]
    close_ev = [e.clv_ev for e in priced if e.clv_ev is not None]
    bet_ev = [
        e.fair_prob * _decimal(e) - 1.0
        for e in priced
        if e.clv_ev is not None and e.fair_prob is not None
    ]
    pts = [e.clv_pts for e in priced if e.clv_pts is not None]
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
        bet_ev=sum(bet_ev) / len(bet_ev) if bet_ev else 0.0,
        clv_pts_n=len(pts),
        clv_pts=sum(pts) / len(pts) if pts else 0.0,
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


BRIER_MARKETS: tuple[tuple[str, str], ...] = (
    ("game_ml", "Moneyline (favorite)"),
    ("game_ats", "Spread (favorite)"),
    ("game_total", "Total (over)"),
)


def _favorite_side(e: LedgerEntry) -> bool:
    if e.market == "game_total":
        return _side(e) == "over"
    if e.market == "game_ats" and e.line:
        return e.line < 0
    return (e.fair_prob or 0.0) >= 0.5


def _game_rows(
    rows: list[LedgerEntry], market: str
) -> list[tuple[float, float, float | None, float]]:
    """(model, market, close, outcome) once per game, oriented to the favorite/over.

    A game whose favorite row is missing is read off the other side, flipped.
    """
    games: dict[tuple[str, str], list[LedgerEntry]] = {}
    for e in rows:
        if e.market == market and e.result in (WIN, LOSS) and e.fair_prob is not None:
            games.setdefault((e.date, e.matchup), []).append(e)
    out: list[tuple[float, float, float | None, float]] = []
    for sides in games.values():
        fav = next((e for e in sides if _favorite_side(e)), None)
        if fav is not None:
            y = 1.0 if fav.result == WIN else 0.0
            out.append((fav.model_prob, fav.fair_prob or 0.0, fav.close_prob, y))
            continue
        o = sides[0]
        y = 0.0 if o.result == WIN else 1.0
        close = 1.0 - o.close_prob if o.close_prob is not None else None
        out.append((1.0 - o.model_prob, 1.0 - (o.fair_prob or 0.0), close, y))
    return out


def game_briers(rows: list[LedgerEntry]) -> list[GameBrier]:
    out: list[GameBrier] = []
    for market, label in BRIER_MARKETS:
        g = _game_rows(rows, market)
        closed = [(m, c, y) for m, _, c, y in g if c is not None]
        out.append(
            GameBrier(
                label=label,
                n=len(g),
                model=_brier([(m, y) for m, _, _, y in g]),
                market=_brier([(k, y) for _, k, _, y in g]),
                n_close=len(closed),
                model_at_close=_brier([(m, y) for m, _, y in closed]),
                close=_brier([(c, y) for _, c, y in closed]),
            )
        )
    return out


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


def _cents(v: float) -> str:
    return f"{v * 100:+.1f}c"


def close_finding(b: Segment) -> str | None:
    if not b.clv_n:
        return None
    moved = b.close_ev - b.bet_ev
    if b.close_ev > 0:
        verdict = "the close paid more than the juice: the sign of a real edge if it holds"
    elif moved > 0:
        verdict = "the close moved toward the bets, but not enough to pay the juice"
    else:
        verdict = "the close did not move toward the bets"
    pts = f", {b.clv_pts:+.2f} points on {b.clv_pts_n} spreads/totals" if b.clv_pts_n else ""
    return (
        f"Judged at the closing price the average buy was worth {_cents(b.close_ev)} per dollar, "
        f"against {_cents(b.bet_ev)} for the same prices at bet time (the juice), so "
        f"{verdict} ({b.clv_n} buys with a close; beat it {b.beat_close:.0%}, "
        f"{b.clv_mean * 100:+.2f} probability points{pts})."
    )


def brier_finding(briers: list[GameBrier]) -> str | None:
    scored = [g for g in briers if g.n]
    if not scored:
        return None
    parts = "; ".join(
        f"{g.label.split(' ')[0].lower()} {g.model:.3f} vs {g.market:.3f} ({g.n} games)"
        for g in scored
    )
    ahead = sum(g.market < g.model for g in scored)
    who = (
        "The bet-time market was more accurate than the model"
        if ahead == len(scored)
        else "The model was more accurate than the bet-time market"
        if ahead == 0
        else f"The bet-time market was more accurate than the model on {ahead} of {len(scored)} markets"
    )
    return f"{who} (Brier, model vs market, lower is better: {parts})."


def findings(scope: ScopeAudit) -> list[str]:
    b = scope.buys
    out: list[str] = []
    if b.n < 2:
        return ["Too few graded buys to say anything."]
    for line in (close_finding(b), brier_finding(scope.briers)):
        if line:
            out.append(line)
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
    if b.clv_n and b.close_ev < 0 < b.units:
        out.append(
            "The profit is running ahead of what the prices justify at the close (likely "
            "variance): a record read beyond noise on its own is not an edge the close confirms."
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
        briers=game_briers(rows),
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
