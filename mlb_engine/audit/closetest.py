"""When may an MLB market be bought? The NHL close test and the NFL graduation bar.

**Close test** (NHL ``closetest``), fixed before the season, per market:

    (result - p_close) = b * (p_model - p_close)      OLS, no intercept

``b`` is how much of the model's disagreement with the close showed up in
results: 0 means none of it, 1 means all of it. |b| <= 1 SE, or b < 0, keeps
the market graded-only; b > 1 SE supports an anchor of about ``1 - b`` and
probation. Tickets on one game win and lose together, so the SE is clustered
by game. On the 2026 ledger b was about +0.05 on batter props and 0.00 on game
lines: the disagreement the engine bought was noise.

**Graduation** (NFL ``graduation``), per market, on the would-be buys a
graded-only market records (``pass_gate`` ``graded_only_<market>``) and any
real buys: all four tests are required before a market is bought again.

1. Volume: ``MIN_N`` graded win/loss rows, one per selection per slate.
2. Margin: ROI above zero by more than one SE.
3. Consistency: both chronological halves profitable.
4. CLV: mean EV at the close above zero by more than one SE, on ``MIN_N``
   rows whose close was judged ``clean`` (a timely close, see ``clv``).

Display only: neither result changes a price or a bet.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean, stdev

from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.audit.ledger import ENGINE, LedgerEntry, one_side_per_prop
from mlb_engine.market.tiers import Tier

MIN_N = 100
GRADED_ONLY_PREFIX = "graded_only_"
_BUY = frozenset({Tier.STRONG.value, Tier.MODERATE.value})

WATCHING = "WATCHING"
NOT_YET = "NOT YET"
MEETS_BAR = "MEETS BAR"


def family(market: str) -> str:
    if market.startswith("batter_"):
        return "batter"
    if market.startswith("pitcher_"):
        return "pitcher"
    return "game"


def _graded(entries: Iterable[LedgerEntry]) -> list[LedgerEntry]:
    rows = [e for e in entries if e.source == ENGINE and e.result in (WIN, LOSS)]
    return one_side_per_prop(rows)


@dataclass(frozen=True)
class CloseFit:
    label: str
    n: int
    games: int
    slope: float
    se: float
    brier_model: float
    brier_close: float

    @property
    def verdict(self) -> str:
        if self.games < 2 or not math.isfinite(self.se):
            return "underpowered"
        if abs(self.slope) <= self.se:
            return "graded-only (b within 1 SE of 0)"
        if self.slope < 0:
            return "graded-only (b negative)"
        return f"anchor ~{max(0.0, min(1.0, 1.0 - self.slope)):.2f}, probation"


def close_fit(entries: Iterable[LedgerEntry], label: str) -> CloseFit:
    """The no-intercept slope of (result - close) on (model - close), game-clustered SE."""
    rows = [e for e in entries if e.close_prob is not None and 0.0 < e.close_prob < 1.0]
    x = [e.model_prob - (e.close_prob or 0.0) for e in rows]
    y = [(1.0 if e.result == WIN else 0.0) - (e.close_prob or 0.0) for e in rows]
    n = len(rows)
    sxx = sum(v * v for v in x)
    if not n or sxx <= 0:
        return CloseFit(label, n, 0, math.nan, math.nan, math.nan, math.nan)
    b = sum(a * c for a, c in zip(x, y, strict=True)) / sxx
    by_game: dict[tuple[str, str], float] = {}
    for e, a, c in zip(rows, x, y, strict=True):
        key = (e.date, e.matchup)
        by_game[key] = by_game.get(key, 0.0) + a * (c - b * a)
    g = len(by_game)
    adj = g / (g - 1) if g > 1 else math.nan
    se = math.sqrt(adj * sum(s * s for s in by_game.values())) / sxx if g > 1 else math.nan
    won = [1.0 if e.result == WIN else 0.0 for e in rows]
    bm = fmean((e.model_prob - w) ** 2 for e, w in zip(rows, won, strict=True))
    bc = fmean(((e.close_prob or 0.0) - w) ** 2 for e, w in zip(rows, won, strict=True))
    return CloseFit(label, n, g, b, se, bm, bc)


def close_fits(entries: Iterable[LedgerEntry], *, min_rows: int = MIN_N) -> list[CloseFit]:
    """One fit per family, then per market with at least ``min_rows`` closes."""
    graded = _graded(entries)
    out = [close_fit([e for e in graded if family(e.market) == f], f) for f in ("batter", "pitcher", "game")]
    markets = sorted({e.market for e in graded})
    for m in markets:
        fit = close_fit([e for e in graded if e.market == m], m)
        if fit.n >= min_rows:
            out.append(fit)
    return out


@dataclass(frozen=True)
class Graduation:
    market: str
    n: int
    roi: float
    roi_se: float
    first_half: float
    second_half: float
    clv_n: int
    clv: float | None
    clv_se: float
    status: str
    failed: tuple[str, ...]


def would_be_buy(e: LedgerEntry) -> bool:
    return e.tier in _BUY or e.pass_gate.startswith(GRADED_ONLY_PREFIX)


def _mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        return values[0], 0.0
    return fmean(values), stdev(values) / len(values) ** 0.5


def _beyond_one_se(mean: float, se: float) -> bool:
    return mean > se if se > 0 else mean > 0


def judge(market: str, buys: list[LedgerEntry], *, min_n: int = MIN_N) -> Graduation:
    """The four tests on one market's would-be buys, in slate order."""
    buys = sorted(buys, key=lambda e: e.date)
    roi, roi_se = _mean_se([e.pnl for e in buys])
    mid = len(buys) // 2
    h1 = fmean([e.pnl for e in buys[:mid]]) if mid else 0.0
    h2 = fmean([e.pnl for e in buys[mid:]]) if buys[mid:] else 0.0
    clvs = [e.clv_ev for e in buys if e.clv_quarantine == "clean" and e.clv_ev is not None]
    clv, clv_se = _mean_se(clvs)
    failed: list[str] = []
    if len(buys) < min_n:
        failed.append(f"volume ({len(buys)} of {min_n})")
    if not _beyond_one_se(roi, roi_se):
        failed.append("margin")
    if not (h1 > 0 and h2 > 0):
        failed.append("consistency")
    if len(clvs) < min_n or not _beyond_one_se(clv, clv_se):
        failed.append("clv")
    status = WATCHING if len(buys) < min_n else (NOT_YET if failed else MEETS_BAR)
    return Graduation(market, len(buys), roi, roi_se, h1, h2, len(clvs),
                      clv if clvs else None, clv_se, status, tuple(failed))


def graduation(entries: Iterable[LedgerEntry], *, min_n: int = MIN_N) -> list[Graduation]:
    """One verdict per market that has any would-be buy."""
    by_market: dict[str, list[LedgerEntry]] = {}
    for e in _graded(entries):
        if e.odds is not None and would_be_buy(e):
            by_market.setdefault(e.market, []).append(e)
    return [judge(m, rows, min_n=min_n) for m, rows in sorted(by_market.items())]


def close_test_lines(fits: list[CloseFit]) -> list[str]:
    lines = []
    for f in fits:
        lines.append(
            f"  {f.label:<16} n={f.n:<7} games={f.games:<5} b={f.slope:+.3f} (se {f.se:.3f})"
            f"  Brier model {f.brier_model:.4f} / close {f.brier_close:.4f}  -> {f.verdict}"
        )
    return lines


def graduation_lines(grads: list[Graduation], *, min_n: int = MIN_N) -> list[str]:
    lines = []
    for g in grads:
        clv = "n/a" if g.clv is None else f"{g.clv * 100:+.1f}% (se {g.clv_se * 100:.1f}) on {g.clv_n}"
        line = (
            f"  {g.status:<9} {g.market:<16} n={g.n:<5} ROI {g.roi * 100:+.1f}% (se {g.roi_se * 100:.1f})"
            f"  halves {g.first_half * 100:+.1f}% / {g.second_half * 100:+.1f}%  EV at close {clv}"
        )
        if g.failed:
            line += f"  not met: {', '.join(g.failed)}"
        lines.append(line)
    if not lines:
        lines.append(f"  no would-be buys graded yet (each market needs {min_n})")
    return lines
