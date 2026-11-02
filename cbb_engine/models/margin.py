"""v0 margin distribution: the market's spread owns the location, the fit owns the shape.

``MarginWidth`` comes from ``params/margin.json`` written by ``scripts/cbb/fit_margin.py`` on
last season's closes: the SD of ``final margin − close spread`` per month of the season, the
key-number ratios of the final margin (fewer 1-point games, more 3-5; never 0, overtime), and
a moneyline width multiplier fitted on who won, and the first half as a fitted share of the full game (margin and total), with its own SD and key
ratios (a tie at the half is possible). Spreads, moneylines and the three first-half markets
are priced off one location: the home margin implied by the devigged consensus spread.
With no fitted file those rows stay market-only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from cbb_engine.market import consensus

PARAMS = Path(__file__).resolve().parents[1] / "params" / "margin.json"
SPREAD = "game_ats"
TOTAL = "game_total"
H1 = ("h1_ats", "h1_ml", "h1_total")
PRICED = ("game_ats", "game_ml", *H1)


def phase(day: str) -> str:
    """Month of the season from the slate date (``YYYY-MM-DD``); April folds into March."""
    m = int(day[5:7]) if len(day) >= 7 and day[5:7].isdigit() else 0
    return {11: "nov", 12: "dec", 1: "jan", 2: "feb", 3: "mar", 4: "mar"}.get(m, "all")


@dataclass(frozen=True)
class MarginWidth:
    sd_by_phase: dict[str, float]
    key_ratio: dict[int, float] = field(default_factory=dict)  # |final margin| -> mass multiplier
    ml_scale: float = 1.0  # moneyline width = ml_scale × cover width (fitted on who won)
    h1_margin_share: float = 0.5
    h1_margin_sd: float = 0.0
    h1_key_ratio: dict[int, float] = field(default_factory=dict)  # |half margin| -> multiplier (0 = tie)
    h1_total_a: float = 0.0
    h1_total_b: float = 0.5
    h1_total_sd: float = 0.0
    fitted_on: str = ""
    n: int = 0

    def sd(self, day: str) -> float | None:
        return self.sd_by_phase.get(phase(day)) or self.sd_by_phase.get("all")


def load(path: Path = PARAMS) -> MarginWidth | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return MarginWidth(
        sd_by_phase={str(k): float(v) for k, v in raw.get("sd_by_phase", {}).items()},
        key_ratio={int(k): float(v) for k, v in raw.get("key_ratio", {}).items()},
        ml_scale=float(raw.get("ml_scale", 1.0)),
        h1_margin_share=float(raw.get("h1_margin_share", 0.5)),
        h1_margin_sd=float(raw.get("h1_margin_sd", 0.0)),
        h1_key_ratio={int(k): float(v) for k, v in raw.get("h1_key_ratio", {}).items()},
        h1_total_a=float(raw.get("h1_total_a", 0.0)),
        h1_total_b=float(raw.get("h1_total_b", 0.5)),
        h1_total_sd=float(raw.get("h1_total_sd", 0.0)),
        fitted_on=str(raw.get("fitted_on", "")),
        n=int(raw.get("n", 0)),
    )


def _cdf(x: float, mu: float, sd: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))


def pmf(mu: float, sd: float, ratio: dict[int, float], *, tie: bool) -> dict[int, float]:
    """Home margin on integers: normal around ``mu``, key ratios on |k|; no 0 unless ``tie`` (a half)."""
    lo, hi = int(mu - 6 * sd) - 1, int(mu + 6 * sd) + 1
    out = {}
    for k in range(lo, hi + 1):
        if k == 0 and not tie:
            continue
        out[k] = (_cdf(k + 0.5, mu, sd) - _cdf(k - 0.5, mu, sd)) * ratio.get(abs(k), 1.0)
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def p_side(dist: dict[int, float], side: str, line: float | None) -> float | None:
    """P(side wins | no push) for a spread at ``line`` (side-relative) or a moneyline (``None``)."""
    sign = 1 if side == "home" else -1 if side == "away" else 0
    if not sign:
        return None
    adj = line or 0.0
    win = sum(v for k, v in dist.items() if sign * k + adj > 0)
    push = sum(v for k, v in dist.items() if sign * k + adj == 0)
    return win / (1 - push) if push < 1 else None


def p_total(mu: float, sd: float, side: str, line: float) -> float:
    """P(over/under | no push) on integer totals under N(mu, sd)."""
    if float(line).is_integer():
        over, under = 1 - _cdf(line + 0.5, mu, sd), _cdf(line - 0.5, mu, sd)
    else:
        over, under = 1 - _cdf(line, mu, sd), _cdf(line, mu, sd)
    return (over if side == "over" else under) / max(over + under, 1e-12)


def solve_mu(side: str, line: float, p: float, sd: float, ratio: dict[int, float]) -> float:
    """The home-margin location at which ``side`` at ``line`` wins with probability ``p``."""
    lo, hi = -80.0, 80.0
    for _ in range(50):
        mid = (lo + hi) / 2
        q = p_side(pmf(mid, sd, ratio, tie=False), side, line) or 0.0
        # the home side's chance rises with mu, the away side's falls
        if (q < p) == (side == "home"):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def spread_mu(anchors: Sequence[consensus.Anchor], event_id: str, sd: float, mw: MarginWidth) -> float | None:
    """Fair home margin for one game: the median over the paired main-spread anchors of each
    anchor's devigged cover probability inverted through the fitted shape."""
    mus = []
    for a in anchors:
        if a.event_id != event_id or a.market != SPREAD or a.line is None or not a.paired_books:
            continue
        side = a.sides[0]
        p = a.p_fair.get(side)
        if side in ("home", "away") and p is not None and 0.01 < p < 0.99:
            mus.append(solve_mu(side, a.line, p, sd, mw.key_ratio))
    return float(median(mus)) if mus else None


def total_mu(anchors: Sequence[consensus.Anchor], event_id: str, sd: float) -> float | None:
    """Fair full-game total: the median over paired total anchors of the line shifted by its Over lean."""
    mus = []
    for a in anchors:
        if a.event_id != event_id or a.market != TOTAL or a.line is None or not a.paired_books:
            continue
        p = a.p_fair.get("over")
        if p is not None and 0.01 < p < 0.99:
            mus.append(a.line + sd * math.sqrt(2) * _erfinv(2 * p - 1))
    return float(median(mus)) if mus else None


def price(
    market: str, side: str, line: float | None, *, home_mu: float | None, full_total: float | None,
    day: str, mw: MarginWidth,
) -> float | None:
    """p_model for a spread, moneyline or first-half row; ``None`` when its location is missing."""
    sd = mw.sd(day)
    if market == "game_ats":
        if home_mu is None or sd is None:
            return None
        return p_side(pmf(home_mu, sd, mw.key_ratio, tie=False), side, line)
    if market == "game_ml":
        if home_mu is None or sd is None:
            return None
        return p_side(pmf(home_mu, sd * mw.ml_scale, mw.key_ratio, tie=False), side, None)
    if market in ("h1_ats", "h1_ml"):
        if home_mu is None or not mw.h1_margin_sd:
            return None
        d = pmf(mw.h1_margin_share * home_mu, mw.h1_margin_sd, mw.h1_key_ratio, tie=True)
        return p_side(d, side, line if market == "h1_ats" else None)
    if market == "h1_total":
        if full_total is None or line is None or not mw.h1_total_sd or side not in ("over", "under"):
            return None
        return p_total(mw.h1_total_a + mw.h1_total_b * full_total, mw.h1_total_sd, side, line)
    return None


def _erfinv(y: float) -> float:
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


__all__ = [
    "H1", "PARAMS", "PRICED", "MarginWidth", "load", "p_side", "p_total", "phase", "pmf", "price",
    "solve_mu", "spread_mu", "total_mu",
]
