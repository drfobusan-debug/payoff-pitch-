"""v0 game-total distribution (§3.6): consensus-centred, residual width, fitted key masses.

``Width`` parameters come from ``params/width.json`` written by
``scripts/cbb/fit_width.py`` on ``final − close`` residuals per pre-close bucket.
They are never hand-set: with no fitted file the model channel refuses every row
(``width_unfit``) and the market-only channel carries the card.

The pmf is discrete on integer totals. Location is solved so the pmf median hits
the target within ``anchor_tolerance_pts``; a miss is the ``anchor_miss`` refusal.
The pmf is closed-form, so ``paths == 0`` and ``se == 0.0`` on every row; ``se_cap``
exists for the Monte Carlo rungs above v0 and is wired, not exercised.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

PARAMS = Path(__file__).resolve().parents[1] / "params" / "width.json"


@dataclass(frozen=True)
class Width:
    sd_by_bucket: dict[str, float]
    key_mass: dict[int, float] = field(default_factory=dict)  # extra mass at a total, on top of the normal
    parity_shift: float = 0.0  # P(odd) − 0.5
    fitted_on: str = ""
    n: int = 0

    def sd(self, bucket: str) -> float | None:
        return self.sd_by_bucket.get(bucket) or self.sd_by_bucket.get("all")


def load_width(path: Path = PARAMS) -> Width | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return Width(
        sd_by_bucket={str(k): float(v) for k, v in raw.get("sd_by_bucket", {}).items()},
        key_mass={int(k): float(v) for k, v in raw.get("key_mass", {}).items()},
        parity_shift=float(raw.get("parity_shift", 0.0)),
        fitted_on=str(raw.get("fitted_on", "")),
        n=int(raw.get("n", 0)),
    )


def quality_bucket(paired_books: int, spread_abs: float | None) -> str:
    """Pre-close inputs only (§4.a): how many books priced it and how lopsided it is."""
    depth = "deep" if paired_books >= 6 else "mid" if paired_books >= 3 else "thin"
    size = "close" if spread_abs is None or spread_abs < 6 else "wide" if spread_abs < 15 else "blowout"
    return f"{depth}_{size}"


@dataclass(frozen=True)
class TotalDist:
    pmf: dict[int, float]
    target_median: float
    paths: int = 0
    se: float = 0.0

    @property
    def median(self) -> float:
        """CDF crossing of 0.5, interpolated inside the integer bin [k − ½, k + ½]."""
        acc = 0.0
        for k in sorted(self.pmf):
            m = self.pmf[k]
            if acc + m >= 0.5:
                return (k - 0.5) + (0.5 - acc) / m if m else float(k)
            acc += m
        return float(max(self.pmf))

    def p_over(self, line: float) -> float:
        return sum(v for k, v in self.pmf.items() if k > line)

    def p_under(self, line: float) -> float:
        return sum(v for k, v in self.pmf.items() if k < line)

    def push_mass(self, line: float) -> float:
        return self.pmf.get(int(line), 0.0) if float(line).is_integer() else 0.0

    def parity_mass(self) -> float:
        return sum(v for k, v in self.pmf.items() if k % 2 == 1)


def _normal_pmf(mu: float, sd: float, lo: int, hi: int) -> dict[int, float]:
    def cdf(x: float) -> float:
        return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

    return {k: cdf(k + 0.5) - cdf(k - 0.5) for k in range(lo, hi + 1)}


def build_total(target_median: float, sd: float, width: Width) -> TotalDist:
    lo, hi = int(target_median - 6 * sd), int(target_median + 6 * sd)
    base = _normal_pmf(target_median, sd, lo, hi)
    for k, extra in width.key_mass.items():
        if k in base:
            base[k] += extra
    if width.parity_shift:
        for k in base:
            base[k] *= 1 + (width.parity_shift * 2 if k % 2 else -width.parity_shift * 2)
    z = sum(base.values())
    return TotalDist({k: v / z for k, v in base.items()}, target_median)


def solve_total(target_median: float, sd: float, width: Width, *, tol: float, iters: int = 30) -> TotalDist:
    """Shift the location until the discrete median lands on the target (two-point bracket)."""
    lo, hi = target_median - 3.0, target_median + 3.0
    best = build_total(target_median, sd, width)
    for _ in range(iters):
        mid = (lo + hi) / 2
        d = build_total(mid, sd, width)
        if abs(d.median - target_median) < abs(best.median - target_median):
            best = d
        if abs(d.median - target_median) <= tol:
            return TotalDist(d.pmf, target_median, d.paths, d.se)
        if d.median < target_median:
            lo = mid
        else:
            hi = mid
    return TotalDist(best.pmf, target_median, best.paths, best.se)


__all__ = ["PARAMS", "TotalDist", "Width", "build_total", "load_width", "quality_bucket", "solve_total"]
