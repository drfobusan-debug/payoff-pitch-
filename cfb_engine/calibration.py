"""Probability calibration learned from the historical graded ledger.

Raw model probabilities are typically over-confident (a "65%" wins less than
65%). Feeding those inflated probabilities into the EV/tier logic manufactures
phantom edges -> false positives. This module maps each raw model probability
onto the win rate that probability *actually* achieved historically, using
isotonic regression (monotone, non-parametric) fit per market.

Applying the map before EV/tier pulls over-confident probabilities toward their
true rate, preserves ranking (isotonic is monotone, so it never flips the
favored side), and is fully auditable (the fitted breakpoints are stored JSON).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _min_samples() -> int:
    raw = os.getenv("CFBE_CALIB_MIN_SAMPLES")
    return int(raw) if raw not in (None, "") else 300


_N_BINS = 40


def _pav(y: list[float], w: list[float]) -> list[float]:
    """Pool-adjacent-violators: least-squares monotone (non-decreasing) fit."""
    vals = list(y)
    wts = list(w)
    blocks: list[list[float]] = [[vals[i], wts[i], 1.0] for i in range(len(vals))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0]:
            i += 1
            continue
        m0, w0, c0 = blocks[i]
        m1, w1, c1 = blocks[i + 1]
        tw = w0 + w1
        merged = [(m0 * w0 + m1 * w1) / tw if tw else 0.0, tw, c0 + c1]
        blocks[i : i + 2] = [merged]
        if i > 0:
            i -= 1
    out: list[float] = []
    for mean, _tw, count in blocks:
        out.extend([mean] * int(count))
    return out


@dataclass
class IsotonicMap:
    """Piecewise-linear monotone map raw prob -> calibrated prob."""

    x: list[float]
    y: list[float]

    def apply(self, p: float) -> float:
        if not self.x:
            return p
        if p <= self.x[0]:
            cal = self.y[0]
        elif p >= self.x[-1]:
            cal = self.y[-1]
        else:
            cal = self.y[-1]
            for i in range(1, len(self.x)):
                if p <= self.x[i]:
                    x0, x1 = self.x[i - 1], self.x[i]
                    y0, y1 = self.y[i - 1], self.y[i]
                    frac = (p - x0) / (x1 - x0) if x1 > x0 else 0.0
                    cal = y0 + frac * (y1 - y0)
                    break
        return min(max(cal, 1e-6), 1 - 1e-6)

    @classmethod
    def fit(cls, pairs: list[tuple[float, int]], n_bins: int = _N_BINS) -> IsotonicMap:
        if not pairs:
            return cls([], [])
        buckets: dict[int, list[tuple[float, int]]] = {}
        for prob, won in pairs:
            idx = min(int(prob * n_bins), n_bins - 1)
            buckets.setdefault(idx, []).append((prob, won))
        xs: list[float] = []
        raw_y: list[float] = []
        wts: list[float] = []
        for idx in sorted(buckets):
            items = buckets[idx]
            xs.append(sum(p for p, _ in items) / len(items))
            raw_y.append(sum(w for _, w in items) / len(items))
            wts.append(float(len(items)))
        cal_y = _pav(raw_y, wts)
        return cls(xs, cal_y)


@dataclass
class Calibrator:
    """Per-market isotonic maps with a pooled fallback."""

    maps: dict[str, IsotonicMap]
    default: IsotonicMap

    def apply(self, market: str, prob: float) -> float:
        m = self.maps.get(market, self.default)
        return m.apply(prob)

    @classmethod
    def fit(cls, graded: list[tuple[str, float, int]]) -> Calibrator:
        """Fit from ``(market, raw_prob, won)`` rows (pushes already dropped)."""
        by_market: dict[str, list[tuple[float, int]]] = {}
        allp: list[tuple[float, int]] = []
        for market, prob, won in graded:
            by_market.setdefault(market, []).append((prob, won))
            allp.append((prob, won))
        maps = {
            mk: IsotonicMap.fit(pairs)
            for mk, pairs in by_market.items()
            if len(pairs) >= _min_samples()
        }
        return cls(maps=maps, default=IsotonicMap.fit(allp))

    def to_json(self, path: Path) -> None:
        payload = {
            "markets": {mk: {"x": m.x, "y": m.y} for mk, m in self.maps.items()},
            "default": {"x": self.default.x, "y": self.default.y},
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> Calibrator:
        data = json.loads(path.read_text())
        maps = {mk: IsotonicMap(v["x"], v["y"]) for mk, v in data.get("markets", {}).items()}
        d = data.get("default", {"x": [], "y": []})
        return cls(maps=maps, default=IsotonicMap(d["x"], d["y"]))

    @classmethod
    def identity(cls) -> Calibrator:
        return cls(maps={}, default=IsotonicMap([], []))


@dataclass(frozen=True)
class ConfidenceShrink:
    """Pull the confident tail toward the pivot.

    Isotonic maps only correct a tail once it has enough graded history; this is
    the standing guard that keeps an untrained tail from manufacturing Strong
    Buys in the meantime. Everything beyond the pivot is compressed by ``slope``
    (``p' = pivot + slope * (p - pivot)``), which keeps the map continuous and
    monotone and never crosses .5, so it cannot flip the favored side. Only the
    upper tail moves.
    """

    pivot: float = 0.62
    slope: float = 0.55

    def apply(self, p: float) -> float:
        if p >= self.pivot:
            return self.pivot + self.slope * (p - self.pivot)
        return p
