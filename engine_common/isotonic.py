"""Dependency-free isotonic (PAV) regression and a piecewise-linear monotone map."""

from __future__ import annotations

from dataclasses import dataclass


def pav(y: list[float], w: list[float]) -> list[float]:
    """Pool-adjacent-violators: weighted least-squares non-decreasing fit of ``y``."""
    blocks: list[list[float]] = [[y[i], w[i], 1.0] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0]:
            i += 1
            continue
        m0, w0, c0 = blocks[i]
        m1, w1, c1 = blocks[i + 1]
        tw = w0 + w1
        blocks[i : i + 2] = [[(m0 * w0 + m1 * w1) / tw if tw else 0.0, tw, c0 + c1]]
        if i > 0:
            i -= 1
    out: list[float] = []
    for mean, _tw, count in blocks:
        out.extend([mean] * int(count))
    return out


@dataclass(frozen=True)
class IsotonicMap:
    """Monotone map raw probability -> observed frequency, linear between knots."""

    x: tuple[float, ...] = ()
    y: tuple[float, ...] = ()

    @property
    def is_identity(self) -> bool:
        return not self.x

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
    def fit(cls, pairs: list[tuple[float, int]], n_bins: int = 40) -> IsotonicMap:
        """Bin ``(prob, won)`` pairs on ``prob``, then PAV the bin win rates."""
        if not pairs:
            return cls()
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
        return cls(tuple(xs), tuple(pav(raw_y, wts)))


__all__ = ["IsotonicMap", "pav"]
