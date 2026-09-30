"""Per-market calibration from the graded ledger (master plan §5.8).

Two layers, applied to the sim's probability before EV/tier:

1. **Isotonic map per market**, fit on graded rows (pushes dropped) once a
   market has ``min_samples`` of them. Until then the map is the identity and
   the card says ``calibration: none (n=<graded>)``. A map is stamped with the
   sim's ``FEATURE_BASIS``; a change that moves raw probabilities bumps the
   basis and retires old maps rather than re-imposing the bias they corrected.
2. **Confidence shrink** beyond the pivot: ``p' = pivot + slope x (p - pivot)``
   for ``p >= pivot``. The standing guard for an untrained tail; it never
   crosses .5 so the favoured side cannot flip, and it is what keeps a fresh
   sim from manufacturing Strong buys on 70% favourites before the ledger has
   said anything about them.

Both layers are logged as adjustment reasons on the row.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from engine_common.isotonic import IsotonicMap
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.models.markets import Prob

log = logging.getLogger("nhl_engine")

FEATURE_BASIS = "sim-periods-2026.09"
MIN_SAMPLES = 200


@dataclass(frozen=True)
class ConfidenceShrink:
    pivot: float = 0.62
    slope: float = 0.55

    def apply(self, p: float) -> float:
        return self.pivot + self.slope * (p - self.pivot) if p >= self.pivot else p


@dataclass
class Calibrator:
    maps: dict[str, IsotonicMap] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    shrink: ConfidenceShrink = field(default_factory=ConfidenceShrink)

    def apply(self, market: str, p: Prob) -> tuple[Prob, list[str]]:
        reasons: list[str] = []
        raw = p.win
        m = self.maps.get(market)
        cal = raw
        if m is not None and not m.is_identity:
            cal = m.apply(raw)
            reasons.append(
                f"isotonic[{market}] {raw:.3f}->{cal:.3f} (n={self.counts.get(market, 0)})"
            )
        else:
            reasons.append(f"calibration: none (n={self.counts.get(market, 0)})")
        shr = self.shrink.apply(cal)
        if shr != cal:
            reasons.append(f"confidence shrink {cal:.3f}->{shr:.3f}")
        return Prob(shr, p.push), reasons

    @classmethod
    def fit(cls, rows: list[LedgerRow], min_samples: int = MIN_SAMPLES) -> Calibrator:
        by_market: dict[str, list[tuple[float, int]]] = {}
        for r in rows:
            if r.outcome in ("win", "loss"):
                by_market.setdefault(r.market, []).append(
                    (r.model_prob, 1 if r.outcome == "win" else 0)
                )
        maps = {
            mk: IsotonicMap.fit(pairs)
            for mk, pairs in by_market.items()
            if len(pairs) >= min_samples
        }
        return cls(maps=maps, counts={mk: len(v) for mk, v in by_market.items()})

    def save(self, path: Path) -> None:
        payload = {
            "basis": FEATURE_BASIS,
            "counts": self.counts,
            "markets": {mk: {"x": list(m.x), "y": list(m.y)} for mk, m in self.maps.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if data.get("basis") != FEATURE_BASIS:
            log.warning(
                "calibration %s fit on basis %r, engine is %r: ignored",
                path.name,
                data.get("basis"),
                FEATURE_BASIS,
            )
            return cls()
        maps = {
            mk: IsotonicMap(tuple(v["x"]), tuple(v["y"]))
            for mk, v in data.get("markets", {}).items()
        }
        return cls(maps=maps, counts={k: int(v) for k, v in data.get("counts", {}).items()})


def calibration_path(data_dir: Path) -> Path:
    return data_dir / "calibration.json"


__all__ = ["FEATURE_BASIS", "Calibrator", "ConfidenceShrink", "calibration_path"]
