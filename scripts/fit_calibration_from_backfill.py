"""Refit the isotonic calibration map from a backfilled graded history.

The packaged map is fit on 2024. This refits from the CSVs produced by
`scripts/backfill_graded_history.py`, validates out-of-sample on a temporal split
(fit the early slates, score the later ones, then swap), and only writes the
markets that actually improve out-of-sample Brier -- a market whose refit map is
worse than the packaged one keeps the packaged one.

Fits on `raw_prob`, not `model_prob`: `model_prob` already has a calibration map
applied, so fitting on it stacks the correction.

    python scripts/fit_calibration_from_backfill.py [--write]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from mlb_engine.calibration import Calibrator, ConfidenceShrink
from mlb_engine.config import load_config

PACKAGED = Path("mlb_engine/data/calibration_2024.json")
OUT = Path("mlb_engine/data/calibration_2026.json")

Rows = list[tuple[str, float, int]]  # (market, raw_prob, won)


def _brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - w) ** 2 for p, w in pairs) / len(pairs) if pairs else 0.0


def load_rows(audit_dir: Path) -> list[tuple[str, str, float, int]]:
    """(date, market, raw_prob, won) for every graded backfill row."""
    out: list[tuple[str, str, float, int]] = []
    for path in sorted(audit_dir.glob("backfill_shard*.csv")):
        with path.open() as fh:
            for r in csv.DictReader(fh):
                if r["result"] not in ("win", "loss"):
                    continue
                out.append((r["date"], r["market"], float(r["raw_prob"]),
                            1 if r["result"] == "win" else 0))
    return out


def main() -> int:
    cfg = load_config()
    rows = load_rows(cfg.audit_dir)
    if not rows:
        print(f"no backfill_shard*.csv under {cfg.audit_dir}")
        return 2
    dates = sorted({d for d, *_ in rows})
    split = dates[len(dates) // 2]
    print(f"{len(rows)} graded rows, {len(dates)} slates, temporal split at {split}\n")

    shrink = ConfidenceShrink()
    packaged = Calibrator.from_json(PACKAGED) if PACKAGED.exists() else Calibrator.identity()

    early = [(m, p, w) for d, m, p, w in rows if d < split]
    late = [(m, p, w) for d, m, p, w in rows if d >= split]
    wins: list[set[str]] = []
    print(f"{'market':<14}{'n test':>8}  {'packaged':>9} {'refit':>9}  {'delta':>8}  verdict")
    for train, test, tag in ((early, late, "early->late"), (late, early, "late->early")):
        refit = Calibrator.fit(train)
        by_market: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for m, p, w in test:
            by_market[m].append((p, w))
        won: set[str] = set()
        print(f"-- {tag}")
        for mk in sorted(by_market):
            pairs = by_market[mk]
            old = [(shrink.apply(packaged.apply(mk, p)), w) for p, w in pairs]
            new = [(shrink.apply(refit.apply(mk, p)), w) for p, w in pairs]
            bo, bn = _brier(old), _brier(new)
            better = bn < bo and mk in refit.maps
            if better:
                won.add(mk)
            print(f"{mk:<14}{len(pairs):>8}  {bo:>9.5f} {bn:>9.5f}  "
                  f"{(bo - bn) / bo * 100:>+7.2f}%  {'refit' if better else 'packaged'}")
        wins.append(won)

    # a market has to win in *both* directions -- winning one half is a coin flip
    keep = wins[0] & wins[1]
    print(f"\nrefit wins in both directions: {sorted(keep)}")
    print(f"one direction only (left on the packaged map): "
          f"{sorted((wins[0] | wins[1]) - keep)}")
    if "--write" not in sys.argv:
        print("\n(dry run; pass --write to emit the production map)")
        return 0

    full = Calibrator.fit([(m, p, w) for _d, m, p, w in rows])
    merged = Calibrator(
        maps={**packaged.maps, **{mk: m for mk, m in full.maps.items() if mk in keep}},
        default=full.default if full.default.x else packaged.default,
    )
    merged.to_json(OUT)
    print(f"\nwrote {OUT}: {len(merged.maps)} market maps "
          f"({len(keep)} refit from {len(rows)} 2026 rows, rest packaged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
