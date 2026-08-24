"""How many fastballs each of a starter's physical measures needs to repeat.

Companion to ``scripts/measure_swing_reliability.py``, which asks the same
question of bat tracking in competitive swings. Method is identical: a pitcher's
fastballs are ordered, split into adjacent, equal, non-overlapping blocks of
``n``, and consecutive blocks are correlated pooled across the league. Adjacent
blocks rather than a shuffled split-half because velocity genuinely drifts across
a season, and the adjacent form charges that drift against the metric.

    python scripts/measure_arm_reliability.py --seasons 2025 2026

The answer is that every one of them half-repeats on a single pitch: a release
point is measured, not estimated from outcomes, so between-arm spread swamps
pitch-to-pitch scatter. That is why ``mlb_engine/features/arm.py`` sizes its
window from the out-of-time panel instead of from these curves, and stores them
only as the floor below which a level is not read. Blocks are counted per
pitcher-season, so an offseason never sits inside a block pair.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.features.arm import CURVES, READABLE_R, fastballs_of

log = logging.getLogger("arm-reliability")

GRID = (1, 2, 3, 5, 10, 25, 50, 100, 200, 400)
MIN_PITCHES = 40  # an arm below this contributes no block pair anywhere
MIN_PAIRS = 30  # a correlation on fewer block pairs than this is not reported

METRICS = tuple(CURVES)


def measures(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fastball, carrying every physical the arm model reads."""
    fb = fastballs_of(df)
    hand = fb["p_throws"].map({"L": -1.0, "R": 1.0}).fillna(1.0)
    velo = pd.to_numeric(fb["release_speed"], errors="coerce")
    ext = pd.to_numeric(fb["release_extension"], errors="coerce")
    dates = pd.to_datetime(fb["game_date"])
    out = pd.DataFrame(
        {
            "pitcher": fb["pitcher"],
            "season": dates.dt.year,
            "game_date": dates,
            "velo": velo,
            "pvelo": velo + 1.1 * ext - 6.0,
            "ext": ext,
            "rel_x": pd.to_numeric(fb["release_pos_x"], errors="coerce") * -hand,
            "rel_z": pd.to_numeric(fb["release_pos_z"], errors="coerce"),
            "spin": pd.to_numeric(fb["release_spin_rate"], errors="coerce"),
            "ivb": pd.to_numeric(fb["pfx_z"], errors="coerce") * 12.0,
            # Absent from frames cached before it was ingested, so the blocks
            # count readings rather than rows.
            "hb": (
                pd.to_numeric(fb["pfx_x"], errors="coerce") * 12.0 * -hand
                if "pfx_x" in fb
                else np.nan
            ),
        }
    )
    return out.sort_values(["pitcher", "season", "game_date"]).reset_index(drop=True)


def curves(d: pd.DataFrame) -> dict[str, list[float]]:
    """Block-to-block correlation of each metric over the pitch grid."""
    groups = [g for _, g in d.groupby(["pitcher", "season"], sort=False) if len(g) >= MIN_PITCHES]
    log.info("%d pitcher-seasons, %d fastballs", len(groups), len(d))
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for n in GRID:
        for metric in METRICS:
            a: list[float] = []
            b: list[float] = []
            for g in groups:
                v = g[metric].dropna().to_numpy(dtype=float)
                k = len(v) // n
                if k < 2:
                    continue
                blocks = [float(v[i * n : (i + 1) * n].mean()) for i in range(k)]
                a.extend(blocks[:-1])
                b.extend(blocks[1:])
            out[metric].append(
                float(np.corrcoef(a, b)[0, 1]) if len(a) >= MIN_PAIRS else float("nan")
            )
    return out


def crossing(ns: tuple[int, ...], rs: list[float], target: float) -> float:
    """Interpolated pitch count where the curve first reaches ``target``."""
    if rs and rs[0] == rs[0] and rs[0] >= target:
        return float(ns[0])
    for (n0, r0), (n1, r1) in zip(zip(ns, rs, strict=False), zip(ns[1:], rs[1:], strict=False),
                                  strict=False):
        if r0 != r0 or r1 != r1:
            continue
        if r0 < target <= r1:
            return n0 + (target - r0) / (r1 - r0) * (n1 - n0)
    return float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cfg = load_config()
    repo = StatcastRepository(cfg.cache_dir)
    frames = []
    for season in args.seasons:
        start = Date(season, 3, 1)
        end = min(Date(season, 11, 15), Date.today())
        frames.append(measures(repo.load_range(start, end)))
    d = pd.concat(frames, ignore_index=True)

    table = curves(d)
    print(f"{'n pitches':>9s}" + "".join(f"{m:>10s}" for m in METRICS))
    for i, n in enumerate(GRID):
        cells = "".join(
            f"{table[m][i]:>10.3f}" if table[m][i] == table[m][i] else f"{'-':>10s}"
            for m in METRICS
        )
        print(f"{n:>9d}{cells}")
    print(f"\nr={READABLE_R:.2f} crossing, in fastballs")
    for m in METRICS:
        print(f"  {m:10s} {crossing(GRID, table[m], READABLE_R):6.1f}")

    print("\nwithin-arm pitch-to-pitch scatter against between-arm spread")
    big = d.groupby(["pitcher", "season"]).filter(lambda g: len(g) >= 200)
    for m in METRICS:
        g = big.groupby(["pitcher", "season"])[m]
        mu, sd = g.mean().dropna(), g.std().dropna()
        if mu.empty:
            print(f"  {m:10s} unmeasured in this cache")
            continue
        print(
            f"  {m:10s} league {mu.mean():9.3f}  between-arm sd {mu.std():8.3f}  "
            f"within-arm sd {sd.mean():8.3f}  n {len(mu)}"
        )


if __name__ == "__main__":
    main()
