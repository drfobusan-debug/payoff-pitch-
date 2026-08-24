"""How many competitive swings each bat-tracking measure needs to repeat.

Companion to ``scripts/measure_metric_reliability.py``, which asks the same
question of the eleven screened metrics in plate appearances. Bat tracking sits
on every competitive swing rather than only on batted balls, so its natural unit
is the swing and its windows are far shorter -- which is the whole reason the
screen can read a hitter's swing on a fortnight and cannot read his wOBA on one.

Method is block-to-block correlation: a hitter's swings are ordered, split into
adjacent, equal, non-overlapping blocks of ``n``, and consecutive blocks are
correlated pooled across the league. Adjacent blocks rather than a shuffled
split-half because a swing measure genuinely drifts within a season, and the
adjacent form charges that drift against the metric instead of hiding it.

Squared-up and blast rate are not published per swing. They are reconstructed
from the collision model -- exit speed as a share of ``1.23 * bat speed + 0.23 *
plate speed`` -- with the two cuts *calibrated* so the league rate reproduces
Savant's own leaderboard figure for the same dates. The uncalibrated model puts
the league at 45% squared up against a true 25%, so what the collision model gets
wrong is the threshold rather than the ordering.

    python scripts/measure_swing_reliability.py --seasons 2025 2026

Prints the curve each metric traces and the ``n`` where it crosses r=.50; those
are the numbers ``mlb_engine/features/swing.py`` stores in ``CURVES``. Re-run it
when another season of tracking has accumulated.
"""

from __future__ import annotations

import argparse
import io
import logging
from datetime import date as Date

import numpy as np
import pandas as pd
import requests

from mlb_engine.config import load_config
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.features.swing import (
    FAST_SWING_MPH,
    MIN_TRACKED_MPH,
    READABLE_R,
    SWING_DESC,
)

log = logging.getLogger("swing-reliability")

GRID = (3, 4, 5, 6, 8, 10, 25, 50, 100, 150, 250)
MIN_SWINGS = 40  # a hitter below this contributes no block pair anywhere
MIN_PAIRS = 30  # a correlation on fewer block pairs than this is not reported

#: The window Savant's leaderboard is asked for, to calibrate the two cuts.
CAL_START, CAL_END = Date(2026, 7, 12), Date(2026, 8, 21)
LEADERBOARD = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"

METRICS = ("bat_speed", "fast", "squared_up", "blast", "swing_length", "attack_angle")


def league_rates(start: Date, end: Date) -> tuple[float, float]:
    """Savant's own squared-up and blast rate per swing over ``start..end``.

    The leaderboard takes a date range but cannot be sliced by swing count, which
    is why it calibrates the reconstruction rather than replacing it.
    """
    resp = requests.get(
        LEADERBOARD,
        params={
            "dateStart": start.isoformat(),
            "dateEnd": end.isoformat(),
            "seasonStart": str(start.year),
            "seasonEnd": str(end.year),
            "type": "batter",
            "minSwings": "1",
            "minGroupSwings": "1",
            "csv": "true",
        },
        timeout=120,
    )
    resp.raise_for_status()
    d = pd.read_csv(io.StringIO(resp.text))
    w = d["swings_competitive"].astype(float)
    sq = float((d["squared_up_per_swing"].astype(float) * w).sum() / w.sum())
    bl = float((d["blast_per_swing"].astype(float) * w).sum() / w.sum())
    return sq, bl


def tracked_swings(df: pd.DataFrame) -> pd.DataFrame:
    """Competitive tracked swings, oldest first, with the collision ratio."""
    sw = df[
        df["description"].isin(SWING_DESC)
        & df["bat_speed"].notna()
        & (df["bat_speed"].astype(float) >= MIN_TRACKED_MPH)
    ].copy()
    sw["game_date"] = pd.to_datetime(sw["game_date"])
    bat = sw["bat_speed"].astype(float)
    ev = sw["launch_speed"].astype(float)
    pitch = sw["release_speed"].astype(float)
    sw["ratio"] = (ev / (1.23 * bat + 0.23 * pitch * 0.915)).fillna(0.0)
    sw["fast"] = (bat >= FAST_SWING_MPH).astype(float)
    return sw.sort_values("game_date").reset_index(drop=True)


def calibrate(sw: pd.DataFrame, sq_rate: float, bl_rate: float) -> tuple[float, float]:
    """The ratio and bat-speed cuts that reproduce the published league rates."""
    window = sw[
        (sw["game_date"] >= pd.Timestamp(CAL_START)) & (sw["game_date"] <= pd.Timestamp(CAL_END))
    ]
    if window.empty:
        raise SystemExit("no cached swings inside the calibration window")
    sq_cut = float(np.quantile(window["ratio"].to_numpy(float), 1 - sq_rate))
    is_sq = window["ratio"].to_numpy(float) >= sq_cut
    # A blast is the fast subset of squared-up swings, so the bat-speed cut is
    # the quantile *among* squared-up swings that lands on the published rate.
    share = bl_rate / max(is_sq.mean(), 1e-9)
    bs_cut = float(np.quantile(window["bat_speed"].astype(float).to_numpy()[is_sq], 1 - share))
    return sq_cut, bs_cut


def curves(sw: pd.DataFrame, sq_cut: float, bs_cut: float) -> dict[str, list[float]]:
    """Block-to-block correlation of each metric over the swing grid."""
    sq = sw["ratio"].to_numpy(float) >= sq_cut
    sw = sw.assign(
        squared_up=sq.astype(float),
        blast=(sq & (sw["bat_speed"].astype(float).to_numpy() >= bs_cut)).astype(float),
    )
    groups = [g for _, g in sw.groupby("batter", sort=False) if len(g) >= MIN_SWINGS]
    log.info("%d hitters, %d tracked swings", len(groups), len(sw))
    present = [m for m in METRICS if m in sw.columns]
    out: dict[str, list[float]] = {m: [] for m in present}
    for n in GRID:
        for metric in present:
            a: list[float] = []
            b: list[float] = []
            for g in groups:
                # Swing path is absent from frames cached before it was ingested,
                # so the blocks count readings rather than rows.
                v = g[metric].dropna().to_numpy(dtype=float)
                k = len(v) // n
                if k < 2:
                    continue
                blocks = [float(v[i * n : (i + 1) * n].mean()) for i in range(k)]
                a.extend(blocks[:-1])
                b.extend(blocks[1:])
            r = float(np.corrcoef(a, b)[0, 1]) if len(a) >= MIN_PAIRS else float("nan")
            out[metric].append(r)
    return out


def crossing(ns: tuple[int, ...], rs: list[float], target: float) -> float:
    """Interpolated swing count where the curve first reaches ``target``."""
    for (n0, r0), (n1, r1) in zip(zip(ns, rs, strict=False), zip(ns[1:], rs[1:], strict=False),
                                  strict=False):
        if r0 != r0 or r1 != r1:
            continue
        if r0 < target <= r1:
            return n0 + (target - r0) / (r1 - r0) * (n1 - n0)
    return float(ns[0]) if rs and rs[0] >= target else float("inf")


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
        frames.append(tracked_swings(repo.load_range(start, end)))
    sw = pd.concat(frames, ignore_index=True).sort_values("game_date").reset_index(drop=True)

    sq_rate, bl_rate = league_rates(CAL_START, CAL_END)
    sq_cut, bs_cut = calibrate(sw, sq_rate, bl_rate)
    print(
        f"leaderboard {CAL_START}..{CAL_END}: squared-up/swing {sq_rate:.3f}  "
        f"blast/swing {bl_rate:.3f}"
    )
    print(
        f"calibrated cuts: squared up when EV/maxEV >= {sq_cut:.3f}; a blast also needs "
        f"bat speed >= {bs_cut:.1f} mph\n"
    )

    table = curves(sw, sq_cut, bs_cut)
    metrics = tuple(table)
    print(f"{'n swings':>9s}" + "".join(f"{m:>14s}" for m in metrics))
    for i, n in enumerate(GRID):
        cells = "".join(
            f"{table[m][i]:>14.3f}" if table[m][i] == table[m][i] else f"{'-':>14s}"
            for m in metrics
        )
        print(f"{n:>9d}{cells}")
    print(f"\nr={READABLE_R:.2f} crossing, in competitive swings")
    for m in metrics:
        print(f"  {m:14s} {crossing(GRID, table[m], READABLE_R):6.1f}")


if __name__ == "__main__":
    main()
