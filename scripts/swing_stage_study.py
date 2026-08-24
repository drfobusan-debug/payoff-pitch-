"""Does the swing underneath a luck gap predict the fortnight after it?

The power screen's regression test is stage one: trailing wOBA against xwOBA, cut
at fifty points. This asks what a second stage on bat tracking adds to it, and it
asks it out of time -- every predictor is read strictly before an anchor date and
the target is the fortnight that follows, which no predictor can see.

Three things are separated deliberately, because conflating them is how the
engine has bought noise before:

* **levels** -- each metric read over its own window of competitive swings, the
  windows ``features.swing`` measured and stores;
* **trends** -- the same window against the immediately preceding, non-overlapping
  one, which is the construction PR #109's barrel trend failed on;
* **the rescue** -- inside the rows stage one cuts, whether the better swing half
  goes on to out-produce the worse, and the hitters the cut keeps.

Standard errors are clustered on the hitter, who appears at up to thirteen
anchors, so the naive t-statistics would be roughly a third too large.

    python scripts/swing_stage_study.py --seasons 2025 2026 --scale 4

``--scale`` multiplies every window, since a three-swing read of bat speed is
reliable in the split-half sense and is still one at-bat; production uses 4.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.data.statcast import StatcastRepository
from mlb_engine.features.swing import (
    BLAST_BAT_SPEED,
    FAST_SWING_MPH,
    MIN_TRACKED_MPH,
    SQUARED_UP_RATIO,
    SWING_DESC,
    SWINGS_FOR_READABLE,
)
from mlb_engine.output.power_screen import MAX_LUCK_GAP

log = logging.getLogger("swing-stage")

METRICS = ("bat_speed", "fast", "squared_up", "blast", "swing_length")
TB_MAP = {"single": 1.0, "double": 2.0, "triple": 3.0, "home_run": 4.0}
TARGETS = (("nxt_woba", "wOBA/PA"), ("nxt_tb", "TB/PA"), ("nxt_hit", "H/PA"), ("nxt_hr", "HR/PA"))
HORIZON = 14  # days of the out-of-time target
FORM_DAYS = 42  # the screen's own trailing window, for stage one
MIN_OUT_PA = 25  # a target read on fewer plate appearances is mostly noise
MIN_FORM_PA = 40


def ols(
    y: np.ndarray, x: np.ndarray, cluster: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least squares with standard errors clustered on ``cluster``."""
    xtx_inv = np.linalg.pinv(x.T @ x)
    b = xtx_inv @ x.T @ y
    e = y - x @ b
    meat = np.zeros((x.shape[1], x.shape[1]))
    for g in np.unique(cluster):
        m = cluster == g
        u = x[m].T @ e[m]
        meat += np.outer(u, u)
    groups = len(np.unique(cluster))
    adj = groups / (groups - 1) * (len(y) - 1) / (len(y) - x.shape[1])
    v = xtx_inv @ (meat * adj) @ xtx_inv
    r2 = 1.0 - float(e @ e) / float(((y - y.mean()) ** 2).sum())
    return b, np.sqrt(np.diag(v)), r2


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def slices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The swing feed and the plate-appearance feed the panel is built from."""
    df = df.assign(game_date=pd.to_datetime(df["game_date"]).dt.date)
    sw = df[
        df["description"].isin(SWING_DESC)
        & df["bat_speed"].notna()
        & (df["bat_speed"].astype(float) >= MIN_TRACKED_MPH)
    ].copy()
    bat = sw["bat_speed"].astype(float)
    ratio = sw["launch_speed"].astype(float) / (
        1.23 * bat + 0.23 * sw["release_speed"].astype(float) * 0.915
    )
    sq = ratio.fillna(0.0) >= SQUARED_UP_RATIO
    sw["squared_up"] = sq.astype(float)
    sw["blast"] = (sq & (bat >= BLAST_BAT_SPEED)).astype(float)
    sw["fast"] = (bat >= FAST_SWING_MPH).astype(float)
    sw = sw.sort_values(["batter", "game_date"])[["game_date", "batter", *METRICS]]

    pa = df[df["woba_denom"].notna()].copy()
    pa["tb"] = pa["events"].map(TB_MAP).fillna(0.0)
    pa["hit"] = (pa["tb"] > 0).astype(float)
    pa["hr"] = (pa["events"] == "home_run").astype(float)
    pa["woba"] = pa["woba_value"].fillna(0.0)
    pa["xwoba"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba"])
    pa = pa.sort_values(["batter", "game_date"])[
        ["game_date", "batter", "woba", "xwoba", "tb", "hit", "hr"]
    ]
    return sw, pa


def panel(repo: StatcastRepository, seasons: list[int], scale: int) -> pd.DataFrame:
    """One row per hitter per anchor: swing levels and trends, then the target."""
    window = {m: max(1, int(np.ceil(SWINGS_FOR_READABLE[m])) * scale) for m in METRICS}
    rows: list[dict[str, object]] = []
    for season in seasons:
        start = Date(season, 3, 1)
        end = min(Date(season, 11, 15), Date.today())
        sw, pa = slices(repo.load_range(start, end))
        sw_by = dict(tuple(sw.groupby("batter", sort=False)))
        pa_by = dict(tuple(pa.groupby("batter", sort=False)))
        anchor = start + timedelta(90)  # deep enough in that the windows can fill
        anchors = []
        while anchor + timedelta(HORIZON) <= end:
            anchors.append(anchor)
            anchor += timedelta(HORIZON)
        for day in anchors:
            kept = 0
            for pid, s in sw_by.items():
                p = pa_by.get(pid)
                if p is None:
                    continue
                past = s[s["game_date"] < day]
                fut = p[(p["game_date"] >= day) & (p["game_date"] <= day + timedelta(HORIZON - 1))]
                form = p[(p["game_date"] >= day - timedelta(FORM_DAYS)) & (p["game_date"] < day)]
                if len(fut) < MIN_OUT_PA or len(form) < MIN_FORM_PA:
                    continue
                row: dict[str, object] = {
                    "season": season,
                    "anchor": day.isoformat(),
                    "batter": int(pid),
                }
                if any(len(past) < 2 * window[m] for m in METRICS):
                    continue
                for metric in METRICS:
                    n = window[metric]
                    v = past[metric].to_numpy(dtype=float)
                    row[f"lvl_{metric}"] = float(v[-n:].mean())
                    row[f"d_{metric}"] = float(v[-n:].mean() - v[-2 * n : -n].mean())
                now_woba = float(form["woba"].mean())
                now_xwoba = float(form["xwoba"].mean())
                row["now_woba"] = now_woba
                row["now_xwoba"] = now_xwoba
                row["luck_gap"] = now_woba - now_xwoba
                for k in ("woba", "tb", "hit", "hr"):
                    row[f"nxt_{k}"] = float(fut[k].mean())
                rows.append(row)
                kept += 1
            log.info("%s anchor %s  n=%d", season, day, kept)
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def report(p: pd.DataFrame, scale: int) -> None:
    for m in METRICS:
        p[f"zl_{m}"] = _z(p[f"lvl_{m}"])
        p[f"zd_{m}"] = _z(p[f"d_{m}"])
    p["swing_l"] = p[[f"zl_{m}" for m in METRICS]].mean(axis=1)
    p["swing_d"] = p[[f"zd_{m}" for m in METRICS]].mean(axis=1)
    p["flagged"] = p["luck_gap"] > MAX_LUCK_GAP
    cl = p["batter"].to_numpy()
    one = np.ones(len(p))
    base = np.column_stack([one, p["now_woba"], p["now_xwoba"]])

    print(
        f"panel x{scale}: {len(p)} batter-windows, {p['batter'].nunique()} hitters, "
        f"{p['anchor'].nunique()} anchors; stage-one flag fires on {p['flagged'].mean():.1%}"
    )

    print("\nstage one alone -- the next fortnight on trailing wOBA and xwOBA")
    for col, label in TARGETS:
        y = p[col].to_numpy(float)
        b, se, r2 = ols(y, base, cl)
        print(
            f"  {label:8s} R2 {r2:.4f} | wOBA {b[1]:+.3f} (t {b[1] / se[1]:+.2f})"
            f"  xwOBA {b[2]:+.3f} (t {b[2] / se[2]:+.2f})"
        )

    for kind, prefix in (("level", "zl_"), ("trend", "zd_")):
        print(f"\nstage two by {kind}, each added on top of stage one")
        for col, label in TARGETS:
            y = p[col].to_numpy(float)
            _, _, r2b = ols(y, base, cl)
            print(f"  target {label} (stage-one R2 {r2b:.4f})")
            for m in METRICS:
                x = np.column_stack([base, p[f"{prefix}{m}"]])
                b, se, r2 = ols(y, x, cl)
                print(
                    f"    {m:14s} coef {b[3]:+.5f}  t {b[3] / se[3]:+.2f}  dR2 {r2 - r2b:+.5f}"
                )

    print("\ndoes the swing rescue a hitter stage one wants cut? (flagged rows only)")
    q = p[p["flagged"]]
    for col, label in TARGETS[:2]:
        y = q[col].to_numpy(float)
        x = np.column_stack([np.ones(len(q)), q["now_woba"], q["now_xwoba"], q["swing_l"]])
        b, se, _ = ols(y, x, q["batter"].to_numpy())
        top = q["swing_l"] > q["swing_l"].median()
        print(
            f"  {label:8s} swing level coef {b[3]:+.5f} (t {b[3] / se[3]:+.2f}, n {len(q)}) | "
            f"better half {q.loc[top, col].mean():.4f} vs worse {q.loc[~top, col].mean():.4f} | "
            f"the rows the cut keeps {p.loc[~p['flagged'], col].mean():.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--scale", type=int, default=4, help="multiple of each r=.50 window")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cfg = load_config()
    p = panel(StatcastRepository(cfg.cache_dir), args.seasons, args.scale)
    if p.empty:
        raise SystemExit("no batter-windows: the cache does not cover these seasons")
    report(p, args.scale)


if __name__ == "__main__":
    main()
