"""Fit the pitch-shape whiff models that ``features.stuff`` reads.

One logistic model per pitch type, on swings only: given the physics of the pitch
(velocity, ride, spin, extension, release height and side, and separation from
the arm's own fastball), how often is it missed? The grade a pitcher gets is the
usage-weighted amount by which his own pitches beat the *same pitch type's*
league whiff rate, which is why the base rate is exported alongside the slopes.

    python -m scripts.stuff_fit                # refit from the Statcast cache
    python -m scripts.stuff_fit --out /tmp/x.json

``scripts.stuff_study`` is the evaluation; this only produces the coefficients.
Refitting rewrites ``mlb_engine/data/stuff_shape.json``, which moves prices, so
the calibration map should be refit after it (``mlb-engine calibrate``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import dedupe_pitches
from mlb_engine.features.stuff import FASTBALL_GROUPS, PITCH_GROUP

CACHE = Path(os.path.expanduser("~/.mlb_engine/cache"))
OUT = Path("mlb_engine/data/stuff_shape.json")
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWING = WHIFF | {"foul", "hit_into_play"}
FEATURES = ["velo", "ivb", "spin", "ext", "rel_z", "rel_x", "velo_diff", "ivb_diff"]
MIN_SWINGS = 400  # below this a pitch type is too rare to fit
RIDGE = 1.0  # mild L2, on standardised features, so a rare shape cannot run away


def logistic_fit(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, float]:
    """Ridge-penalised logistic regression by IRLS, intercept unpenalised."""
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    beta = np.zeros(design.shape[1])
    beta[0] = float(np.log(max(y.mean(), 1e-6) / max(1.0 - y.mean(), 1e-6)))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(design @ beta)))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        z = design @ beta + (y - p) / w
        weighted = design * w[:, None]
        step = np.linalg.solve(weighted.T @ design + penalty, weighted.T @ z)
        if np.max(np.abs(step - beta)) < 1e-10:
            beta = step
            break
        beta = step
    return beta[1:], float(beta[0])


def load_cache() -> pd.DataFrame:
    """Every cached pitch, stitched and de-duplicated across overlapping ranges."""
    files = sorted(glob.glob(str(CACHE / "statcast_*.pkl")))
    if not files:
        raise SystemExit(f"no Statcast caches in {CACHE}")
    frames = [pd.read_pickle(f) for f in files]
    df = dedupe_pitches(pd.concat(frames, ignore_index=True))
    print(f"{len(df):,} pitches from {len(files)} cache file(s)")
    return df


def shape_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "pitcher": pd.to_numeric(df["pitcher"], errors="coerce"),
            "group": df["pitch_type"].map(PITCH_GROUP),
            "velo": pd.to_numeric(df["release_speed"], errors="coerce"),
            "ivb": pd.to_numeric(df["pfx_z"], errors="coerce") * 12.0,
            "spin": pd.to_numeric(df["release_spin_rate"], errors="coerce"),
            "ext": pd.to_numeric(df["release_extension"], errors="coerce"),
            "rel_z": pd.to_numeric(df["release_pos_z"], errors="coerce"),
            "rel_x": pd.to_numeric(df["release_pos_x"], errors="coerce")
            * df["p_throws"].map({"L": -1.0, "R": 1.0}).fillna(1.0),
            "swing": df["description"].astype(str).isin(SWING),
            "whiff": df["description"].astype(str).isin(WHIFF),
        }
    )
    out = out[out["group"].notna()].dropna(
        subset=["velo", "ivb", "spin", "ext", "rel_z", "rel_x", "pitcher"]
    )
    # Separation from the arm's own fastball, as ``features.stuff`` computes it.
    fb = out[out["group"].isin(FASTBALL_GROUPS)].groupby("pitcher")[["velo", "ivb"]].mean()
    out = out.join(fb.rename(columns={"velo": "fb_velo", "ivb": "fb_ivb"}), on="pitcher")
    out["velo_diff"] = (out["velo"] - out["fb_velo"]).fillna(0.0)
    out["ivb_diff"] = (out["ivb"] - out["fb_ivb"]).fillna(0.0)
    return out


def fit(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for group, g in df.groupby("group", sort=True):
        swings = g[g["swing"]]
        if len(swings) < MIN_SWINGS:
            print(f"  {group:9s} skipped: {len(swings)} swings")
            continue
        x = swings[FEATURES].to_numpy(float)
        y = swings["whiff"].to_numpy(int)
        mean, sd = x.mean(0), x.std(0)
        coef, intercept = logistic_fit((x - mean) / sd, y.astype(float))
        groups[str(group)] = {
            "mean": [round(float(v), 6) for v in mean],
            "sd": [round(float(v), 6) for v in sd],
            "coef": [round(float(v), 6) for v in coef],
            "intercept": round(float(intercept), 6),
            "base_whiff": round(float(y.mean()), 6),
            "swings": int(len(swings)),
        }
        top = sorted(zip(FEATURES, coef, strict=True), key=lambda kv: -abs(kv[1]))[:3]
        drivers = ", ".join(f"{k} {v:+.2f}" for k, v in top)
        print(f"  {group:9s} swings {len(swings):7d} whiff {y.mean():.4f}   {drivers}")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    groups = fit(shape_frame(load_cache()))
    payload = {"features": FEATURES, "groups": groups}
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    swings = sum(int(np.asarray(spec["swings"])) for spec in groups.values())
    print(f"wrote {args.out}: {len(groups)} pitch types, {swings:,} swings")


if __name__ == "__main__":
    main()
