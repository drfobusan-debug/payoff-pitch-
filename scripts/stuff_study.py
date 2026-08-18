"""Does a pitch-shape grade add anything to the strikeout read already priced?

The engine's small-sample strikeout prior is xK% = 0.34*CSW% + 0.71*SwStr%, and
#195's finding was that the whiff family is one signal counted several ways: the
K multiplier had to be deleted because the stuff it charged for was already
inside the blended rate it multiplied. So the question here is not whether shape
predicts strikeouts -- of course it does -- but whether a *physical* measurement
of the pitch adds to the *results* the engine already reads.

Method, all point-in-time. The shape -> whiff models are fitted on pitches
through 2026-06-15 only (``scripts.stuff_fit`` ships the same form fitted on
everything). Each start is then graded off pitches thrown in the 42 days
*strictly before* it, and scored against the strikeouts in that start. Holdout
is chronological; the walk-forward refits each week on every prior week. The
leak-free subset drops every start within a week of the model's fitting window.

    python -m scripts.stuff_study

Findings, 2,901 starter-starts, 245 pitchers, 2026-04-07..08-15:

1. It adds, out of sample. On the 1,230 leak-free starts the blended prediction
   the engine actually prices improves from wRMSE 0.10141 to 0.10097, and the
   prior on its own -- where a prior should be judged -- from 0.10280 to 0.10105.
   In a weekly walk-forward against the same baseline, deviance goes 1.04281 ->
   1.04195, and 1.04170 with four-seam velocity beside it, so the grade and the
   velocity read are not the same thing either.

2. The signal is shape, not arsenal. Splitting the grade into "which pitch types
   he throws" and "how good his versions of them are": the mix half fits with the
   *wrong sign* (-0.170) and leaves the error where it was (0.10138 vs 0.10141),
   while the shape half carries all of it (+0.582, 0.10098). So the league rate
   of each pitch type is subtracted out rather than kept -- a grade that rewarded
   throwing sliders would be measuring the arsenal, not the arm.

3. It does not rescue thin samples, which was the reason to expect it. Under 60
   trailing PA -- 78 starts, first or second outings -- adding it made the
   prediction worse (0.11072 -> 0.11236). The gain is in the 60-180 PA band. The
   hypothesis was right that shape helps and wrong about where.

4. Dose is flat: 0.25/0.40/0.60/0.90 give 0.10115/0.10105/0.10097/0.10100
   against 0.10141 for none of it. The fitted value is +0.58 and 0.6 ships.

5. It is roughly EV Analytics' THE BATcast+, independently arrived at. Measured
   outside this script, against a user-supplied 2026 leaderboard export (that
   data is not redistributable, so it is not a step here): on 411 shared arms the
   season-long grade correlates +0.51 with theirs, and +0.68 at the individual
   pitch level (fastball +0.54, slider +0.51, sinker +0.30). Same family, not the
   same number -- and unlike a season-to-date leaderboard, this one is dated, so
   it can be tested at all.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import dedupe_pitches
from mlb_engine.features.regression import (
    BL_CSW,
    BL_K_PCT,
    BL_SWSTR,
    XK_CSW_COEF,
    XK_SWSTR_COEF,
)
from mlb_engine.features.stuff import FASTBALL_GROUPS, PITCH_GROUP
from scripts.stuff_fit import FEATURES, MIN_SWINGS, WHIFF, logistic_fit

CACHE = Path(os.path.expanduser("~/.mlb_engine/cache"))
SWING = WHIFF | {"foul", "hit_into_play"}
CALLED = {"called_strike"}
K_EVENTS = {"strikeout", "strikeout_double_play"}
WINDOW = 42  # days, the window the engine prices
TRAIN_END = pd.Timestamp("2026-06-15")
GAP_DAYS = 7  # between the model's fitting window and the leak-free subset
K_PRIOR_PA = 150.0  # ``blend_k_rate``'s prior weight
MIN_WINDOW_PITCHES = 100
MIN_START_PA = 10


def load() -> pd.DataFrame:
    files = sorted(glob.glob(str(CACHE / "statcast_*.pkl")))
    raw = dedupe_pitches(pd.concat([pd.read_pickle(f) for f in files], ignore_index=True))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["game_date"]),
            "pitcher": pd.to_numeric(raw["pitcher"], errors="coerce"),
            "inning": pd.to_numeric(raw["inning"], errors="coerce"),
            "group": raw["pitch_type"].map(PITCH_GROUP),
            "velo": pd.to_numeric(raw["release_speed"], errors="coerce"),
            "ivb": pd.to_numeric(raw["pfx_z"], errors="coerce") * 12.0,
            "spin": pd.to_numeric(raw["release_spin_rate"], errors="coerce"),
            "ext": pd.to_numeric(raw["release_extension"], errors="coerce"),
            "rel_z": pd.to_numeric(raw["release_pos_z"], errors="coerce"),
            "rel_x": pd.to_numeric(raw["release_pos_x"], errors="coerce")
            * raw["p_throws"].map({"L": -1.0, "R": 1.0}).fillna(1.0),
            "swing": raw["description"].astype(str).isin(SWING),
            "whiff": raw["description"].astype(str).isin(WHIFF),
            "called": raw["description"].astype(str).isin(CALLED),
            "pa": raw["events"].notna(),
            "k": raw["events"].astype(str).isin(K_EVENTS),
        }
    )
    df = df[df["group"].notna()].dropna(subset=FEATURES[:6] + ["pitcher"])
    fb = df[df["group"].isin(FASTBALL_GROUPS)].groupby(["pitcher", "date"])[["velo", "ivb"]].mean()
    df = df.join(fb.rename(columns={"velo": "fb_velo", "ivb": "fb_ivb"}), on=["pitcher", "date"])
    df["velo_diff"] = (df["velo"] - df["fb_velo"]).fillna(0.0)
    df["ivb_diff"] = (df["ivb"] - df["fb_ivb"]).fillna(0.0)
    print(f"{len(df):,} pitches, {df['pitcher'].nunique()} pitchers, "
          f"{df['date'].min().date()}..{df['date'].max().date()}")
    return df


def grade_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Score every pitch with models fitted on the training slice only."""
    train = df[df["date"] <= TRAIN_END]
    df = df.copy()
    df["p_whiff"] = np.nan
    df["base_whiff"] = np.nan
    for group, g in train.groupby("group", sort=False):
        swings = g[g["swing"]]
        if len(swings) < MIN_SWINGS:
            continue
        x = swings[FEATURES].to_numpy(float)
        mean, sd = x.mean(0), x.std(0)
        coef, intercept = logistic_fit((x - mean) / sd, swings["whiff"].to_numpy(float))
        rows = df["group"] == group
        z = intercept + ((df.loc[rows, FEATURES].to_numpy(float) - mean) / sd) @ coef
        df.loc[rows, "p_whiff"] = 1.0 / (1.0 + np.exp(-z))
        df.loc[rows, "base_whiff"] = float(swings["whiff"].mean())
    return df.dropna(subset=["p_whiff"])


def starts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per start: the trailing window's features, that start's outcome."""
    daily = (
        df.assign(edge=df["p_whiff"] - df["base_whiff"])
        .groupby(["pitcher", "date"])
        .agg(
            n=("velo", "size"),
            edge_sum=("edge", "sum"),
            mix_sum=("base_whiff", "sum"),
            whiffs=("whiff", "sum"),
            called=("called", "sum"),
            pa=("pa", "sum"),
            k=("k", "sum"),
            first_inning=("inning", "min"),
        )
        .reset_index()
    )
    cols = ["n", "edge_sum", "mix_sum", "whiffs", "called", "pa", "k"]
    out = []
    for _, g in daily.groupby("pitcher", sort=False):
        g = g.sort_values("date").set_index("date")
        prior = g[cols].rolling(f"{WINDOW}D").sum() - g[cols]  # the day itself excluded
        prior.columns = ["w_" + c for c in cols]
        out.append(pd.concat([g.reset_index(), prior.reset_index(drop=True)], axis=1))
    t = pd.concat(out, ignore_index=True)
    t = t[
        (t["first_inning"] == 1)
        & (t["pa"] >= MIN_START_PA)
        & (t["w_n"] >= MIN_WINDOW_PITCHES)
    ].copy()
    t["shape"] = t["w_edge_sum"] / t["w_n"]
    t["mix"] = t["w_mix_sum"] / t["w_n"]
    t["csw"] = (t["w_whiffs"] + t["w_called"]) / t["w_n"]
    t["swstr"] = t["w_whiffs"] / t["w_n"]
    t["xk"] = BL_K_PCT + XK_CSW_COEF * (t["csw"] - BL_CSW) + XK_SWSTR_COEF * (t["swstr"] - BL_SWSTR)
    t["engine_k"] = (t["w_k"] + t["xk"] * K_PRIOR_PA) / (t["w_pa"] + K_PRIOR_PA)
    return t.dropna(subset=["shape", "mix", "engine_k"]).reset_index(drop=True)


def wrmse(p: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    return float(np.sqrt(np.sum(n * (k / n - p) ** 2) / n.sum()))


def blended(prior: np.ndarray, t: pd.DataFrame) -> np.ndarray:
    wk = t["w_k"].to_numpy(float)
    wpa = t["w_pa"].to_numpy(float)
    return (wk + np.clip(prior, 0.02, 0.6) * K_PRIOR_PA) / (wpa + K_PRIOR_PA)


def main() -> None:
    t = starts(grade_pitches(load()))
    shape = (t["shape"] - t["shape"].mean()).to_numpy(float)
    k = t["k"].to_numpy(float)
    n = t["pa"].to_numpy(float)
    tr = (t["date"] <= TRAIN_END).to_numpy()
    te = (t["date"] > TRAIN_END + pd.Timedelta(days=GAP_DAYS)).to_numpy()
    print(f"{len(t)} starts, {t['pitcher'].nunique()} pitchers; fit {tr.sum()} / score {te.sum()}")
    print(f"grade: sd {shape.std():.4f}, 1st/99th {np.quantile(shape, 0.01):+.4f}/"
          f"{np.quantile(shape, 0.99):+.4f}")

    base = t["xk"].to_numpy(float)
    resid = k / n - base
    fitted = float(np.sum(n[tr] * shape[tr] * resid[tr]) / np.sum(n[tr] * shape[tr] ** 2))
    print(f"\nfitted shape coefficient on the shipped line: {fitted:+.3f}")

    print("\nheld-out error of the blended prediction the engine prices:")
    print(f"  shipped line                {wrmse(blended(base, t)[te], k[te], n[te]):.5f}")
    for dose in (0.25, 0.4, 0.6, 0.9):
        p = blended(base + dose * np.clip(shape, -0.05, 0.05), t)
        print(f"  + {dose:.2f} x grade           {wrmse(p[te], k[te], n[te]):.5f}")

    print("\nby how thin the trailing window is:")
    for lo, hi in ((0, 60), (60, 100), (100, 140), (140, 180), (180, 10_000)):
        m = te & (t["w_pa"].to_numpy() >= lo) & (t["w_pa"].to_numpy() < hi)
        if m.sum() < 30:
            continue
        off = wrmse(blended(base, t)[m], k[m], n[m])
        on = wrmse(blended(base + 0.6 * np.clip(shape, -0.05, 0.05), t)[m], k[m], n[m])
        print(f"  {lo:4d}-{hi:<6d} n={m.sum():4d}  without {off:.5f}  with {on:.5f}  "
              f"{'better' if on < off else 'WORSE'}")

    print("\nshape against arsenal composition (which half carries it):")
    mix = (t["mix"] - t["mix"].mean()).to_numpy(float)
    for label, term in (("shape only", shape), ("mix only", mix)):
        coef = float(np.sum(n[tr] * term[tr] * resid[tr]) / np.sum(n[tr] * term[tr] ** 2))
        p = blended(base + coef * np.clip(term, -0.05, 0.05), t)
        print(f"  {label:12s} coefficient {coef:+.3f}  held-out {wrmse(p[te], k[te], n[te]):.5f}")


if __name__ == "__main__":
    main()
