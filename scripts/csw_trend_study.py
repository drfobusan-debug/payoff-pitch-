"""Does a starter's three-week CSW% trend predict his next start?

The staleness study says the CSW arrow is the only one of the three that repeats
across a chronological split (backing a declining arm returned -31.8% then
-11.8%), but it says it on 34 graded bets, which is not a sample. This asks the
same question where the sample is large: not "did we win the bet" but "did the
arm actually get worse".

For every start, features are computed from pitches thrown strictly *before* the
game -- the six-week CSW% level the engine already prices, and the trend, being
the last 21 days against the 21 before them. The outcomes are what the props are
built on: strikeouts per plate appearance, and xwOBA allowed per batted ball.

The question is not whether CSW% predicts (it does; that is why it is a level in
the model). It is whether the *change* adds anything once the level is known --
which is the only thing the arrow could be telling a reader that the price does
not already say.

    python -m scripts.csw_trend_study

Finding, 1,734 starts, 187 pitchers, 04-28 to 07-27: it adds nothing. The level
is enormous (z=61 on strikeout rate, z=-39 on xwOBA allowed) and the trend is
z=-1.6 on strikeouts with the halves reversing sign, +3.0 then -4.3. On contact
it is significant with the *wrong* sign -- given the same level, the arm whose
CSW just rose allows slightly worse contact -- and the flat cut agrees: starters
whose CSW fell more than a point went on to allow .3164 against .3190 for those
whose CSW rose. So the arrow stays a description in the report and gets no
weight in a price: the -25% ROI on backing a declining arm is 34 bets of noise.
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

SC = os.path.expanduser("~/.mlb_engine/cache/statcast_2026-04-01_2026-07-27.pkl")
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
CALLED = {"called_strike"}
K_EVENTS = {"strikeout", "strikeout_double_play"}

RECENT_D = 21  # the trend's near half
WINDOW_D = 42  # the level the engine prices
MIN_PITCHES = 120  # per half, so a trend is two or three starts a side
MIN_PA = 12  # in the start being predicted


def load() -> pd.DataFrame:
    df = pd.read_pickle(SC)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["game_date"]).dt.date,
            "pitcher": df["pitcher"].astype("int64"),
            "inning": pd.to_numeric(df["inning"], errors="coerce"),
            "csw": df["description"].astype(str).isin(WHIFF | CALLED),
            "is_pa": df["events"].notna(),
            "is_k": df["events"].astype(str).isin(K_EVENTS),
            "xw": pd.to_numeric(df["estimated_woba_using_speedangle"], errors="coerce"),
        }
    )
    return out


def starts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, pitcher) start: what he did, and how much of it."""
    first = df[df["inning"] == 1].groupby(["date", "pitcher"]).size()
    g = df.groupby(["date", "pitcher"]).agg(
        pa=("is_pa", "sum"),
        k=("is_k", "sum"),
        xw=("xw", "mean"),
        bip=("xw", "count"),
        pitches=("csw", "size"),
    )
    g = g[g.index.isin(first.index)]  # he opened the game
    return g[g["pa"] >= MIN_PA].reset_index()


def features(df: pd.DataFrame, made: pd.DataFrame) -> pd.DataFrame:
    """Six-week CSW% level and the 21d-vs-21d change, from pitches before the game."""
    by_day = (
        df.groupby(["pitcher", "date"])["csw"].agg(["sum", "size"]).reset_index()
    )
    rows = []
    for day, pid, pa, k, xw, bip, _ in made.itertuples(index=False):
        his = by_day[by_day["pitcher"] == pid]
        recent = his[(his["date"] < day) & (his["date"] >= day - timedelta(RECENT_D))]
        prior = his[
            (his["date"] < day - timedelta(RECENT_D))
            & (his["date"] >= day - timedelta(WINDOW_D))
        ]
        if recent["size"].sum() < MIN_PITCHES or prior["size"].sum() < MIN_PITCHES:
            continue
        c_recent = recent["sum"].sum() / recent["size"].sum()
        c_prior = prior["sum"].sum() / prior["size"].sum()
        level = (recent["sum"].sum() + prior["sum"].sum()) / (
            recent["size"].sum() + prior["size"].sum()
        )
        rows.append(
            {
                "date": day,
                "pitcher": pid,
                "level": level,
                "d_csw": c_recent - c_prior,
                "k_rate": k / pa,
                "xwoba": xw,
                "pa": pa,
                "bip": bip,
            }
        )
    return pd.DataFrame(rows)


def ols(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares with an intercept, and the standard errors."""
    x = np.column_stack([np.ones(len(x)), x])
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x, xtw @ y)
    resid = y - x @ beta
    s2 = float((w * resid**2).sum() / (w.sum() - x.shape[1]))
    se = np.sqrt(np.diag(np.linalg.inv(xtw @ x) * s2))
    return beta, se


def fit(d: pd.DataFrame, target: str, weight: str) -> None:
    d = d[d[target].notna() & (d[weight] > 0)]
    x = d[["level", "d_csw"]].to_numpy(float)
    y = d[target].to_numpy(float)
    w = d[weight].to_numpy(float)
    beta, se = ols(x, y, w)
    print(f"\n=== {target} ({len(d):,} starts) ===")
    for name, b, s in zip(("intercept", "csw level", "csw trend"), beta, se, strict=True):
        print(f"  {name:<12}{b:+9.4f}{b / s:8.2f}")

    # A one-standard-deviation fall in the trend, in the units of the outcome.
    sd = float(np.std(d["d_csw"]))
    print(f"  a 1sd ({100 * sd:.1f} pt) fall moves it {-beta[2] * sd:+.4f}")

    # Chronological halves: the condition that has caught every false finding.
    cut = d["date"].quantile(0.5)
    for half, ss in (("train", d[d["date"] <= cut]), ("test", d[d["date"] > cut])):
        b, s = ols(
            ss[["level", "d_csw"]].to_numpy(float),
            ss[target].to_numpy(float),
            ss[weight].to_numpy(float),
        )
        print(f"    {half:<6} n={len(ss):<5} trend {b[2]:+.4f}  z {b[2] / s[2]:+.2f}")

    # And the flat cut a reader of the report would make.
    lo = d[d["d_csw"] <= -0.01]
    hi = d[d["d_csw"] >= 0.01]
    mid = d[(d["d_csw"] > -0.01) & (d["d_csw"] < 0.01)]
    for name, ss in (("falling >1pt", lo), ("flat", mid), ("rising >1pt", hi)):
        print(f"    {name:<14} n={len(ss):<5} {target} {ss[target].mean():.4f}")


def main() -> None:
    df = load()
    made = starts(df)
    d = features(df, made)
    print(
        f"{len(d):,} starts with a readable trend, "
        f"{d['date'].min()} to {d['date'].max()}, {d['pitcher'].nunique()} pitchers"
    )
    fit(d, "k_rate", "pa")
    fit(d, "xwoba", "bip")


if __name__ == "__main__":
    main()
