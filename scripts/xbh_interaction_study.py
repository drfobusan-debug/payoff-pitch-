"""Does a ground-ball arm suppress the sweet-spot bat *less* than an average one?

The extra-base multiplier stack carries the bat's sweet-spot rate and the arm's
ground-ball rate allowed as independent factors, and multiplies them. That is an
assumption: it says a sinkerballer takes the same fraction off every hitter's
doubles rate. The plausible alternative is that the hitter who lives on the
sweet spot is the one who beats the sinker, i.e. a positive interaction.

Fitted out of time. Features come from a 42-day window; the outcome is the
plate appearances in the days *after* that window, so a double cannot raise the
sweet-spot rate that predicts it. Usage::

    python -m scripts.xbh_interaction_study \
      --statcast ~/.mlb_engine/cache/statcast_2026-04-01_2026-07-27.pkl \
      --statcast ~/.mlb_engine/cache/statcast_2026-07-03_2026-08-13.pkl
"""

from __future__ import annotations

import argparse
import pickle
from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.features.xtb import LeagueXTB

XBH_EVENTS = ("double", "triple")
TB_VALUE = {"single": 1.0, "double": 2.0, "triple": 3.0, "home_run": 4.0}
BAT_TERMS = ("sweet", "hard", "xwoba", "bat_speed", "xslg", "barrel", "max_ev")
MIN_BBE_BAT = 40  # batted balls needed to read a hitter's sweet-spot rate
MIN_BBE_PIT = 50  # GB% allowed stabilizes by ~50 batted balls


def load_statcast(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        with open(p, "rb") as fh:
            frames.append(pickle.load(fh))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    keys = [k for k in ("game_pk", "at_bat_number", "pitch_number") if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)
    return df


def _window(df: pd.DataFrame, lo: Date, hi: Date) -> pd.DataFrame:
    d = df["game_date"]
    return df[(d >= lo) & (d < hi)]


def sweet_spot_rates(win: pd.DataFrame) -> pd.Series:
    """Share of a hitter's batted balls launched between 8 and 32 degrees."""
    la = win.dropna(subset=["launch_angle"])
    g = la.groupby("batter")["launch_angle"]
    rate = g.apply(lambda s: float(s.between(8, 32).mean()))
    return rate[g.size() >= MIN_BBE_BAT]


def gb_allowed_rates(win: pd.DataFrame) -> pd.Series:
    """Share of the balls in play against a pitcher that are ground balls."""
    bb = win.dropna(subset=["bb_type"])
    g = bb.groupby("pitcher")["bb_type"]
    rate = g.apply(lambda s: float(s.eq("ground_ball").mean()))
    return rate[g.size() >= MIN_BBE_PIT]


def bat_contact_rates(win: pd.DataFrame) -> pd.DataFrame:
    """Per-hitter contact measures over the window, on batted balls only.

    ``sweet`` is what the extra-base multiplier currently reads. The others are
    the alternatives available in the same frame -- including ``xslg`` off the
    engine's own expected-total-bases grid, the largest term in the stack -- so
    each is judged beside the rest rather than in isolation.
    """
    bb = win.dropna(subset=["launch_angle"])
    g = bb.groupby("batter")
    out = pd.DataFrame(
        {
            "sweet": g["launch_angle"].apply(lambda s: float(s.between(8, 32).mean())),
            "hard": g["launch_speed"].apply(lambda s: float((s >= 95).mean())),
            "xwoba": g["estimated_woba_using_speedangle"].mean(),
            "bat_speed": g["bat_speed"].mean(),
            # The two terms the total-bases selector weights heaviest.
            "barrel": g["launch_speed_angle"].apply(
                lambda s: float(s.dropna().eq(6).mean()) if s.notna().any() else float("nan")
            ),
            "max_ev": g["launch_speed"].max(),
            "bbe": g.size(),
        }
    )
    grid = LeagueXTB.from_statcast(win)
    if grid is None:
        out["xslg"] = float("nan")
    else:
        exp = grid.expected(bb)
        out["xslg"] = exp.groupby(bb.loc[exp.index, "batter"]).mean()
    return out[out["bbe"] >= MIN_BBE_BAT].dropna()


def forward_pa(df: pd.DataFrame, lo: Date, hi: Date, target: str) -> pd.DataFrame:
    """One row per plate appearance in [lo, hi), carrying the target flag.

    ``xbh`` is 2B/3B per PA with home runs left in the denominator as failures.
    A ball that clears the fence is not a double, so that measure charges a
    slugger for his power; ``xbh_of_bip`` drops home runs and strikeouts to ask
    the narrower question -- given contact, is it a double -- and ``xbh_hr``
    puts home runs in the target instead.
    """
    fwd = _window(df, lo, hi)
    pa = fwd[fwd["events"].notna()].copy()
    if target == "xbh_of_bip":
        pa = pa[pa["bb_type"].notna() & (pa["events"] != "home_run")]
    if target == "tb":
        # Total bases per PA, home runs included at four: the selector's own
        # market, where contact quality has a claim the doubles line does not.
        pa["xbh"] = pa["events"].map(TB_VALUE).fillna(0.0)
    else:
        hit = [*XBH_EVENTS, "home_run"] if target == "xbh_hr" else list(XBH_EVENTS)
        pa["xbh"] = pa["events"].isin(hit).astype(float)
    return pa[["game_date", "batter", "pitcher", "xbh"]]


def fit_logit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson logistic fit; returns coefficients and standard errors."""
    beta = np.zeros(x.shape[1])
    for _ in range(60):
        eta = x @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        grad = x.T @ (y - p)
        hess = x.T @ (x * w[:, None])
        step = np.linalg.solve(hess + 1e-9 * np.eye(x.shape[1]), grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(x.T @ (x * w[:, None]) + 1e-9 * np.eye(x.shape[1]))
    return beta, np.sqrt(np.diag(cov))


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares with heteroskedasticity-robust standard errors."""
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    meat = x.T @ (x * (resid**2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def two_sided_p(z: float) -> float:
    """Normal tail probability without scipy."""
    from math import erfc, sqrt

    return float(erfc(abs(z) / sqrt(2.0)))


def build_rolls(
    df: pd.DataFrame, window_days: int, forward_days: int, rolls: int, target: str
) -> pd.DataFrame:
    """Stack (features from a window, outcomes from the days after it) blocks."""
    last = max(df["game_date"])
    first = min(df["game_date"])
    out = []
    for i in range(rolls):
        fwd_hi = last - timedelta(days=forward_days * i)
        fwd_lo = fwd_hi - timedelta(days=forward_days)
        win_lo = fwd_lo - timedelta(days=window_days)
        if win_lo < first:
            break
        win = _window(df, win_lo, fwd_lo)
        bat = bat_contact_rates(win)
        pit = gb_allowed_rates(win)
        pa = forward_pa(df, fwd_lo, fwd_hi, target)
        pa = pa[pa["batter"].isin(bat.index) & pa["pitcher"].isin(pit.index)].copy()
        if pa.empty:
            continue
        for col in (*BAT_TERMS, "bbe"):
            pa[col] = pa["batter"].map(bat[col])
        pa["gb"] = pa["pitcher"].map(pit)
        pa["roll"] = i
        out.append(pa)
        print(
            f"  roll {i}: window {win_lo}..{fwd_lo} -> games {fwd_lo}..{fwd_hi}   "
            f"{len(pa)} PA, {len(bat)} bats, {len(pit)} arms, "
            f"2B+3B {pa['xbh'].mean() * 100:.2f}%"
        )
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statcast", action="append", required=True)
    ap.add_argument("--window-days", type=int, default=42)
    ap.add_argument("--forward-days", type=int, default=7)
    ap.add_argument("--rolls", type=int, default=8)
    ap.add_argument(
        "--target",
        default="xbh",
        choices=("xbh", "xbh_of_bip", "xbh_hr", "tb"),
        help="2B+3B per PA, per ball in play, 2B+3B+HR per PA, or total bases per PA",
    )
    args = ap.parse_args()

    df = load_statcast(args.statcast)
    print(
        f"{len(df)} pitches, {min(df['game_date'])}..{max(df['game_date'])}\n"
        f"out-of-time rolls (features from {args.window_days}d, "
        f"outcome the next {args.forward_days}d):"
    )
    d = build_rolls(df, args.window_days, args.forward_days, args.rolls, args.target)
    if d.empty:
        print("no usable rolls")
        return

    sweet = (d["sweet"] - d["sweet"].mean()) / d["sweet"].std()
    gb = (d["gb"] - d["gb"].mean()) / d["gb"].std()
    y = d["xbh"].to_numpy()
    ones = np.ones(len(d))

    print(f"\n{len(d)} rows, target '{args.target}' rate {y.mean() * 100:.2f}%")
    print(f"\nlogistic on {args.target}, inputs in standard deviations")
    print(f"  {'term':<22}{'coef':>9}{'se':>8}{'z':>7}{'p':>9}")
    designs = {
        "main effects only": np.column_stack([ones, sweet, gb]),
        "with interaction": np.column_stack([ones, sweet, gb, sweet * gb]),
    }
    names = {
        "main effects only": ["intercept", "sweet_spot", "gb_allowed"],
        "with interaction": [
            "intercept",
            "sweet_spot",
            "gb_allowed",
            "sweet_spot x gb",
        ],
    }
    zs = {
        c: ((d[c] - d[c].mean()) / d[c].std()).to_numpy() for c in (*BAT_TERMS, "gb")
    }
    label = "every contact measure in the extra-base stack, together"
    designs[label] = np.column_stack([ones, *(zs[c] for c in (*BAT_TERMS, "gb"))])
    names[label] = ["intercept", *BAT_TERMS, "gb_allowed"]
    # And each on its own, since a dead term beside four others can be a dead
    # term or a collinear one.
    for c in BAT_TERMS:
        designs[f"{c} alone"] = np.column_stack([ones, zs[c]])
        names[f"{c} alone"] = ["intercept", c]
    # Max exit velocity is the maximum of a sample, so it rises with the number
    # of batted balls whether or not the hitter is stronger. Anything it claims
    # has to survive the batted-ball count sitting beside it.
    if "max_ev" in BAT_TERMS:
        bbe = ((d["bbe"] - d["bbe"].mean()) / d["bbe"].std()).to_numpy()
        designs["max_ev, controlling for batted balls"] = np.column_stack(
            [ones, zs["max_ev"], bbe]
        )
        names["max_ev, controlling for batted balls"] = ["intercept", "max_ev", "bbe"]
    fit = fit_ols if args.target == "tb" else fit_logit
    for label, x in designs.items():
        beta, se = fit(x, y)
        print(f"  -- {label}")
        for nm, b, s in zip(names[label], beta, se, strict=True):
            z = b / s if s > 0 else 0.0
            print(f"  {nm:<22}{b:>9.4f}{s:>8.4f}{z:>7.2f}{two_sided_p(z):>9.4f}")

    # Does the interaction hold its sign across the rolls? A real effect says the
    # same thing in every block; a lucky one does not.
    print("\nper-roll interaction coefficient")
    for r, g in d.groupby("roll"):
        if len(g) < 2000 or g["xbh"].sum() < 40:
            print(f"  roll {r}: too thin ({len(g)} PA)")
            continue
        s = (g["sweet"] - d["sweet"].mean()) / d["sweet"].std()
        b_ = (g["gb"] - d["gb"].mean()) / d["gb"].std()
        x = np.column_stack([np.ones(len(g)), s, b_, s * b_])
        beta, se = fit(x, g["xbh"].to_numpy())
        print(
            f"  roll {r}: n={len(g):<6} sweet {beta[1]:+.4f}  gb {beta[2]:+.4f}  "
            f"interaction {beta[3]:+.4f} (se {se[3]:.4f})"
        )

    # Same question of any term that did survive the pooled fit: one block at a
    # time, with the batted-ball count controlled.
    if "max_ev" in BAT_TERMS:
        print("\nper-roll max_ev coefficient, controlling for batted balls")
        for r, g in d.groupby("roll"):
            if len(g) < 2000:
                continue
            mev = (g["max_ev"] - d["max_ev"].mean()) / d["max_ev"].std()
            nbbe = (g["bbe"] - d["bbe"].mean()) / d["bbe"].std()
            x = np.column_stack([np.ones(len(g)), mev, nbbe])
            beta, se = fit(x, g["xbh"].to_numpy())
            print(f"  roll {r}: n={len(g):<6} max_ev {beta[1]:+.4f} (se {se[1]:.4f})")

    # And what it would be worth: the implied rate in the four corners.
    print("\nrealised 2B+3B per PA by tercile, as the interaction would show up")
    st = pd.qcut(d["sweet"], 3, labels=["low", "mid", "high"])
    gt = pd.qcut(d["gb"], 3, labels=["low", "mid", "high"])
    tab = d.groupby([st, gt], observed=True)["xbh"].agg(["mean", "size"])
    print(f"  {'sweet':<7}{'gb':<7}{'2B+3B%':>9}{'n':>9}")
    for (a, b), row in tab.iterrows():
        print(f"  {str(a):<7}{str(b):<7}{row['mean'] * 100:>9.2f}{int(row['size']):>9}")


if __name__ == "__main__":
    main()
