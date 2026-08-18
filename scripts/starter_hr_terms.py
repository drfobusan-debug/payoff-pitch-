"""Does barrel rate allowed still earn a place on the starter's HR multiplier?

The shipped HR line multiplies four allowed-contact reads together:

    hr = base * hard * (1 + clip((barrel_allowed - .080) * 2.0, -.10, .18))
    hr *= batted_ball_hr_mult()      # GB brake above 50%, FB x hard amplifier
    hr *= 1 + clip((ivb - 15.0) * 0.008, ...)

The barrel term is the oldest of them and the only survivor of the contact-*level*
family #118 deleted from the hits line, where inverse BABIP and dxwOBA allowed
both came back indistinguishable from noise and made the holdout worse. It also
carries a comment claiming it is the highest-PPV HR read, which predates the
ground-ball/fly-ball pair that replaced it in practice, and its 595-BBE
empirical-Bayes prior against a median 95-BBE window leaves it able to move a
home-run rate by about 1%.

So this asks the #195 question: is the term doing anything, and would the model
be better without it? Outcome is home runs allowed per PA in the *next* start,
every feature built from a trailing 42-day window that excludes the start being
predicted, K% controlled (an arm that misses bats allows fewer of everything).
Chronological 60/40 holdout plus a dose search over the barrel slope.
"""

from __future__ import annotations

import glob
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import batted_balls, dedupe_pitches
from mlb_engine.features.regression import (
    BL_BARREL_ALLOWED,
    BL_FB_ALLOWED,
    BL_GB_ALLOWED,
    BL_HARD_HIT,
    BL_IVB,
    FB_ALLOWED_FLOOR,
    FB_HARD_CAP,
    FB_HARD_GAIN,
    FOUR_SEAM_TYPES,
    GB_ALLOWED_CEILING,
    GB_ALLOWED_FLOOR,
    GB_ALLOWED_SLOPE,
    IVB_CLIP,
    IVB_SLOPE,
    STARTER_PRIOR_BBE,
    shrink_starter_rate,
)

CACHE = "/home/ubuntu/.mlb_engine/cache/statcast_*.pkl"
WINDOW = 42
MIN_BBE = 40
K_EV = ["strikeout", "strikeout_double_play"]


def load() -> pd.DataFrame:
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(CACHE))]
    shared = sorted(set.intersection(*(set(f.columns) for f in frames)))
    d = dedupe_pitches(pd.concat([f[shared] for f in frames], ignore_index=True))
    d["date"] = pd.to_datetime(d.game_date)
    d["game"] = (
        d.date.dt.strftime("%Y-%m-%d")
        + "|"
        + d.away_team.astype(str)
        + "@"
        + d.home_team.astype(str)
    )
    first = d[d.inning == 1]
    starters = (
        first.groupby(["game", "inning_topbot"])
        .pitcher.agg(lambda s: s.value_counts().index[0])
        .rename("starter")
        .reset_index()
    )
    d = d.merge(starters, on=["game", "inning_topbot"], how="left")
    return d[d.starter.notna() & (d.pitcher == d.starter)]


def profile(g: pd.DataFrame) -> dict[str, float] | None:
    """The trailing allowed-contact profile the engine would read, shrinkage on."""
    bb = batted_balls(g)
    n_bbe = len(bb)
    if n_bbe < MIN_BBE:
        return None
    ev = g.events.dropna()
    if len(ev) < 60:
        return None
    speed = bb.launch_speed.dropna()
    hard_raw = float((speed >= 95).mean()) if len(speed) else BL_HARD_HIT
    if "launch_speed_angle" in bb:
        barrel_raw = float((bb.launch_speed_angle == 6).mean())
    else:
        barrel_raw = BL_BARREL_ALLOWED
    bbt = bb.bb_type.dropna() if "bb_type" in bb else pd.Series(dtype=object)
    n_bbt = len(bbt)
    gb = float(bbt.eq("ground_ball").mean()) if n_bbt else BL_GB_ALLOWED
    fb = float(bbt.isin(["fly_ball", "popup"]).mean()) if n_bbt else BL_FB_ALLOWED
    ivb = float("nan")
    if "pfx_z" in g and "pitch_type" in g:
        four = g[g.pitch_type.isin(FOUR_SEAM_TYPES)].pfx_z.dropna()
        if len(four) >= 20:
            ivb = float(four.mean() * 12.0)
    return {
        "bbe": float(n_bbe),
        "barrel": shrink_starter_rate(
            barrel_raw, BL_BARREL_ALLOWED, n_bbe, STARTER_PRIOR_BBE["barrel"], 1.0
        ),
        "barrel_raw": barrel_raw,
        "hard": shrink_starter_rate(
            hard_raw, BL_HARD_HIT, n_bbe, STARTER_PRIOR_BBE["hard_hit"], 1.0
        ),
        "gb": gb,
        "fb": fb,
        "ivb": ivb,
        "k_pct": float(ev.isin(K_EV).mean()),
    }


def build(d: pd.DataFrame) -> pd.DataFrame:
    games = d[d.events.notna()].groupby(["game", "date", "pitcher"], as_index=False).agg(
        pa=("events", "size"),
        hr=("events", lambda s: int(s.eq("home_run").sum())),
    )
    by_pitcher = {pid: g.sort_values("date") for pid, g in d.groupby("pitcher")}
    rows = []
    for r in games.itertuples():
        hist = by_pitcher[r.pitcher]
        prior = hist[(hist.date < r.date) & (hist.date >= r.date - timedelta(days=WINDOW))]
        prof = profile(prior)
        if prof is None or r.pa < 10:
            continue
        rows.append({"date": r.date, "pitcher": r.pitcher, "pa": r.pa, "hr": r.hr, **prof})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["ivb"] = out.ivb.fillna(BL_IVB)
    return out


def deviance(p: np.ndarray, hr: np.ndarray, pa: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = hr * np.log(p) + (pa - hr) * np.log1p(-p)
    return float(-2 * ll.sum() / pa.sum())


def fit(x: np.ndarray, hr: np.ndarray, pa: np.ndarray, iters: int = 60) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(hr.sum() / (pa.sum() - hr.sum()))
    for _ in range(iters):
        eta = x @ beta
        p = 1 / (1 + np.exp(-eta))
        w = pa * p * (1 - p)
        z = eta + (hr - pa * p) / np.maximum(w, 1e-9)
        xw = x * w[:, None]
        new = np.linalg.solve(x.T @ xw + 1e-9 * np.eye(x.shape[1]), xw.T @ z)
        if np.max(np.abs(new - beta)) < 1e-9:
            return new
        beta = new
    return beta


def tstats(x: np.ndarray, hr: np.ndarray, pa: np.ndarray, beta: np.ndarray) -> np.ndarray:
    p = 1 / (1 + np.exp(-(x @ beta)))
    w = pa * p * (1 - p)
    cov = np.linalg.inv(x.T @ (x * w[:, None]) + 1e-9 * np.eye(x.shape[1]))
    return beta / np.sqrt(np.diag(cov))


def gb_brake(gb: np.ndarray) -> np.ndarray:
    over = np.maximum(gb - GB_ALLOWED_CEILING, 0.0)
    return np.clip(1.0 - over * GB_ALLOWED_SLOPE, GB_ALLOWED_FLOOR, 1.0)


def fb_amp(fb: np.ndarray, hard: np.ndarray) -> np.ndarray:
    excess = np.maximum(fb - FB_ALLOWED_FLOOR, 0.0) * np.maximum(hard - BL_HARD_HIT, 0.0)
    return 1.0 + np.clip(excess * FB_HARD_GAIN, 0.0, FB_HARD_CAP)


def barrel_term(barrel: np.ndarray, slope: float) -> np.ndarray:
    return 1.0 + np.clip((barrel - BL_BARREL_ALLOWED) * slope, -0.10, 0.18)


def evaluate(df: pd.DataFrame) -> None:
    hr, pa = df.hr.to_numpy(float), df.pa.to_numpy(float)
    k = (df.k_pct - df.k_pct.mean()).to_numpy()
    ones = np.ones(len(df))
    cut = int(len(df) * 0.6)
    print(
        f"\n{len(df)} starts, {int(pa.sum())} PA vs starters, {int(hr.sum())} HR allowed, "
        f"{df.date.min().date()} to {df.date.max().date()}"
    )
    print(f"median trailing BBE {df.bbe.median():.0f}   train {cut} / holdout {len(df) - cut}")

    barrel = (df.barrel - BL_BARREL_ALLOWED).to_numpy()
    barrel_r = (df.barrel_raw - BL_BARREL_ALLOWED).to_numpy()
    hard = (df.hard - BL_HARD_HIT).to_numpy()
    gb = (df.gb - BL_GB_ALLOWED).to_numpy()
    fb = (df.fb - BL_FB_ALLOWED).to_numpy()
    ivb = (df.ivb - BL_IVB).to_numpy()

    specs: dict[str, list[np.ndarray]] = {
        "none (K only)": [],
        "barrel (shrunk, shipped)": [barrel],
        "barrel (raw, unshrunk)": [barrel_r],
        "hard-hit": [hard],
        "GB%": [gb],
        "FB%": [fb],
        "IVB (4-seam)": [ivb],
        "GB + FB": [gb, fb],
        "GB + FB + barrel": [gb, fb, barrel_r],
        "GB + FB + hard + barrel": [gb, fb, hard, barrel_r],
    }
    print(f"\n{'terms':<26} {'coef(s)':>28} {'t':>22} {'train':>9} {'holdout':>9}")
    for name, cols in specs.items():
        x = np.column_stack([ones, k, *cols])
        b_tr = fit(x[:cut], hr[:cut], pa[:cut])
        dev_tr = deviance(1 / (1 + np.exp(-x[:cut] @ b_tr)), hr[:cut], pa[:cut])
        dev_ho = deviance(1 / (1 + np.exp(-x[cut:] @ b_tr)), hr[cut:], pa[cut:])
        b_all = fit(x, hr, pa)
        t_all = tstats(x, hr, pa, b_all)
        coefs = " ".join(f"{c:+7.2f}" for c in b_all[2:]) or "--"
        ts = " ".join(f"{t:+6.2f}" for t in t_all[2:]) or "--"
        print(f"{name:<26} {coefs:>28} {ts:>22} {dev_tr:>9.5f} {dev_ho:>9.5f}")

    print("\nthe shipped multiplier, scored as the engine applies it")
    base_rate = hr[:cut].sum() / pa[:cut].sum()
    shipped_full = (
        barrel_term(df.barrel.to_numpy(), 2.0)
        * gb_brake(df.gb.to_numpy())
        * fb_amp(df.fb.to_numpy(), df.hard.to_numpy())
        * (1.0 + np.clip(ivb * IVB_SLOPE, *IVB_CLIP))
    )
    no_barrel = shipped_full / barrel_term(df.barrel.to_numpy(), 2.0)
    for label, mult in (
        ("flat league rate", np.ones(len(df))),
        ("shipped multiplier", shipped_full),
        ("shipped minus barrel term", no_barrel),
        ("barrel term alone", barrel_term(df.barrel.to_numpy(), 2.0)),
    ):
        p = np.clip(base_rate * mult, 1e-6, 0.2)
        print(
            f"  {label:<28} holdout dev {deviance(p[cut:], hr[cut:], pa[cut:]):.5f}   "
            f"mult mean {mult.mean():.4f} sd {mult.std():.4f} "
            f"range {mult.min():.4f}-{mult.max():.4f}"
        )

    print("\ndose search on the barrel slope (holdout deviance, other terms as shipped)")
    for slope in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0):
        mult = no_barrel * barrel_term(df.barrel.to_numpy(), slope)
        p = np.clip(base_rate * mult, 1e-6, 0.2)
        print(f"  slope {slope:5.1f}   holdout dev {deviance(p[cut:], hr[cut:], pa[cut:]):.5f}")

    print("\nvariants of the shipped multiplier, weekly walk-forward (fixed constants)")
    weeks = df.date.dt.to_period("W")
    order = sorted(weeks.unique())
    variants = {
        "no barrel term": lambda: no_barrel,
        "shipped (slope 2.0 on shrunk)": lambda: shipped_full,
        "slope 2.0 on RAW barrel": lambda: no_barrel * barrel_term(
            df.barrel_raw.to_numpy(), 2.0
        ),
        "slope 17 on shrunk barrel": lambda: no_barrel * barrel_term(
            df.barrel.to_numpy(), 17.0
        ),
        "marginal slope 1.1 on RAW": lambda: no_barrel * barrel_term(
            df.barrel_raw.to_numpy(), 1.1
        ),
    }
    for name, make in variants.items():
        mult = make()
        errs_hr: list[float] = []
        errs_pa: list[float] = []
        preds: list[float] = []
        for i, wk in enumerate(order):
            if i < 4:  # need a few weeks of history for the base rate
                continue
            past = weeks.isin(order[:i]).to_numpy()
            now = (weeks == wk).to_numpy()
            rate = hr[past].sum() / pa[past].sum()
            preds.extend(np.clip(rate * mult[now], 1e-6, 0.2))
            errs_hr.extend(hr[now])
            errs_pa.extend(pa[now])
        print(
            f"  {name:<30} walk-forward dev "
            f"{deviance(np.array(preds), np.array(errs_hr), np.array(errs_pa)):.5f}"
            f"   n={len(preds)}"
        )

    print("\nis the barrel coefficient stable? (raw rate, joint with GB/FB/hard, by half)")
    half = len(df) // 2
    for label, sl in (("first half", slice(0, half)), ("second half", slice(half, len(df)))):
        x = np.column_stack([ones[sl], k[sl], gb[sl], fb[sl], hard[sl], barrel_r[sl]])
        b = fit(x, hr[sl], pa[sl])
        t = tstats(x, hr[sl], pa[sl], b)
        print(f"  {label:<12} barrel {b[5]:+6.2f} (t {t[5]:+5.2f})   GB {b[2]:+6.2f}   FB {b[3]:+6.2f}")

    print("\nreliability: does the trailing read predict the next window at all?")
    for col, name in (("barrel_raw", "barrel allowed (raw)"), ("gb", "GB% allowed"),
                      ("fb", "FB% allowed"), ("hard", "hard-hit allowed")):
        realised = df.hr / df.pa
        r = float(np.corrcoef(df[col], realised)[0, 1])
        print(f"  corr({name:<22}, next-start HR/PA) {r:+.4f}")


if __name__ == "__main__":
    d = load()
    print(f"{len(d)} deduped pitches thrown by starters")
    evaluate(build(d))
