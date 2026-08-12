"""Which contact instrument belongs on the starter's allowed-hits term?

The starter path still prices contact with *inverse BABIP allowed*:

    base *= 1 + clip((BL_BABIP - babip_allowed) * 0.6, -0.08, 0.08)

The bullpen path used the same instrument until #114 measured it on relief PAs
and found it was worse than having no term at all (deviance .41997 vs .41972 for
nothing, .41921 for xwOBAcon level). This asks the same question on the other
half of the innings, where the term is still shipped.

Three candidates, all built from a trailing 42-day window that *excludes* the
game being predicted, with the shipped empirical-Bayes shrinkage applied:

    none        no contact term
    inv_babip   the shipped term
    xwobacon    xwOBAcon allowed level, the pen's replacement
    dxwoba      xwOBAcon - wOBAcon (the "getting bailed out" gap, also shipped)

Outcome is hits per plate appearance against the starter. K% allowed is
controlled throughout: a pitcher who misses bats allows fewer hits for reasons
that have nothing to do with contact quality, and every one of these instruments
correlates with strikeouts.

Run twice, because ``starter_contact_shrink`` defaults to 0.0: the raw pass is
what production actually prices, the shrunk pass is what it would price with the
empirical-Bayes knob turned on.
"""

from __future__ import annotations

import glob
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import batted_balls, dedupe_pitches
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_XBA,
    STARTER_PRIOR_BBE,
    shrink_starter_rate,
)

CACHE = "/home/ubuntu/.mlb_engine/cache/statcast_*.pkl"
WINDOW = 42
MIN_BBE = 40  # matches the engine's MIN_BBE gate on allowed_multipliers
HITS = ["single", "double", "triple", "home_run"]
K_EV = ["strikeout", "strikeout_double_play"]
BB_EV = ["walk", "intent_walk"]


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
    d["vs_starter"] = d.starter.notna() & (d.pitcher == d.starter)
    return d


def _babip(g: pd.DataFrame) -> float:
    ev = g.events.dropna()
    if ev.empty:
        return BL_BABIP
    hits = int(ev.isin(["single", "double", "triple"]).sum())
    hr = int((ev == "home_run").sum())
    k = int(ev.isin(K_EV).sum())
    bb = int(ev.isin(BB_EV).sum())
    denom = len(ev) - k - bb - hr
    return float(hits / denom) if denom > 0 else BL_BABIP


def profile(g: pd.DataFrame, shrink: float) -> dict[str, float] | None:
    """A starter's trailing allowed-contact profile at the engine's shrinkage."""
    bb = batted_balls(g)
    n_bbe = len(bb)
    if n_bbe < MIN_BBE:
        return None
    ev = g.events.dropna()
    if len(ev) < 60:
        return None
    xw = bb.estimated_woba_using_speedangle.dropna()
    wo = bb.woba_value.dropna() if "woba_value" in bb else pd.Series(dtype=float)
    xwobacon = float(xw.mean()) if len(xw) else BL_XBA
    wobacon = float(wo.mean()) if len(wo) else xwobacon
    babip = _babip(g)

    def s(raw: float, baseline: float, prior: float) -> float:
        return shrink_starter_rate(raw, baseline, n_bbe, prior, shrink)

    return {
        "bbe": float(n_bbe),
        "babip": s(babip, BL_BABIP, STARTER_PRIOR_BBE["babip"]),
        "xwobacon": s(xwobacon, BL_XBA, STARTER_PRIOR_BBE["xwoba"]),
        "wobacon": s(wobacon, BL_XBA, STARTER_PRIOR_BBE["xwoba"]),
        "k_pct": float(ev.isin(K_EV).mean()),
        "babip_raw": babip,
        "xwobacon_raw": xwobacon,
    }


def build(d: pd.DataFrame, shrink: float) -> pd.DataFrame:
    """One row per (game, starter): trailing profile + that game's hits allowed."""
    sp = d[d.vs_starter]
    games = sp[sp.events.notna()].groupby(["game", "date", "pitcher"], as_index=False).agg(
        pa=("events", "size"),
        hits=("events", lambda s: int(s.isin(HITS).sum())),
    )
    by_pitcher = {pid: g.sort_values("date") for pid, g in sp.groupby("pitcher")}
    rows = []
    for r in games.itertuples():
        hist = by_pitcher[r.pitcher]
        prior = hist[(hist.date < r.date) & (hist.date >= r.date - timedelta(days=WINDOW))]
        prof = profile(prior, shrink)
        if prof is None or r.pa < 10:
            continue
        rows.append({"date": r.date, "pitcher": r.pitcher, "pa": r.pa, "hits": r.hits, **prof})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def deviance(p: np.ndarray, hits: np.ndarray, pa: np.ndarray) -> float:
    """Mean binomial deviance per PA (lower is better)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = hits * np.log(p) + (pa - hits) * np.log1p(-p)
    return float(-2 * ll.sum() / pa.sum())


def fit(x: np.ndarray, hits: np.ndarray, pa: np.ndarray, iters: int = 60) -> np.ndarray:
    """Binomial IRLS on a design matrix ``x`` (first column intercept)."""
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(hits.sum() / (pa.sum() - hits.sum()))
    for _ in range(iters):
        eta = x @ beta
        p = 1 / (1 + np.exp(-eta))
        w = pa * p * (1 - p)
        z = eta + (hits - pa * p) / np.maximum(w, 1e-9)
        xw = x * w[:, None]
        beta_new = np.linalg.solve(x.T @ xw + 1e-9 * np.eye(x.shape[1]), xw.T @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            return beta_new
        beta = beta_new
    return beta


def evaluate(df: pd.DataFrame) -> None:
    df = df.copy()
    df["dxwoba"] = df.xwobacon - df.wobacon
    hits, pa = df.hits.to_numpy(float), df.pa.to_numpy(float)
    k = (df.k_pct - df.k_pct.mean()).to_numpy()
    ones = np.ones(len(df))

    specs = {
        "none": [],
        "inv_babip": [(BL_BABIP - df.babip).to_numpy()],
        "xwobacon": [(df.xwobacon - BL_XBA).to_numpy()],
        "dxwoba": [df.dxwoba.to_numpy()],
        "xwobacon+dxwoba": [(df.xwobacon - BL_XBA).to_numpy(), df.dxwoba.to_numpy()],
    }
    cut = int(len(df) * 0.6)
    print(f"\n{len(df)} starts, {int(pa.sum())} PA vs starters, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    print(f"train {cut} starts / holdout {len(df) - cut}\n")
    print(f"{'instrument':<18} {'coef':>8} {'t':>7} {'train dev':>11} {'holdout dev':>12}")
    for name, cols in specs.items():
        x = np.column_stack([ones, k, *cols]) if cols else np.column_stack([ones, k])
        b_tr = fit(x[:cut], hits[:cut], pa[:cut])
        dev_tr = deviance(1 / (1 + np.exp(-x[:cut] @ b_tr)), hits[:cut], pa[:cut])
        dev_ho = deviance(1 / (1 + np.exp(-x[cut:] @ b_tr)), hits[cut:], pa[cut:])
        b_all = fit(x, hits, pa)
        if cols:
            eta = x @ b_all
            p = 1 / (1 + np.exp(-eta))
            w = pa * p * (1 - p)
            cov = np.linalg.inv(x.T @ (x * w[:, None]))
            coef, t = b_all[2], b_all[2] / np.sqrt(cov[2, 2])
            print(f"{name:<18} {coef:>8.3f} {t:>7.2f} {dev_tr:>11.5f} {dev_ho:>12.5f}")
        else:
            print(f"{name:<18} {'--':>8} {'--':>7} {dev_tr:>11.5f} {dev_ho:>12.5f}")

    # How far the two shipped terms actually moved a price between them.
    inv = 1 + np.clip((BL_BABIP - df.babip) * 0.6, -0.08, 0.08)
    dx = 1 + np.clip(df.dxwoba * 1.2, -0.06, 0.08)
    print()
    for name, m in (("inverse-BABIP", inv), ("dxwOBA", dx), ("both", inv * dx)):
        print(f"  {name:<14} mean {m.mean():.4f} sd {m.std():.4f} "
              f"range {m.min():.4f}-{m.max():.4f}")
    both = inv * dx
    print(f"  moved the allowed-hit rate >2% in {float((abs(both - 1) > 0.02).mean()) * 100:.0f}%"
          f" of starts, >5% in {float((abs(both - 1) > 0.05).mean()) * 100:.0f}%")
    print(f"  BABIP spread sd {df.babip.std():.4f}, xwOBAcon sd {df.xwobacon.std():.4f}, "
          f"median trailing BBE {df.bbe.median():.0f}")


if __name__ == "__main__":
    d = load()
    print(f"{len(d)} deduped pitches")
    for shrink, label in ((0.0, "raw (production: starter_contact_shrink=0)"), (1.0, "shrunk")):
        print(f"\n===== {label} =====")
        evaluate(build(d, shrink))
