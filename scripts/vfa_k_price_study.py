"""Should the four-seam velocity term be priced, on the bar that retired the last one?

``PitcherRegression.velocity_k_multiplier`` is fitted and shipped unpriced behind
``MLBE_VFA_K_WEIGHT`` (default 0.0). Its only consumer is ``k_multiplier``, which
#190's successor retired from the strikeout path entirely, so the weight is dead
wiring today: turning it to 1.0 changes no probability on any slate (verified in
``scripts/vfa_k_backtest.py`` -- 6,675 graded picks, identical Brier to five
decimals). Pricing velocity means putting it where the rate is actually built,
on the blended rate, and that has to clear the bar the retired multiplier failed.

Same panel as ``scripts/k_multiplier_study.py``: every start predicted from
pitches thrown strictly before it over the engine's 42-day window, weighted by
the plate appearances he faced, scored by weekly walk-forward wRMSE against the
blended rate alone (0.09763 there, the number the stuff multiplier's 0.10400 lost
to). Two terms, exactly as fitted: his window four-seam level against the league,
and how his most recent start sat against that level.

    python -m scripts.vfa_k_price_study

It passes here -- 0.09634 against 0.09749 for the blend alone, with the dose
search keeping ~0.8 of the fitted term where the same search kept none of stuff
-- and then fails as a price: nine graded slates replayed at both weights move
strikeout Brier .20197 -> .20166 but log loss .60567 -> .62127, and 16 of the 18
other markets worse, because scaling a starter's K rate rescales every other
outcome he allows. So the wiring is live, ``MLBE_VFA_K_WEIGHT`` ships at 0, and
what would change that is more graded slates rather than a better fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_engine.features.regression import (
    BL_CSW,
    BL_VFA,
    FOUR_SEAM_TYPES,
    K_EVENTS_P,
    MIN_VFA_PITCHES,
    MIN_VFA_START,
    VFA_K_DEV_CLIP,
    VFA_K_DEV_SLOPE,
    VFA_K_LEVEL_CLIP,
    VFA_K_LEVEL_SLOPE,
    WHIFF_DESC,
)
from mlb_engine.features.rolling import blend_k_rate, rates_from_events
from scripts.xk_refit_study import (
    K_CLIP,
    MIN_PA_START,
    MIN_PA_WINDOW,
    WINDOW_D,
    load_pitches,
    wrmse,
)

SUMS = ["pitches", "n_csw", "n_whiff", "pa", "n_k", "n_bb", "n_fb", "sum_fb"]


def per_start(df: pd.DataFrame) -> pd.DataFrame:
    """One row per starter-start, carrying its four-seam radar total."""
    d = df.copy()
    d["is_pa"] = d["events"].notna() & (d["events"] != "")
    d["k"] = d["is_pa"] & d["events"].isin(K_EVENTS_P)
    d["bb"] = d["is_pa"] & d["events"].isin(["walk", "hit_by_pitch"])
    d["csw"] = d["description"].isin(["called_strike", *WHIFF_DESC])
    d["whiff"] = d["description"].isin(WHIFF_DESC)
    d["opened"] = d["inning"] <= 1
    fb = d["pitch_type"].isin(FOUR_SEAM_TYPES) & d["release_speed"].notna()
    d["is_fb"] = fb
    d["fb_speed"] = np.where(fb, pd.to_numeric(d["release_speed"], errors="coerce"), 0.0)
    g = d.groupby(["pitcher", "game_key", "game_date"], as_index=False).agg(
        pitches=("csw", "size"),
        n_csw=("csw", "sum"),
        n_whiff=("whiff", "sum"),
        pa=("is_pa", "sum"),
        n_k=("k", "sum"),
        n_bb=("bb", "sum"),
        n_fb=("is_fb", "sum"),
        sum_fb=("fb_speed", "sum"),
        opened=("opened", "max"),
    )
    return g[g["opened"] & (g["pa"] >= MIN_PA_START)].reset_index(drop=True)


def panel(starts: pd.DataFrame, days: int = WINDOW_D) -> pd.DataFrame:
    """Each start paired with the window, and with his previous start's velocity.

    ``vfa`` and ``vfa_dev`` follow ``_four_seam_velocity``: the level needs 60
    four-seamers in the window and the deviation needs 15 in the start it is
    measured from, so an arm who barely throws the pitch reads as unavailable
    rather than as league average.
    """
    rows = []
    for pid, g in starts.groupby("pitcher"):
        g = g.sort_values("game_date").reset_index(drop=True)
        dates = g["game_date"].to_numpy()
        arr = g[SUMS].to_numpy(dtype=float)
        n_fb = g["n_fb"].to_numpy(dtype=float)
        sum_fb = g["sum_fb"].to_numpy(dtype=float)
        floor = dates - np.timedelta64(days, "D")
        for i in range(len(g)):
            mask = (dates < dates[i]) & (dates >= floor[i])
            if not mask.any():
                continue
            row = {"pitcher": pid, "game_date": g["game_date"].iloc[i]}
            row.update({c: v for c, v in zip(SUMS, arr[mask].sum(axis=0), strict=True)})
            prev = int(np.flatnonzero(mask)[-1])
            row["last_fb_n"] = n_fb[prev]
            row["last_fb_speed"] = sum_fb[prev] / n_fb[prev] if n_fb[prev] else np.nan
            row["y_pa"] = float(g["pa"].iloc[i])
            row["y_k"] = float(g["n_k"].iloc[i])
            rows.append(row)
    p = pd.DataFrame(rows)
    p["csw"] = p.n_csw / p.pitches
    p["swstr"] = p.n_whiff / p.pitches
    p["k_pct"] = p.n_k / p.pa
    p["bb_pct"] = p.n_bb / p.pa
    p["y_k_pct"] = p.y_k / p.y_pa
    p["vfa"] = np.where(p.n_fb >= MIN_VFA_PITCHES, p.sum_fb / p.n_fb.replace(0, np.nan), np.nan)
    p["vfa_dev"] = np.where(p.last_fb_n >= MIN_VFA_START, p.last_fb_speed - p.vfa, np.nan)
    p = p[(p.pa >= MIN_PA_WINDOW) & (p.pitches >= 100)]
    return p.sort_values("game_date").reset_index(drop=True)


def engine_k_rate(p: pd.DataFrame) -> pd.Series:
    """The rate the simulator holds: window K% blended toward xK% at 150 PA."""
    out = []
    for row in p.itertuples():
        events = pd.Series(
            ["strikeout"] * int(row.n_k)
            + ["walk"] * int(row.n_bb)
            + ["field_out"] * int(row.pa - row.n_k - row.n_bb)
        )
        xk = float(np.clip(0.220 + 0.34 * (row.csw - BL_CSW) + 0.71 * (row.swstr - 0.110), *K_CLIP))
        out.append(blend_k_rate(rates_from_events(events), xk).p_k)
    return pd.Series(out, index=p.index)


def shipped_multiplier(p: pd.DataFrame, dose: float = 1.0) -> np.ndarray:
    """``velocity_k_multiplier`` at ``vfa_k=dose``, on a whole panel at once."""
    m = np.ones(len(p))
    lvl = p.vfa.to_numpy()
    dev = p.vfa_dev.to_numpy()
    ok = ~np.isnan(lvl)
    m[ok] *= 1.0 + dose * np.clip(
        (lvl[ok] - BL_VFA) * VFA_K_LEVEL_SLOPE, *VFA_K_LEVEL_CLIP
    )
    okd = ~np.isnan(dev)
    m[okd] *= 1.0 + dose * np.clip(dev[okd] * VFA_K_DEV_SLOPE, *VFA_K_DEV_CLIP)
    return m


def _design(p: pd.DataFrame) -> pd.DataFrame:
    """Both terms, zero where unreadable so an arm without a four-seamer is neutral."""
    return sm.add_constant(
        pd.DataFrame(
            {
                "level": np.where(p.vfa.notna(), p.vfa.fillna(BL_VFA) - BL_VFA, 0.0),
                "dev": p.vfa_dev.fillna(0.0).to_numpy(),
            },
            index=p.index,
        )
    )


def fitted(train: pd.DataFrame, test: pd.DataFrame, base_tr: pd.Series, base_te: pd.Series):
    """Fit velocity as what it would be priced as: a correction to the blend."""
    y = np.log(np.clip(train.y_k_pct.to_numpy(), 0.02, None) / base_tr.to_numpy())
    res = sm.WLS(y, _design(train), weights=train.y_pa).fit()
    pred = np.exp(res.predict(_design(test)).to_numpy()) * base_te.to_numpy()
    return res, pred


def main() -> None:
    pitches = load_pitches()
    p = panel(per_start(pitches))
    base = engine_k_rate(p)
    p = p.assign(base=base)
    readable = int(p.vfa.notna().sum())
    print(
        f"{len(p)} starts by {p.pitcher.nunique()} pitchers, "
        f"{p.game_date.min().date()}..{p.game_date.max().date()}; "
        f"{readable} with a readable four-seam level ({readable / len(p) * 100:.0f}%), "
        f"{int(p.vfa_dev.notna().sum())} with a last-start deviation"
    )

    cut = p.game_date.quantile(0.6)
    tr, te = p[p.game_date <= cut], p[p.game_date > cut]
    print(f"\n===== chronological holdout ({len(tr)} train / {len(te)} test) =====")
    print(f"  {'blended rate alone':<28} wRMSE {wrmse(te.base, te.y_k_pct, te.y_pa):.5f}")
    for dose in (0.5, 1.0):
        pred = te.base.to_numpy() * shipped_multiplier(te, dose)
        print(f"  {f'x shipped term, dose {dose}':<28} wRMSE {wrmse(pred, te.y_k_pct, te.y_pa):.5f}")
    res, pred = fitted(tr, te, tr.base, te.base)
    print(f"  {'x refitted term':<28} wRMSE {wrmse(pred, te.y_k_pct, te.y_pa):.5f}")
    print("\n  refitted slopes (log K rate against the blend):")
    for name in ("const", "level", "dev"):
        print(
            f"    {name:<6} {res.params[name]:+.5f}  t={res.tvalues[name]:+.2f}  "
            f"p={res.pvalues[name]:.3f}"
        )
    print(f"    shipped for comparison: level {VFA_K_LEVEL_SLOPE:+.3f}  dev {VFA_K_DEV_SLOPE:+.3f}")

    print("\n===== weekly walk-forward (fit on every prior week) =====")
    weeks = p.game_date.dt.to_period("W")
    order = sorted(weeks.unique())
    doses_fixed = (0.25, 0.5, 0.65, 0.8, 1.0, 1.25)
    chunks: dict[str, list[np.ndarray]] = {"base": [], "shipped": [], "refit": []}
    chunks.update({f"dose {d}": [] for d in doses_fixed})
    ys, ws, doses = [], [], []
    for wk in order[4:]:
        tr, te = p[weeks < wk], p[weeks == wk]
        if len(te) == 0 or len(tr) < 200:
            continue
        chunks["base"].append(te.base.to_numpy())
        chunks["shipped"].append(te.base.to_numpy() * shipped_multiplier(te))
        for d in doses_fixed:
            chunks[f"dose {d}"].append(te.base.to_numpy() * shipped_multiplier(te, d))
        _, pred = fitted(tr, te, tr.base, te.base)
        chunks["refit"].append(pred)
        # How much of the shipped term the prior weeks would have kept.
        grid = np.arange(0.0, 2.01, 0.1)
        doses.append(
            min(grid, key=lambda d: wrmse(tr.base * shipped_multiplier(tr, d), tr.y_k_pct, tr.y_pa))
        )
        ys.append(te.y_k_pct.to_numpy())
        ws.append(te.y_pa.to_numpy())
    y, w = pd.Series(np.concatenate(ys)), np.concatenate(ws)
    for label, parts in chunks.items():
        print(f"  {label:<10} wRMSE {wrmse(np.concatenate(parts), y, w):.5f}  n={len(y)}")
    print(f"  dose picked by the training weeks: median {np.median(doses):.1f}, mean {np.mean(doses):.2f}")

    print("\n===== calibration by deviation bucket =====")
    d = p[p.vfa_dev.notna()].copy()
    d["bucket"] = pd.cut(d.vfa_dev, [-9, -1.0, -0.4, 0.4, 1.0, 9])
    d["priced"] = d.base.to_numpy() * shipped_multiplier(d)
    agg = d.groupby("bucket", observed=True).apply(
        lambda g: pd.Series(
            {
                "starts": len(g),
                "blended": np.average(g.base, weights=g.y_pa),
                "priced": np.average(g.priced, weights=g.y_pa),
                "realised": np.average(g.y_k_pct, weights=g.y_pa),
            }
        )
    )
    print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
