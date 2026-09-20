"""Should the contact-quality home-run multiplier exist on top of xHR?

The home-run rate the simulator uses is a hitter's observed HR/PA blended toward
``xHR/PA`` at 200 equivalent PA -- and then multiplied by the HR term of
:meth:`BatterRegression.multipliers`, which is built on barrel rate, bat speed,
max exit velocity on air contact and pulled-air share, with brakes for soft air
contact, ground balls and pop-ups.

Every one of those inputs is a property of the same batted balls ``xHR`` scores.
xHR asks whether each ball cleared the wall it was hit toward; the multiplier
asks whether the balls were hard and in the air. So contact quality is priced
twice, once inside the prior and once on top of it -- the shape #195 found on the
strikeout side, where CSW% sat inside xK% and was then multiplied back in.

The panel is the batter analogue of the one #185 and #195 used: every
batter-game predicted from that hitter's own pitches over the engine's 42-day
window, strictly before the game, weighted by the plate appearances he took in
it.

    python -m scripts.hr_multiplier_study

Run it before trusting the module docstring: the numbers below are from the
2026 caches on this machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_engine.features.regression import (
    BL_BARREL,
    BL_BAT_SPEED,
    BL_MAX_EV,
    BL_PULL_AIR,
    FB_LD_EV_FLOOR,
    GB_RATE_CEILING,
    IFFB_CEILING,
    build_batter_regression,
)
from mlb_engine.features.rolling import HR_PRIOR_WEIGHT, blend_hr_rate, rates_from_events
from mlb_engine.features.xhr import batter_xhr
from scripts.xk_refit_study import load_pitches, wrmse

WINDOW_D = 42
MIN_PA_WINDOW = 60  # the engine's own floor for reading a bat
MIN_PA_GAME = 3
PANEL_CACHE = Path.home() / ".mlb_engine" / "cache" / "hr_multiplier_panel.pkl"

# The shipped HR multiplier, term by term, so each piece can be switched off.
# (attribute, baseline, slope, clip)
TERMS: dict[str, tuple[str, float, float, tuple[float, float]]] = {
    "barrel": ("barrel_rate", BL_BARREL, 2.5, (-0.12, 0.15)),
    "bat_speed": ("bat_speed", BL_BAT_SPEED, 0.010, (-0.06, 0.06)),
    "air_max_ev": ("air_max_ev", BL_MAX_EV, 0.009, (-0.06, 0.09)),
    "pull_air": ("pull_air_pct", BL_PULL_AIR, 0.50, (-0.05, 0.08)),
}
PRODUCT_CLIP = (0.50, 1.32)


def per_game(df: pd.DataFrame) -> pd.DataFrame:
    """Index of batter-games, so windows can be cut without re-scanning pitches."""
    d = df[df["batter"].notna()].copy()
    d["is_pa"] = d["events"].notna() & (d["events"] != "")
    d["hr"] = d["is_pa"] & d["events"].eq("home_run")
    g = d.groupby(["batter", "game_key", "game_date"], as_index=False).agg(
        pa=("is_pa", "sum"), n_hr=("hr", "sum")
    )
    return g[g["pa"] >= MIN_PA_GAME].reset_index(drop=True)


def panel(pitches: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each batter-game paired with the 42-day window the engine would have read.

    The window features come from :func:`build_batter_regression` and
    :func:`batter_xhr` on the real pitch slice rather than reconstructed
    counters, so the multiplier under test is byte-for-byte the shipped one.
    """
    rows: list[dict[str, float]] = []
    by_batter = {int(b): g for b, g in pitches.groupby("batter")}
    for i, (bid, gp) in enumerate(games.groupby("batter")):
        slice_all = by_batter.get(int(bid))
        if slice_all is None:
            continue
        dates = slice_all["game_date"].to_numpy()
        gp = gp.sort_values("game_date")
        for row in gp.itertuples():
            day = np.datetime64(row.game_date)
            mask = (dates < day) & (dates >= day - np.timedelta64(WINDOW_D, "D"))
            if not mask.any():
                continue
            win = slice_all[mask]
            ev = win["events"].dropna()
            if len(ev) < MIN_PA_WINDOW:
                continue
            reg = build_batter_regression(win)
            prof = batter_xhr(win)
            xhr = prof.xhr_per_pa
            if xhr != xhr:  # no distance data: the blend is a no-op anyway
                continue
            base = blend_hr_rate(rates_from_events(ev), xhr, HR_PRIOR_WEIGHT).p_hr
            mult = reg.multipliers().get("HR", 1.0)
            rows.append({
                "batter": int(bid),
                "game_date": row.game_date,
                "pa_window": float(len(ev)),
                "hr_window": float(ev.eq("home_run").sum()),
                "xhr_per_pa": float(xhr),
                "base": float(base),
                "m_shipped": float(mult),
                "barrel": float(reg.barrel_rate),
                "bat_speed": float(reg.bat_speed),
                "air_max_ev": float(reg.air_max_ev),
                "pull_air": float(reg.pull_air_pct),
                "air_hard_hit": float(reg.air_hard_hit),
                "fb_ld_ev": float(reg.fb_ld_ev),
                "gb_rate": float(reg.gb_rate),
                "iffb_pct": float(reg.iffb_pct),
                "bbe": float(reg.bbe),
                "y_pa": float(row.pa),
                "y_hr": float(row.n_hr),
            })
        if i % 50 == 0:
            print(f"  ... {i} batters, {len(rows):,} rows", flush=True)
    p = pd.DataFrame(rows).dropna(subset=["base", "m_shipped"])
    p["obs_hr_pa"] = p.hr_window / p.pa_window
    p["y_hr_pa"] = p.y_hr / p.y_pa
    return p.sort_values("game_date").reset_index(drop=True)


def shipped_multiplier(
    p: pd.DataFrame, terms: tuple[str, ...], *, brakes: bool = True
) -> np.ndarray:
    """The shipped HR multiplier rebuilt from ``terms``, brakes optional.

    The contact-quality terms lift a hitter (PPV); the brakes describe contact
    that cannot leave the park at all (NPV). They are separable, and the point
    of the study is that they need not stand or fall together.
    """
    m = np.ones(len(p))
    for name in terms:
        _attr, baseline, slope, clip = TERMS[name]
        m = m * (1.0 + np.clip((p[name].to_numpy() - baseline) * slope, *clip))
    if brakes:
        m = m * brake_factor(p)
    return np.clip(m, *PRODUCT_CLIP)


def brake_factor(p: pd.DataFrame) -> np.ndarray:
    """The NPV half of the shipped multiplier: contact that cannot go out."""
    m: np.ndarray = np.ones(len(p))
    m = np.where(p.air_hard_hit.to_numpy() < 0.30, m * 0.80, m)
    ev = p.fb_ld_ev.to_numpy()
    m = np.where(
        (ev == ev) & (ev < FB_LD_EV_FLOOR),
        m * np.clip(1.0 - (FB_LD_EV_FLOOR - ev) * 0.06, 0.70, 1.0),
        m,
    )
    gb = p.gb_rate.to_numpy()
    m = np.where(
        gb > GB_RATE_CEILING,
        m * np.clip(1.0 - (gb - GB_RATE_CEILING) * 2.0, 0.80, 1.0),
        m,
    )
    iffb = p.iffb_pct.to_numpy()
    return np.where(
        (iffb == iffb) & (iffb > IFFB_CEILING),
        m * np.clip(1.0 - (iffb - IFFB_CEILING) * 1.5, 0.85, 1.0),
        m,
    )


def fitted_multiplier(train: pd.DataFrame, test: pd.DataFrame):
    """Fit the multiplier as what it claims to be: a correction to the blend."""
    cols = list(TERMS)
    y = np.log(np.clip(train.y_hr_pa.to_numpy(), 0.005, None) / train.base.to_numpy())
    X = sm.add_constant(pd.DataFrame({c: train[c] - TERMS[c][1] for c in cols}))
    res = sm.WLS(y, X, weights=train.y_pa).fit()
    Xt = sm.add_constant(pd.DataFrame({c: test[c] - TERMS[c][1] for c in cols}))
    return res, np.exp(res.predict(Xt).to_numpy()) * test.base.to_numpy()


def build() -> pd.DataFrame:
    if PANEL_CACHE.exists() and not os.environ.get("HRSTUDY_REBUILD"):
        return pd.read_pickle(PANEL_CACHE)
    pitches = load_pitches()
    games = per_game(pitches)
    print(f"{len(games):,} batter-games; building the 42-day windows", flush=True)
    p = panel(pitches, games)
    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    p.to_pickle(PANEL_CACHE)
    return p


def main() -> None:
    p = build()
    print(f"\n{len(p):,} batter-games by {p.batter.nunique()} hitters, "
          f"{p.game_date.min().date()}..{p.game_date.max().date()}")
    print(f"observed HR/PA {np.average(p.obs_hr_pa, weights=p.pa_window):.4f}   "
          f"xHR/PA {np.average(p.xhr_per_pa, weights=p.pa_window):.4f}   "
          f"blended {np.average(p.base, weights=p.y_pa):.4f}   "
          f"priced {np.average(p.base * p.m_shipped, weights=p.y_pa):.4f}   "
          f"realised {np.average(p.y_hr_pa, weights=p.y_pa):.4f}")

    cut = p.game_date.quantile(0.6)
    train, test = p[p.game_date < cut], p[p.game_date >= cut]
    print(f"\n===== next game's HR/PA: holdout from {cut.date()} "
          f"(train {len(train):,}, test {len(test):,}) =====")
    print(f"  {'blended rate alone (no multiplier)':<44} "
          f"wRMSE {wrmse(test.base, test.y_hr_pa, test.y_pa):.5f}")
    print(f"  {'blended rate x shipped HR multiplier':<44} "
          f"wRMSE {wrmse(test.base * test.m_shipped, test.y_hr_pa, test.y_pa):.5f}")
    print(f"  {'observed HR/PA alone (no blend at all)':<44} "
          f"wRMSE {wrmse(test.obs_hr_pa, test.y_hr_pa, test.y_pa):.5f}")
    print(f"  {'xHR/PA alone':<44} "
          f"wRMSE {wrmse(test.xhr_per_pa, test.y_hr_pa, test.y_pa):.5f}")
    print(f"  {'blended rate x the NPV brakes alone':<44} "
          f"wRMSE {wrmse(test.base * brake_factor(test), test.y_hr_pa, test.y_pa):.5f}")
    ppv_only = test.base.to_numpy() * shipped_multiplier(test, tuple(TERMS), brakes=False)
    print(f"  {'blended rate x the PPV terms alone':<44} "
          f"wRMSE {wrmse(ppv_only, test.y_hr_pa, test.y_pa):.5f}")
    for name in TERMS:
        others = tuple(c for c in TERMS if c != name)
        pred = test.base.to_numpy() * shipped_multiplier(test, others)
        print(f"  {'  ... whole multiplier without ' + name:<44} "
              f"wRMSE {wrmse(pred, test.y_hr_pa, test.y_pa):.5f}")
    doses = np.linspace(0.0, 1.0, 21)
    best = min(
        doses,
        key=lambda d: wrmse(train.base * train.m_shipped**d, train.y_hr_pa, train.y_pa),
    )
    label = f"blended rate x multiplier^{best:.2f} (dose on train)"
    print(f"  {label:<44} "
          f"wRMSE {wrmse(test.base * test.m_shipped**best, test.y_hr_pa, test.y_pa):.5f}")
    res, pred = fitted_multiplier(train, test)
    print(f"  {'blended rate x fitted multiplier':<44} "
          f"wRMSE {wrmse(pred, test.y_hr_pa, test.y_pa):.5f}")

    print("\n===== the fitted multiplier, in the shipped parameterisation =====")
    print(f"  {'term':<14}{'shipped':>10}{'fitted':>10}{'t':>8}")
    for name in TERMS:
        print(f"  {name:<14}{TERMS[name][2]:>10.3f}"
              f"{res.params[name]:>10.3f}{res.tvalues[name]:>8.2f}")
    print(f"  {'intercept(log)':<14}{'--':>10}{res.params['const']:>10.3f}"
          f"{res.tvalues['const']:>8.2f}")

    print("\n===== weekly walk-forward (fit on every prior week) =====")
    weeks = p.game_date.dt.to_period("W")
    preds: dict[str, list[np.ndarray]] = {"none": [], "shipped": [], "fitted": [], "dose": []}
    truth, wts, doses_picked = [], [], []
    for wk in sorted(weeks.unique())[1:]:
        tr, te = p[weeks < wk], p[weeks == wk]
        if len(tr) < 300 or te.empty:
            continue
        _, fit_pred = fitted_multiplier(tr, te)
        preds["none"].append(te.base.to_numpy())
        preds["shipped"].append((te.base * te.m_shipped).to_numpy())
        preds["fitted"].append(fit_pred)
        doses = np.linspace(0.0, 1.0, 21)
        best = min(
            doses,
            key=lambda d: wrmse(tr.base * tr.m_shipped**d, tr.y_hr_pa, tr.y_pa),
        )
        doses_picked.append(best)
        preds["dose"].append((te.base * te.m_shipped**best).to_numpy())
        truth.append(te.y_hr_pa.to_numpy())
        wts.append(te.y_pa.to_numpy())
    if not truth:
        print("  too few weeks in the panel for a walk-forward")
        return
    y = pd.Series(np.concatenate(truth))
    w = pd.Series(np.concatenate(wts))
    for label, chunks in preds.items():
        print(f"  {label:<8} wRMSE {wrmse(np.concatenate(chunks), y, w):.5f}  n={len(y):,}")
    print(f"  dose picked on the training weeks: median {np.median(doses_picked):.2f}, "
          f"range {min(doses_picked):.2f}..{max(doses_picked):.2f}")

    print("\n===== is the multiplier calibrated? realised / blended, regressed on it =====")
    ratio = p.y_hr_pa / p.base.clip(lower=1e-6)
    r = sm.WLS(ratio, sm.add_constant(p.m_shipped.rename("m")), weights=p.y_pa).fit()
    print(f"  shipped  slope {r.params['m']:+.3f} (t {r.tvalues['m']:+.2f})"
          f"   -- a calibrated multiplier gives 1.0")
    on_clip = float(np.mean((p.m_shipped <= PRODUCT_CLIP[0] + 1e-9)
                            | (p.m_shipped >= PRODUCT_CLIP[1] - 1e-9)))
    qs = np.quantile(p.m_shipped, [0.0, 0.01, 0.5, 0.99, 1.0])
    print(f"  shipped  min {qs[0]:.3f}  p1 {qs[1]:.3f}  median {qs[2]:.3f}"
          f"  p99 {qs[3]:.3f}  max {qs[4]:.3f}   on the {PRODUCT_CLIP} clip: {on_clip:.2%}")

    q = pd.qcut(p.m_shipped, 5, duplicates="drop")
    table = p.groupby(q, observed=True).apply(
        lambda d: pd.Series({
            "n": len(d),
            "multiplier": np.average(d.m_shipped, weights=d.y_pa),
            "blended": np.average(d.base, weights=d.y_pa),
            "priced": np.average(d.base * d.m_shipped, weights=d.y_pa),
            "realised": np.average(d.y_hr_pa, weights=d.y_pa),
        }),
        include_groups=False,
    )
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
