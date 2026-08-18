"""Are the stuff/command priors xK% and xBB% calibrated? Refit them out of sample.

``PitcherRegression.expected_k_pct`` pulls a starter's strikeout rate toward a
stuff-based prior and ``expected_bb_pct`` pulls his walk rate toward a
command-based one, each blended at 150 equivalent PA. Both are the engine's
answer to a thin sample: regress toward what the arm *is* rather than toward the
league. Neither coefficient was ever fitted -- the source says the xK line is
"anchored so a league-average arm maps to ~.220 K%", which fixes the intercept
and says nothing about the slopes.

This fits them. Every starter's start is predicted from pitches thrown strictly
before it, over the same 42-day window the engine uses, and scored on the only
thing that matters -- what he actually did next -- weighted by the plate
appearances he faced, against a chronological holdout and a weekly walk-forward.

    python -m scripts.xk_refit_study

Finding, 2,936 starts by 228 pitchers over 2026: **both priors are roughly three
times too steep, and the strikeout one is worse than using no prior at all.**
Regressing the next start's rate on the prior gives a slope of 0.286 for xK and
0.369 for xBB where a calibrated prior gives 1.0, and the quintiles show what
that costs: the arms xK puts at .100 strike out .181, the ones it puts at .373
strike out .258. It spans 27 points of prediction across 8 points of reality.

Out of sample the engine's xK scores a weighted RMSE of 0.1217 against 0.1083
for the league mean and 0.1027 for the pitcher's own raw 42-day rate -- so a
prior built to rescue a thin sample is beaten by ignoring the pitcher entirely,
and beaten by the very sample it is regularising, which is *less* extreme than
the prior pulling on it. Weekly walk-forward agrees: 0.1188 against 0.1032.
xBB is milder but the same sign, 0.0647 against 0.0621 for the league mean.

The variables are fine; it is only the slopes. Called-strike rate carries +0.34
(t +4.0) on the next start's strikeout rate and *keeps* it with Zone% and
F-strike% in the model, so CSW% is not smuggling command into the strikeout
prior -- a called strike predicts a strikeout on its own account. Measured, the
2.6/1.4 on CSW%/SwStr% is 0.34/0.71, and the 0.50/0.40/0.30 on Zone%/chase/
F-strike% is 0.21/0.09/0.11.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_engine.features.regression import (
    BL_BB_PCT,
    BL_CHASE,
    BL_FSTRIKE,
    BL_ZONE,
    CALLED_OR_WHIFF,
    K_EVENTS_P,
    SWING_DESC,
    WHIFF_DESC,
    XBB_CHASE_COEF,
    XBB_FSTRIKE_COEF,
    XBB_ZONE_COEF,
    XK_CSW_COEF,
    XK_INTERCEPT,
    XK_SWSTR_COEF,
)

CACHE = os.path.expanduser("~/.mlb_engine/cache")
WINDOW_D = 42  # the engine's pitcher_form_days
MIN_PA_START = 12  # in the start being predicted
MIN_PA_WINDOW = 20  # in the look-back, before a prior is even asked for
K_CLIP = (0.08, 0.42)  # the engine's own clips on the two priors
BB_CLIP = (0.02, 0.20)

# Per-start counters, summed over the look-back window.
SUMS = [
    "pitches", "n_csw", "n_whiff", "n_swing", "n_zone", "n_oz_swing",
    "n_first", "n_first_strike", "pa", "n_k", "n_bb", "n_out_zone",
]

SPECS: dict[str, list[str]] = {
    "swstr only": ["swstr"],
    "csw + swstr (engine's basis)": ["csw", "swstr"],
    "called + swstr": ["called", "swstr"],
    "called + swstr + zone + fstrike": ["called", "swstr", "zone", "fstrike"],
    "whiff-per-swing only": ["whiff_swing"],
    "observed K% only": ["k_pct"],
    "observed K% + swstr + called": ["k_pct", "swstr", "called"],
}


def load_pitches() -> pd.DataFrame:
    """Every cached Statcast pitch, stitched into one non-overlapping season.

    The caches are rolling windows that overlap heavily, and the same pitch is
    not byte-identical between two pulls (the batted-ball estimates are revised),
    so de-duplicating on row equality leaves most of the overlap behind and
    double-counts it. Claiming each date once is exact.
    """
    paths = sorted(p for p in os.listdir(CACHE) if p.startswith("statcast_"))
    if not paths:
        raise SystemExit(f"no Statcast caches in {CACHE}")
    frames: list[pd.DataFrame] = []
    seen: set[object] = set()
    for name in paths:
        part = pd.read_pickle(os.path.join(CACHE, name))
        part["game_date"] = pd.to_datetime(part["game_date"])
        fresh = part[~part["game_date"].isin(seen)]
        if fresh.empty:
            continue
        seen.update(fresh["game_date"].unique())
        frames.append(fresh)
    df = pd.concat(frames, ignore_index=True)
    # No game id in the cached columns; a date and the two clubs identify a game.
    df["game_key"] = df["game_date"].dt.strftime("%Y%m%d") + df["home_team"] + df["away_team"]
    return df


def per_start(df: pd.DataFrame) -> pd.DataFrame:
    """One row per starter-start, with the counters the two priors are built on."""
    d = df.copy()
    d["is_pa"] = d["events"].notna() & (d["events"] != "")
    d["k"] = d["is_pa"] & d["events"].isin(K_EVENTS_P)
    d["bb"] = d["is_pa"] & d["events"].isin(["walk", "hit_by_pitch"])
    d["csw"] = d["description"].isin(CALLED_OR_WHIFF)
    d["whiff"] = d["description"].isin(WHIFF_DESC)
    d["swing"] = d["description"].isin(SWING_DESC)
    d["in_zone"] = d["zone"].between(1, 9)
    d["oz_swing"] = d["swing"] & ~d["in_zone"]
    d["first"] = (d["balls"] == 0) & (d["strikes"] == 0)
    d["first_strike"] = d["first"] & (d["type"] != "B")
    d["opened"] = d["inning"] <= 1

    g = d.groupby(["pitcher", "game_key", "game_date"], as_index=False).agg(
        pitches=("csw", "size"),
        n_csw=("csw", "sum"),
        n_whiff=("whiff", "sum"),
        n_swing=("swing", "sum"),
        n_zone=("in_zone", "sum"),
        n_oz_swing=("oz_swing", "sum"),
        n_first=("first", "sum"),
        n_first_strike=("first_strike", "sum"),
        pa=("is_pa", "sum"),
        n_k=("k", "sum"),
        n_bb=("bb", "sum"),
        opened=("opened", "max"),
    )
    g["n_out_zone"] = g["pitches"] - g["n_zone"]
    # A starter is the arm who was on the mound in the first inning; the PA floor
    # drops openers, whose next "start" is not the thing being predicted.
    return g[g["opened"] & (g["pa"] >= MIN_PA_START)].reset_index(drop=True)


def panel(starts: pd.DataFrame, days: int = WINDOW_D) -> pd.DataFrame:
    """Each start, paired with the pitcher's prior `days` of work before it."""
    rows = []
    for pid, g in starts.groupby("pitcher"):
        g = g.sort_values("game_date").reset_index(drop=True)
        dates = g["game_date"].values
        arr = g[SUMS].to_numpy(dtype=float)
        floor = dates - np.timedelta64(days, "D")
        for i in range(len(g)):
            mask = (dates < dates[i]) & (dates >= floor[i])
            if not mask.any():
                continue
            row = {"pitcher": pid, "game_date": g["game_date"].iloc[i]}
            row.update({f"w_{c}": v for c, v in zip(SUMS, arr[mask].sum(axis=0), strict=True)})
            row["y_pa"] = float(g["pa"].iloc[i])
            row["y_k"] = float(g["n_k"].iloc[i])
            row["y_bb"] = float(g["n_bb"].iloc[i])
            rows.append(row)
    p = pd.DataFrame(rows)
    p["csw"] = p.w_n_csw / p.w_pitches
    p["swstr"] = p.w_n_whiff / p.w_pitches
    p["called"] = (p.w_n_csw - p.w_n_whiff) / p.w_pitches
    p["whiff_swing"] = p.w_n_whiff / p.w_n_swing
    p["zone"] = p.w_n_zone / p.w_pitches
    p["chase"] = p.w_n_oz_swing / p.w_n_out_zone
    p["fstrike"] = p.w_n_first_strike / p.w_n_first
    p["k_pct"] = p.w_n_k / p.w_pa
    p["bb_pct"] = p.w_n_bb / p.w_pa
    p["y_k_pct"] = p.y_k / p.y_pa
    p["y_bb_pct"] = p.y_bb / p.y_pa
    p["xk_engine"] = np.clip(
        XK_INTERCEPT + XK_CSW_COEF * p.csw + XK_SWSTR_COEF * p.swstr, *K_CLIP
    )
    p["xbb_engine"] = np.clip(
        BL_BB_PCT
        + XBB_ZONE_COEF * (BL_ZONE - p.zone)
        + XBB_CHASE_COEF * (BL_CHASE - p.chase)
        + XBB_FSTRIKE_COEF * (BL_FSTRIKE - p.fstrike),
        *BB_CLIP,
    )
    return p[p.w_pa >= MIN_PA_WINDOW].sort_values("game_date").reset_index(drop=True)


def wrmse(pred: np.ndarray, truth: pd.Series, weight: pd.Series) -> float:
    return float(np.sqrt(np.average((np.asarray(pred) - truth.to_numpy()) ** 2, weights=weight)))


def fit(train: pd.DataFrame, cols: list[str], target: str):
    return sm.WLS(train[target], sm.add_constant(train[cols], has_constant="add"),
                  weights=train.y_pa).fit()


def apply(res, test: pd.DataFrame, cols: list[str], clip: tuple[float, float]) -> np.ndarray:
    X = sm.add_constant(test[cols], has_constant="add")
    return np.clip(res.predict(X).to_numpy(), *clip)


def calibration(p: pd.DataFrame, prior: str, target: str, label: str) -> None:
    res = sm.WLS(p[target], sm.add_constant(p[[prior]]), weights=p.y_pa).fit()
    print(f"\n{label}: realised next-start rate regressed on the prior")
    print(f"  slope {res.params[prior]:+.3f} (t {res.tvalues[prior]:+.2f}),"
          f" intercept {res.params['const']:+.4f}  -- calibrated is 1.0")
    q = pd.qcut(p[prior], 5, duplicates="drop")
    obs = "k_pct" if target == "y_k_pct" else "bb_pct"
    table = p.groupby(q, observed=True).apply(
        lambda d: pd.Series({
            "n": len(d),
            "prior": np.average(d[prior], weights=d.y_pa),
            "realised": np.average(d[target], weights=d.y_pa),
            "raw 42d rate": np.average(d[obs], weights=d.w_pa),
        }),
        include_groups=False,
    )
    print(table.round(4).to_string())


def holdout(p: pd.DataFrame, cut: pd.Timestamp) -> None:
    train, test = p[p.game_date < cut], p[p.game_date >= cut]
    print(f"\n===== K%: holdout from {cut.date()}  (train {len(train)}, test {len(test)}) =====")
    base = float(np.average(train.y_k_pct, weights=train.y_pa))
    print(f"  {'league mean':<32} wRMSE {wrmse(np.full(len(test), base), test.y_k_pct, test.y_pa):.5f}")
    print(f"  {'engine xK (hand-set coefs)':<32} wRMSE {wrmse(test.xk_engine, test.y_k_pct, test.y_pa):.5f}")
    for name, cols in SPECS.items():
        res = fit(train, cols, "y_k_pct")
        terms = "  ".join(f"{c} {res.params[c]:+.3f} (t {res.tvalues[c]:+.2f})" for c in cols)
        err = wrmse(apply(res, test, cols, K_CLIP), test.y_k_pct, test.y_pa)
        print(f"  {name:<32} wRMSE {err:.5f}   {terms}")


def walk_forward(p: pd.DataFrame, min_train: int = 300) -> None:
    print("\n===== K%: weekly walk-forward (refit each week on everything before it) =====")
    preds: dict[str, list[np.ndarray]] = {k: [] for k in SPECS}
    preds["engine xK (hand-set coefs)"] = []
    preds["league mean"] = []
    tests: list[pd.DataFrame] = []
    for _, test in p.groupby(p.game_date.dt.to_period("W")):
        train = p[p.game_date < test.game_date.min()]
        if len(train) < min_train or test.empty:
            continue
        tests.append(test)
        preds["engine xK (hand-set coefs)"].append(test.xk_engine.to_numpy())
        base = float(np.average(train.y_k_pct, weights=train.y_pa))
        preds["league mean"].append(np.full(len(test), base))
        for name, cols in SPECS.items():
            preds[name].append(apply(fit(train, cols, "y_k_pct"), test, cols, K_CLIP))
    scored = pd.concat(tests)
    for name, chunks in preds.items():
        err = wrmse(np.concatenate(chunks), scored.y_k_pct, scored.y_pa)
        print(f"  {name:<32} wRMSE {err:.5f}  n={len(scored)}")


def refits(p: pd.DataFrame) -> None:
    print("\n===== refit coefficients on every start =====")
    k = fit(p, ["called", "swstr"], "y_k_pct")
    print(f"  xK  = {k.params['const']:+.4f} {k.params['called']:+.4f}*called"
          f" {k.params['swstr']:+.4f}*swstr        R2 {k.rsquared:.4f}")
    print(f"        engine today, same basis: {XK_INTERCEPT:+.4f} {XK_CSW_COEF:+.4f}*called"
          f" {XK_CSW_COEF + XK_SWSTR_COEF:+.4f}*swstr")
    full = fit(p, ["called", "swstr", "zone", "fstrike"], "y_k_pct")
    print("  with the command signals added (does the called-strike term survive?):")
    for c in ("called", "swstr", "zone", "fstrike"):
        print(f"     {c:<9} {full.params[c]:+.4f}  t {full.tvalues[c]:+.2f}  p {full.pvalues[c]:.4f}")

    b = fit(p, ["zone", "chase", "fstrike"], "y_bb_pct")
    print(f"\n  xBB = {b.params['const']:+.4f} {b.params['zone']:+.4f}*zone"
          f" {b.params['chase']:+.4f}*chase {b.params['fstrike']:+.4f}*fstrike   R2 {b.rsquared:.4f}")
    print(f"        engine today: {-XBB_ZONE_COEF:+.4f}*zone {-XBB_CHASE_COEF:+.4f}*chase"
          f" {-XBB_FSTRIKE_COEF:+.4f}*fstrike")


def bb_holdout(p: pd.DataFrame, cut: pd.Timestamp) -> None:
    train, test = p[p.game_date < cut], p[p.game_date >= cut]
    print(f"\n===== BB%: holdout from {cut.date()}  (test {len(test)}) =====")
    base = float(np.average(train.y_bb_pct, weights=train.y_pa))
    print(f"  {'league mean':<32} wRMSE {wrmse(np.full(len(test), base), test.y_bb_pct, test.y_pa):.5f}")
    print(f"  {'engine xBB (hand-set coefs)':<32} wRMSE {wrmse(test.xbb_engine, test.y_bb_pct, test.y_pa):.5f}")
    print(f"  {'raw 42-day BB%':<32} wRMSE {wrmse(test.bb_pct, test.y_bb_pct, test.y_pa):.5f}")
    for name, cols in (("refit zone+chase+fstrike", ["zone", "chase", "fstrike"]),
                       ("refit + observed BB%", ["bb_pct", "zone", "chase", "fstrike"])):
        res = fit(train, cols, "y_bb_pct")
        err = wrmse(apply(res, test, cols, BB_CLIP), test.y_bb_pct, test.y_pa)
        print(f"  {name:<32} wRMSE {err:.5f}")


def main() -> None:
    pitches = load_pitches()
    starts = per_start(pitches)
    p = panel(starts)
    print(f"{len(pitches):,} pitches -> {len(starts):,} starter-starts -> {len(p):,} predictable, "
          f"{p.pitcher.nunique()} pitchers, {p.game_date.min().date()} to {p.game_date.max().date()}")
    calibration(p, "xk_engine", "y_k_pct", "xK%")
    calibration(p, "xbb_engine", "y_bb_pct", "xBB%")
    cut = p.game_date.quantile(0.65)
    holdout(p, cut)
    walk_forward(p)
    bb_holdout(p, cut)
    refits(p)


if __name__ == "__main__":
    main()
