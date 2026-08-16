"""Should ``k_multiplier`` exist for a starter, and where do its clips belong?

The strikeout rate the simulator uses is the pitcher's observed rate blended
toward xK% at 150 equivalent PA -- and then multiplied by
``PitcherRegression.k_multiplier``, which is itself built on CSW%, K-BB% and the
two-strike put-away rate. CSW% is therefore priced twice, once inside the prior
and once on top of it, and the multiplier's bounds (0.75, 1.30) were never
measured against anything.

#190 showed the bullpen half of that multiplier carries no weight once the pen's
own strikeout rate is in the model. This asks the same question of a starter,
and asks where the clips belong, on the same panel #185 used: every start
predicted from pitches thrown strictly before it over the engine's 42-day
window, weighted by the plate appearances he faced.

    python -m scripts.k_multiplier_study

Finding, 2,777 starts by 222 pitchers over 2026: **the multiplier is worse than
not having one, and the rate it multiplies is already right.** Bucket the starts
by the multiplier the engine would apply and the blended rate lands within a
point of what the arm went on to do in every bucket, while the priced rate is
off by three and a half:

    multiplier   blended   priced   realised
      0.79        .1856    .1473     .1811
      0.88        .2065    .1828     .2021
      0.97        .2218    .2160     .2133
      1.08        .2381    .2572     .2346
      1.24        .2674    .3321     .2658

It is not miscalibrated, it is unnecessary: a multiplier applied to a calibrated
rate can only stretch it. Weekly walk-forward wRMSE is 0.09763 with no
multiplier, 0.10400 with the shipped one and 0.10122 with the terms refitted;
dropping any single term improves the shipped version; and a dose search over
the exponent -- how much of the multiplier to keep -- picks **0.0**. The fitted
version is nearly flat (0.80..0.94 across the whole league against the shipped
0.75..1.30, where 10% of starts sit on a clip), and its slopes are 0.76/0.03/
-0.49 against the shipped 2.5/1.5/0.8, none of them significant.

The reason is double-counting: CSW% and SwStr% are already inside xK%, which is
blended into the strikeout rate at 150 PA precisely to shrink a thin sample, and
the multiplier then re-inflates the spread that blend just removed. #190 found
the same thing on the bullpen, where the pen's pooled K% beat its stuff outright.

Caveat: ``engine_k_rate`` reconstructs the pre-multiplier rate from K/BB/out
counts rather than running the pipeline, so its level carries a small
over-prediction (the fitted intercept, -0.13 in logs) that a real slate does not
necessarily share. Nothing here rests on that level -- the ranking of the four
predictors and the dose of 0.0 are unchanged if the intercept is fitted away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_engine.features.regression import (
    BL_CSW,
    BL_K_MINUS_BB,
    BL_TWO_STRIKE_WHIFF,
    K_EVENTS_P,
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

# The shipped multiplier, term by term, so each piece can be switched off.
SHIPPED_TERMS = {
    "csw": (BL_CSW, 2.5, (-0.15, 0.20)),
    "k_minus_bb": (BL_K_MINUS_BB, 1.5, (-0.12, 0.15)),
    "two_strike_whiff": (BL_TWO_STRIKE_WHIFF, 0.8, (-0.06, 0.08)),
}
PRODUCT_CLIP = (0.75, 1.30)
MIN_PITCHES = 100  # the multiplier's own floor


def per_start(df: pd.DataFrame) -> pd.DataFrame:
    """One row per starter-start, carrying the multiplier's three inputs."""
    d = df.copy()
    d["is_pa"] = d["events"].notna() & (d["events"] != "")
    d["k"] = d["is_pa"] & d["events"].isin(K_EVENTS_P)
    d["bb"] = d["is_pa"] & d["events"].isin(["walk", "hit_by_pitch"])
    d["csw"] = d["description"].isin(["called_strike", *WHIFF_DESC])
    d["whiff"] = d["description"].isin(WHIFF_DESC)
    d["two"] = d["strikes"] == 2
    d["two_whiff"] = d["two"] & d["whiff"]
    d["opened"] = d["inning"] <= 1
    g = d.groupby(["pitcher", "game_key", "game_date"], as_index=False).agg(
        pitches=("csw", "size"),
        n_csw=("csw", "sum"),
        n_whiff=("whiff", "sum"),
        n_two=("two", "sum"),
        n_two_whiff=("two_whiff", "sum"),
        pa=("is_pa", "sum"),
        n_k=("k", "sum"),
        n_bb=("bb", "sum"),
        opened=("opened", "max"),
    )
    return g[g["opened"] & (g["pa"] >= MIN_PA_START)].reset_index(drop=True)


SUMS = ["pitches", "n_csw", "n_whiff", "n_two", "n_two_whiff", "pa", "n_k", "n_bb"]


def panel(starts: pd.DataFrame, days: int = WINDOW_D) -> pd.DataFrame:
    """Each start paired with the window the engine would have read before it."""
    rows = []
    for pid, g in starts.groupby("pitcher"):
        g = g.sort_values("game_date").reset_index(drop=True)
        dates = g["game_date"].to_numpy()
        arr = g[SUMS].to_numpy(dtype=float)
        floor = dates - np.timedelta64(days, "D")
        for i in range(len(g)):
            mask = (dates < dates[i]) & (dates >= floor[i])
            if not mask.any():
                continue
            row = {"pitcher": pid, "game_date": g["game_date"].iloc[i]}
            row.update({c: v for c, v in zip(SUMS, arr[mask].sum(axis=0), strict=True)})
            row["y_pa"] = float(g["pa"].iloc[i])
            row["y_k"] = float(g["n_k"].iloc[i])
            rows.append(row)
    p = pd.DataFrame(rows)
    p["csw"] = p.n_csw / p.pitches
    p["swstr"] = p.n_whiff / p.pitches
    p["two_strike_whiff"] = np.where(p.n_two > 0, p.n_two_whiff / p.n_two, BL_TWO_STRIKE_WHIFF)
    p["k_pct"] = p.n_k / p.pa
    p["bb_pct"] = p.n_bb / p.pa
    p["k_minus_bb"] = p.k_pct - p.bb_pct
    p["y_k_pct"] = p.y_k / p.y_pa
    p = p[(p.pa >= MIN_PA_WINDOW) & (p.pitches >= MIN_PITCHES)]
    return p.sort_values("game_date").reset_index(drop=True)


def engine_k_rate(p: pd.DataFrame) -> pd.Series:
    """The rate the simulator holds *before* the multiplier is applied.

    Observed window K% blended toward xK% at 150 equivalent PA, exactly as
    ``blend_k_rate`` does it on a starter's outcome vector.
    """
    out = []
    for row in p.itertuples():
        events = pd.Series(
            ["strikeout"] * int(row.n_k)
            + ["walk"] * int(row.n_bb)
            + ["field_out"] * int(row.pa - row.n_k - row.n_bb)
        )
        rates = rates_from_events(events)
        xk = float(np.clip(0.220 + 0.34 * (row.csw - BL_CSW) + 0.71 * (row.swstr - 0.110), *K_CLIP))
        out.append(blend_k_rate(rates, xk).p_k)
    return pd.Series(out, index=p.index)


def shipped_multiplier(p: pd.DataFrame, terms: tuple[str, ...]) -> np.ndarray:
    m = np.ones(len(p))
    for name in terms:
        baseline, slope, clip = SHIPPED_TERMS[name]
        m = m * (1.0 + np.clip((p[name] - baseline) * slope, *clip))
    return np.clip(m, *PRODUCT_CLIP)


def fitted_multiplier(train: pd.DataFrame, test: pd.DataFrame, base_train, base_test):
    """Fit the multiplier as what it claims to be: a correction to the blended rate."""
    cols = list(SHIPPED_TERMS)
    y = np.log(np.clip(train.y_k_pct.to_numpy(), 0.02, None) / base_train.to_numpy())
    X = sm.add_constant(pd.DataFrame({c: train[c] - SHIPPED_TERMS[c][0] for c in cols}))
    res = sm.WLS(y, X, weights=train.y_pa).fit()
    Xt = sm.add_constant(pd.DataFrame({c: test[c] - SHIPPED_TERMS[c][0] for c in cols}))
    return res, np.exp(res.predict(Xt).to_numpy()) * base_test.to_numpy()


def main() -> None:
    pitches = load_pitches()
    starts = per_start(pitches)
    p = panel(starts)
    p["base"] = engine_k_rate(p)
    p["m_shipped"] = shipped_multiplier(p, tuple(SHIPPED_TERMS))
    print(f"{len(p):,} starts by {p.pitcher.nunique()} pitchers, "
          f"{p.game_date.min().date()}..{p.game_date.max().date()}")

    cut = p.game_date.quantile(0.6)
    train, test = p[p.game_date < cut], p[p.game_date >= cut]
    print(f"\n===== next start's K%: holdout from {cut.date()} "
          f"(train {len(train)}, test {len(test)}) =====")
    print(f"  {'blended rate alone (no multiplier)':<44} "
          f"wRMSE {wrmse(test.base, test.y_k_pct, test.y_pa):.5f}")
    print(f"  {'blended rate x shipped k_multiplier':<44} "
          f"wRMSE {wrmse(test.base * test.m_shipped, test.y_k_pct, test.y_pa):.5f}")
    for name in SHIPPED_TERMS:
        others = tuple(c for c in SHIPPED_TERMS if c != name)
        pred = test.base * shipped_multiplier(test, others)
        print(f"  {'  ... without the ' + name + ' term':<44} "
              f"wRMSE {wrmse(pred, test.y_k_pct, test.y_pa):.5f}")
    res, pred = fitted_multiplier(train, test, train.base, test.base)
    print(f"  {'blended rate x fitted multiplier':<44} "
          f"wRMSE {wrmse(pred, test.y_k_pct, test.y_pa):.5f}")

    print("\n===== the fitted multiplier, in the shipped parameterisation =====")
    print(f"  {'term':<20}{'shipped':>10}{'fitted':>10}{'t':>8}")
    for name in SHIPPED_TERMS:
        print(f"  {name:<20}{SHIPPED_TERMS[name][1]:>10.2f}"
              f"{res.params[name]:>10.2f}{res.tvalues[name]:>8.2f}")
    print(f"  {'intercept (log)':<20}{'--':>10}{res.params['const']:>10.3f}"
          f"{res.tvalues['const']:>8.2f}")

    full = np.exp(
        res.predict(sm.add_constant(pd.DataFrame(
            {c: p[c] - SHIPPED_TERMS[c][0] for c in SHIPPED_TERMS}
        )))
    ).to_numpy()
    print("\n===== where the clips belong =====")
    for label, m in (("shipped", p.m_shipped.to_numpy()), ("fitted", full)):
        qs = np.quantile(m, [0.0, 0.01, 0.5, 0.99, 1.0])
        on_clip = float(np.mean((p.m_shipped <= PRODUCT_CLIP[0] + 1e-9)
                                | (p.m_shipped >= PRODUCT_CLIP[1] - 1e-9))) if label == "shipped" else 0.0
        print(f"  {label:<8} min {qs[0]:.3f}  p1 {qs[1]:.3f}  median {qs[2]:.3f}"
              f"  p99 {qs[3]:.3f}  max {qs[4]:.3f}"
              + (f"   on the {PRODUCT_CLIP} clip: {on_clip:.2%}" if label == "shipped" else ""))

    print("\n===== weekly walk-forward (fit on every prior week) =====")
    weeks = p.game_date.dt.to_period("W")
    preds: dict[str, list[np.ndarray]] = {"none": [], "shipped": [], "fitted": [], "dose": []}
    truth, wts = [], []
    for wk in sorted(weeks.unique())[4:]:
        tr, te = p[weeks < wk], p[weeks == wk]
        if len(tr) < 200 or te.empty:
            continue
        _, fit_pred = fitted_multiplier(tr, te, tr.base, te.base)
        preds["none"].append(te.base.to_numpy())
        preds["shipped"].append((te.base * te.m_shipped).to_numpy())
        preds["fitted"].append(fit_pred)
        # Is any dose of the shipped multiplier better than none? Shrink it
        # toward 1.0 by the exponent that minimises error on the training weeks.
        doses = np.linspace(0.0, 1.0, 21)
        best = min(doses, key=lambda d: wrmse(tr.base * tr.m_shipped**d, tr.y_k_pct, tr.y_pa))
        preds["dose"].append((te.base * te.m_shipped**best).to_numpy())
        truth.append(te.y_k_pct.to_numpy())
        wts.append(te.y_pa.to_numpy())
    y = pd.Series(np.concatenate(truth))
    w = pd.Series(np.concatenate(wts))
    for label, chunks in preds.items():
        print(f"  {label:<10} wRMSE {wrmse(np.concatenate(chunks), y, w):.5f}  n={len(y)}")

    print("\n===== is the multiplier calibrated? realised / blended, regressed on it =====")
    ratio = p.y_k_pct / p.base
    for label, m in (("shipped", p.m_shipped), ("fitted", pd.Series(full, index=p.index))):
        r = sm.WLS(ratio, sm.add_constant(m.rename("m")), weights=p.y_pa).fit()
        print(f"  {label:<8} slope {r.params['m']:+.3f} (t {r.tvalues['m']:+.2f})"
              f"   -- a calibrated multiplier gives 1.0")
    q = pd.qcut(p.m_shipped, 5, duplicates="drop")
    table = p.groupby(q, observed=True).apply(
        lambda d: pd.Series({
            "n": len(d),
            "multiplier": np.average(d.m_shipped, weights=d.y_pa),
            "blended": np.average(d.base, weights=d.y_pa),
            "priced": np.average(d.base * d.m_shipped, weights=d.y_pa),
            "realised": np.average(d.y_k_pct, weights=d.y_pa),
        }),
        include_groups=False,
    )
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
