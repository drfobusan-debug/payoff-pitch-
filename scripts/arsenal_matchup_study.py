"""Does the arsenal-matching layer predict the matchup it prices?

The engine multiplies every hitter's outcome vector by
:func:`arsenal_matchup_multiplier` -- the pitcher's per-class SwStr% and usage
against the hitter's per-class whiff and xwOBA-on-contact -- worth +-15% on
strikeouts and +-10% on singles, doubles and home runs. It has never been scored
out of sample.

This asks the only question that matters for it: taking the log5 matchup vector
as the baseline, does multiplying by the arsenal term move a *realised* plate
appearance in the direction it claims? The panel is batter-vs-starter, one row
per (game, batter, pitcher) pair, features from the 42-day windows before the
game and outcomes from the plate appearances in it -- so the term is scored on
exactly the matchup it is applied to.

Reported per outcome (K, 1B, H, HR):

* weighted RMSE of the baseline against baseline x the shipped term, on a
  chronological holdout;
* the dose that the training half prefers on the exponent, tested out of sample;
* the gain the training half would fit, against the shipped 0.5;
* and the separation table -- among the matchups the term calls good and bad,
  what actually happened -- which is the true-positive/false-positive reading:
  a term with signal makes the boosted group beat the suppressed group by at
  least as much as it claims.

Run with ``python -m scripts.arsenal_matchup_study`` (``ARSENAL_REBUILD=1`` to
rebuild the panel).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_engine.features.pitch_mix import (
    _K_CLIP,
    _K_GAIN,
    LEAGUE_SWSTR,
    LEAGUE_WHIFF,
    LEAGUE_XWOBA,
    ArsenalProfile,
    BatterPitchProfile,
    build_arsenal,
    build_batter_pitch_profile,
)
from mlb_engine.features.rolling import rates_from_events
from mlb_engine.models.matchup import apply_multipliers, combine
from scripts.xk_refit_study import load_pitches, wrmse

# The terms as they were shipped before this study, reconstructed here so the
# comparison stays reproducible after production changed: gain 0.5 on both
# sides, +-15% on strikeouts and +-10% on contact.
SHIPPED_K_GAIN = 0.5
SHIPPED_K_CLIP = (0.85, 1.15)
SHIPPED_HIT_GAIN = 0.5
SHIPPED_HIT_CLIP = (0.90, 1.10)

WINDOW_D = 42
MIN_PA_BATTER = 60  # the engine's own floor for reading a bat
MIN_PA_PITCHER = 80
MIN_PA_MATCHUP = 2  # a one-PA matchup is a coin, not a sample
PANEL_CACHE = Path.home() / ".mlb_engine" / "cache" / "arsenal_panel.pkl"


def raw_terms(
    arsenal: ArsenalProfile, batter: BatterPitchProfile
) -> tuple[float, float]:
    """The two usage-weighted factors, before any gain or clip.

    Kept raw so a different gain can be scored honestly: 44% of matchups sit on
    the shipped strikeout clip, and rescaling a clipped number is not the same
    as clipping a rescaled one.
    """
    total = sum(arsenal.usage.values())
    if total <= 0:
        return float("nan"), float("nan")
    k_factor = hit_factor = 0.0
    for cls, use in arsenal.usage.items():
        w = use / total
        sw, bw = arsenal.swstr.get(cls), batter.whiff.get(cls)
        if sw is not None and bw is not None:
            k_factor += w * ((sw / LEAGUE_SWSTR[cls]) * (bw / LEAGUE_WHIFF[cls]) - 1.0)
        bx = batter.xwoba.get(cls)
        if bx is not None:
            hit_factor += w * (bx / LEAGUE_XWOBA[cls] - 1.0)
    return k_factor, hit_factor


def term(factor: np.ndarray | float, gain: float, clip: tuple[float, float]):
    return np.clip(1.0 + gain * np.asarray(factor), *clip)


def pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance, with the batter, the pitcher and the game."""
    d = df[df["events"].notna() & (df["events"] != "")].copy()
    d = d[d["batter"].notna() & d["pitcher"].notna()]
    d["batter"] = d["batter"].astype(int)
    d["pitcher"] = d["pitcher"].astype(int)
    return d[["game_date", "game_key", "batter", "pitcher", "events"]]


def starters(pas: pd.DataFrame) -> pd.DataFrame:
    """The pitcher who faced the most batters in each half of each game.

    The arsenal term is applied to the starter matchup, and a starter is the arm
    with the plate appearances -- no lineup card needed.
    """
    n = pas.groupby(["game_key", "pitcher"], as_index=False).size()
    n = n.sort_values("size", ascending=False)
    return n.groupby("game_key").head(2)[["game_key", "pitcher"]]


def matchup_outcomes(pas: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Per (game, batter, starter): plate appearances and what came of them."""
    d = pas.merge(sp, on=["game_key", "pitcher"], how="inner")
    d["is_k"] = d.events.isin(("strikeout", "strikeout_double_play"))
    d["is_1b"] = d.events.eq("single")
    d["is_h"] = d.events.isin(("single", "double", "triple", "home_run"))
    d["is_hr"] = d.events.eq("home_run")
    g = d.groupby(["game_key", "game_date", "batter", "pitcher"], as_index=False).agg(
        y_pa=("events", "size"),
        n_k=("is_k", "sum"),
        n_1b=("is_1b", "sum"),
        n_h=("is_h", "sum"),
        n_hr=("is_hr", "sum"),
    )
    return g[g.y_pa >= MIN_PA_MATCHUP].reset_index(drop=True)


def panel(pitches: pd.DataFrame, pas: pd.DataFrame, mus: pd.DataFrame) -> pd.DataFrame:
    """Each matchup paired with the two 42-day windows the engine would read."""
    by_batter = {int(b): g for b, g in pitches.groupby("batter")}
    by_pitcher = {int(p): g for p, g in pitches.groupby("pitcher")}
    pa_by_batter = {int(b): g for b, g in pas.groupby("batter")}
    pa_by_pitcher = {int(p): g for p, g in pas.groupby("pitcher")}
    rows: list[dict[str, float]] = []

    def window(frame: pd.DataFrame | None, day: np.datetime64) -> pd.DataFrame | None:
        if frame is None:
            return None
        dates = frame["game_date"].to_numpy()
        mask = (dates < day) & (dates >= day - np.timedelta64(WINDOW_D, "D"))
        return frame[mask] if mask.any() else None

    for i, row in enumerate(mus.itertuples()):
        day = np.datetime64(row.game_date)
        bpa = window(pa_by_batter.get(row.batter), day)
        ppa = window(pa_by_pitcher.get(row.pitcher), day)
        if bpa is None or ppa is None:
            continue
        if len(bpa) < MIN_PA_BATTER or len(ppa) < MIN_PA_PITCHER:
            continue
        bwin = window(by_batter.get(row.batter), day)
        pwin = window(by_pitcher.get(row.pitcher), day)
        if bwin is None or pwin is None:
            continue

        base = combine(rates_from_events(bpa.events), rates_from_events(ppa.events))
        f_k, f_hit = raw_terms(build_arsenal(pwin), build_batter_pitch_profile(bwin))
        if f_k != f_k:
            continue
        m_k = float(term(f_k, SHIPPED_K_GAIN, SHIPPED_K_CLIP))
        m_hit = float(term(f_hit, SHIPPED_HIT_GAIN, SHIPPED_HIT_CLIP))
        mult = {"K": m_k, "1B": m_hit, "2B": m_hit, "HR": m_hit}
        priced = apply_multipliers(base, mult)
        rows.append({
            "game_date": row.game_date,
            "batter": row.batter,
            "pitcher": row.pitcher,
            "p_k": base["K"],
            "p_1b": base["1B"],
            "p_h": base["1B"] + base["2B"] + base["3B"] + base["HR"],
            "p_hr": base["HR"],
            "q_k": priced["K"],
            "q_1b": priced["1B"],
            "q_h": priced["1B"] + priced["2B"] + priced["3B"] + priced["HR"],
            "q_hr": priced["HR"],
            "m_k": m_k,
            "m_hit": m_hit,
            "f_k": f_k,
            "f_hit": f_hit,
            "y_pa": float(row.y_pa),
            "y_k": row.n_k / row.y_pa,
            "y_1b": row.n_1b / row.y_pa,
            "y_h": row.n_h / row.y_pa,
            "y_hr": row.n_hr / row.y_pa,
        })
        if i % 20000 == 0:
            print(f"  ... {i:,} matchups scanned, {len(rows):,} rows", flush=True)
    return pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)


def raw_factors(p: pd.DataFrame) -> pd.DataFrame:
    """Flag the matchups that sit on the shipped clips."""
    p = p.copy()
    p["on_k_clip"] = (p.m_k <= SHIPPED_K_CLIP[0] + 1e-9) | (
        p.m_k >= SHIPPED_K_CLIP[1] - 1e-9
    )
    p["on_hit_clip"] = (p.m_hit <= SHIPPED_HIT_CLIP[0] + 1e-9) | (
        p.m_hit >= SHIPPED_HIT_CLIP[1] - 1e-9
    )
    return p


def report(name: str, base: str, truth: str, mult: str, train: pd.DataFrame,
           test: pd.DataFrame) -> None:
    b, q = test[base].to_numpy(), (test[base] * test[mult]).to_numpy()
    print(f"\n--- {name} ---")
    print(f"  baseline (log5 matchup)            wRMSE {wrmse(b, test[truth], test.y_pa):.5f}")
    print(f"  baseline x arsenal term            wRMSE {wrmse(q, test[truth], test.y_pa):.5f}")
    doses = np.linspace(0.0, 2.0, 41)
    best = min(
        doses,
        key=lambda d: wrmse(train[base] * train[mult] ** d, train[truth], train.y_pa),
    )
    dosed = test[base] * test[mult] ** best
    print(f"  baseline x term^{best:.2f} (dose on train)  "
          f"wRMSE {wrmse(dosed, test[truth], test.y_pa):.5f}")
    grid = "  ".join(
        f"^{d:.2f} {wrmse(test[base] * test[mult] ** d, test[truth], test.y_pa):.5f}"
        for d in (0.0, 0.25, 0.4, 0.5, 0.75, 1.0)
    )
    print(f"  on the holdout, dose by dose       {grid}")
    # Two halves of the holdout, because one preferred dose is an anecdote.
    half = test.game_date.quantile(0.5)
    for label, part in (("first half", test[test.game_date <= half]),
                        ("second half", test[test.game_date > half])):
        d_best = min(
            doses,
            key=lambda d: wrmse(part[base] * part[mult] ** d, part[truth], part.y_pa),
        )
        print(f"    {label:<12} prefers ^{d_best:.2f}  (n {len(part):,})")

    # What the term claims against what happened, in its own units.
    ratio = test[truth] / test[base].replace(0, np.nan)
    fit = sm.WLS(
        ratio, sm.add_constant(test[mult], has_constant="add"), weights=test.y_pa
    ).fit(missing="drop")
    slope = fit.params.iloc[1]
    print(f"  realised/baseline on the term      slope {slope:+.3f} "
          f"(t {fit.tvalues.iloc[1]:+.2f})   -- calibrated is +1.0")

    # ...and again holding the baseline still, because the term is correlated
    # with it: a hitter who handles the arm's best pitch is usually the better
    # hitter, whose rates already say so. A slope that survives this control is
    # information the vector does not already have.
    ctrl = sm.WLS(
        ratio,
        sm.add_constant(test[[mult, base]], has_constant="add"),
        weights=test.y_pa,
    ).fit(missing="drop")
    print(f"  ...holding the baseline still     slope {ctrl.params.iloc[1]:+.3f} "
          f"(t {ctrl.tvalues.iloc[1]:+.2f})")

    edges = test[mult].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_numpy()
    bucket = pd.cut(test[mult], np.unique(edges), include_lowest=True)
    tbl = test.groupby(bucket, observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "pa": g.y_pa.sum(),
            "term": g[mult].mean(),
            "baseline": np.average(g[base], weights=g.y_pa),
            "priced": np.average(g[base] * g[mult], weights=g.y_pa),
            "realised": np.average(g[truth], weights=g.y_pa),
        }),
        include_groups=False,
    )
    print(tbl.round(4).to_string())


def main() -> None:
    if PANEL_CACHE.exists() and not os.environ.get("ARSENAL_REBUILD"):
        p = pd.read_pickle(PANEL_CACHE)
    else:
        pitches = load_pitches()
        pas = pa_rows(pitches)
        mus = matchup_outcomes(pas, starters(pas))
        print(f"{len(mus):,} batter-vs-starter matchups; reading the 42-day windows")
        p = panel(pitches, pas, mus)
        PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        p.to_pickle(PANEL_CACHE)
    p = raw_factors(p)

    print(f"\n{len(p):,} matchups, {p.y_pa.sum():,.0f} plate appearances, "
          f"{p.game_date.min():%Y-%m-%d}..{p.game_date.max():%Y-%m-%d}")
    print(f"K term:   median {p.m_k.median():.3f}  p5 {p.m_k.quantile(0.05):.3f}  "
          f"p95 {p.m_k.quantile(0.95):.3f}  on the clip {p.on_k_clip.mean():.1%}")
    print(f"hit term: median {p.m_hit.median():.3f}  p5 {p.m_hit.quantile(0.05):.3f}  "
          f"p95 {p.m_hit.quantile(0.95):.3f}  on the clip {p.on_hit_clip.mean():.1%}")

    cut = p.game_date.quantile(0.6)
    train, test = p[p.game_date <= cut], p[p.game_date > cut]
    print(f"\nholdout from {cut:%Y-%m-%d}: train {len(train):,}, test {len(test):,}")
    report("strikeouts", "p_k", "y_k", "m_k", train, test)
    report("singles", "p_1b", "y_1b", "m_hit", train, test)
    report("hits", "p_h", "y_h", "m_hit", train, test)
    report("home runs", "p_hr", "y_hr", "m_hit", train, test)

    print("\n===== what production does now, on the same holdout =====")
    now_k = term(test.f_k, _K_GAIN, _K_CLIP)
    for label, outcome, base, truth, mult in (
        ("strikeouts", "K", "p_k", "y_k", now_k),
        ("hits", "H", "p_h", "y_h", np.ones(len(test))),
    ):
        was = test[base] * (test.m_k if outcome == "K" else test.m_hit)
        print(f"  {label:<12} before {wrmse(was, test[truth], test.y_pa):.5f}"
              f"   now {wrmse(test[base] * mult, test[truth], test.y_pa):.5f}"
              f"   no term at all {wrmse(test[base], test[truth], test.y_pa):.5f}")

    print("\n===== league baselines the term measures against =====")
    print(f"  swstr {LEAGUE_SWSTR}\n  whiff {LEAGUE_WHIFF}\n  xwoba {LEAGUE_XWOBA}")


if __name__ == "__main__":
    main()
