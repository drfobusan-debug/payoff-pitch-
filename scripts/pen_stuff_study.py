"""Does a bullpen's stuff, its location, and the direction they are moving predict?

The engine reads an opposing bullpen three ways: an outcome vector off the last
21 days of relief work, a strikeout multiplier off its stuff (``k_multiplier``:
CSW%, K-BB%, 2-strike whiff, Stuff+ when a feed exists), and two NPV tripwires,
one of which is location -- Zone% below .40 marks a walk trap. What it does not
read is direction: ``pitcher_trends`` exists for starters only, and nothing in
the pen path compares the recent half of the window with the earlier half.

This measures all three, on the same design as the xK study: every team-game of
relief work is predicted from relief pitches thrown strictly before it, scored on
what the pen actually did next, weighted by batters faced, with a chronological
holdout and a weekly walk-forward.

    python -m scripts.pen_stuff_study

Findings, 3,258 team-games by 30 pens over 2026 (193,675 relief pitches):

* **Stuff and location both predict, weakly, and the longer window is the better
  read.** Out of sample against the next game, weighted by batters faced:

      next-game relief K%      21d      42d
        league mean          .11910   .11911
        CSW% alone           .11864   .11849
        SwStr% alone         .11870   .11858
        observed relief K%   .11804   .11785

  Nothing beats simply reading the pen's pooled strikeout rate over six weeks,
  and CSW%/SwStr% add nothing on top of it (t -0.66 and +0.54 with K% in the
  model). A pen is not a starter: 487 batters faced is enough that the outcome
  is a better estimate of the skill than the peripherals are.

* **The shipped ``k_multiplier`` makes the prediction worse, and it is centred
  below 1.0.** Realised K% against the window rate scores .12081; against the
  window rate times the multiplier, .12634. With the window K% held fixed, none
  of its three terms carries weight (CSW +0.03 t +0.17, K-BB -0.03 t -0.31,
  2-strike whiff +0.18 t +1.49) -- and the multiplier averages **0.926** over
  pen-games and **0.909** over starters, i.e. it is a near-unconditional
  strikeout tax rather than a discriminator.

  The cause is a unit mismatch in one baseline. ``two_strike_whiff`` is built as
  whiffs per two-strike *pitch* (league .145 here) and compared against
  ``BL_TWO_STRIKE_WHIFF = 0.280``, which is a put-away rate per two-strike
  *swing* (.242 here). Every arm therefore sits at that term's -0.06 clip: 98%
  of the 201 starters with 400+ pitches, and effectively every bullpen.

* **Location predicts walks; the tripwire is set outside the distribution.**
  Zone% carries -0.23 (t -1.70) on the next game's relief BB% and chase -0.21
  (t -1.88), and the quintiles are monotone in the right direction (.1108 BB%
  allowed at Zone% .442 against .0953 at .498). But the walk-trap fires below
  Zone% .40, and the *minimum* over 3,258 pen-games is .394 -- it fired twice,
  0.06% of the time.

* **Trend adds nothing.** Recent-half-minus-earlier-half in CSW%, K%, SwStr%,
  velocity, Zone% or wOBA carries no weight once the level is in the model
  (every |t| < 1.5) -- the same answer #147 gave for a starter's CSW trend.

Stuff+ and Location+ are read only from an optional FanGraphs CSV, are never
passed for a bullpen, and ``location_plus`` is not used by any multiplier at all.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_engine.features.regression import (
    BL_CSW,
    BL_K_MINUS_BB,
    BL_TWO_STRIKE_WHIFF,
    CALLED_OR_WHIFF,
    K_EVENTS_P,
    SWING_DESC,
    WHIFF_DESC,
)

CACHE = os.path.expanduser("~/.mlb_engine/cache")
MIN_PA_GAME = 6  # batters faced by the pen in the game being predicted
MIN_PA_WINDOW = 60  # relief PA in the look-back before the pen is read at all
FASTBALLS = ("FF", "FA", "SI", "FC")

SUMS = [
    "pitches", "n_csw", "n_whiff", "n_swing", "n_zone", "n_oz_swing",
    "n_first", "n_first_strike", "n_two_strike", "n_two_strike_whiff",
    "pa", "n_k", "n_bb", "n_out_zone", "n_fb", "velo_sum", "woba_sum", "woba_den",
]

RATES = {
    "csw": lambda p: p.n_csw / p.pitches,
    "swstr": lambda p: p.n_whiff / p.pitches,
    "called": lambda p: (p.n_csw - p.n_whiff) / p.pitches,
    "whiff_swing": lambda p: p.n_whiff / p.n_swing,
    "zone": lambda p: p.n_zone / p.pitches,
    "chase": lambda p: p.n_oz_swing / p.n_out_zone,
    "fstrike": lambda p: p.n_first_strike / p.n_first,
    "two_strike_whiff": lambda p: p.n_two_strike_whiff / p.n_two_strike,
    "k_pct": lambda p: p.n_k / p.pa,
    "bb_pct": lambda p: p.n_bb / p.pa,
    "k_minus_bb": lambda p: (p.n_k - p.n_bb) / p.pa,
    "velo": lambda p: p.velo_sum / p.n_fb,
    "woba": lambda p: p.woba_sum / p.woba_den,
}

K_SPECS: dict[str, list[str]] = {
    "observed relief K% only": ["k_pct"],
    "csw only": ["csw"],
    "swstr only": ["swstr"],
    "csw + swstr": ["csw", "swstr"],
    "engine's k_multiplier basis": ["csw", "k_minus_bb", "two_strike_whiff"],
    "K% + csw + swstr": ["k_pct", "csw", "swstr"],
    "K% + csw + swstr + trend": ["k_pct", "csw", "swstr", "d_csw", "d_k_pct"],
}
BB_SPECS: dict[str, list[str]] = {
    "observed relief BB% only": ["bb_pct"],
    "zone only (the tripwire's variable)": ["zone"],
    "zone + chase + fstrike": ["zone", "chase", "fstrike"],
    "BB% + zone + chase + fstrike": ["bb_pct", "zone", "chase", "fstrike"],
    "+ trend": ["bb_pct", "zone", "chase", "fstrike", "d_zone", "d_bb_pct"],
}
WOBA_SPECS: dict[str, list[str]] = {
    "observed relief wOBA only": ["woba"],
    "stuff only (csw + swstr)": ["csw", "swstr"],
    "stuff + location": ["csw", "swstr", "zone", "chase"],
    "wOBA + stuff + location": ["woba", "csw", "swstr", "zone", "chase"],
    "+ trend": ["woba", "csw", "swstr", "zone", "chase", "d_csw", "d_woba"],
}


def load_pitches() -> pd.DataFrame:
    """Every cached Statcast pitch, stitched into one non-overlapping season.

    The caches are overlapping rolling windows and the same pitch is not
    byte-identical between two pulls (batted-ball estimates are revised), so
    claiming each date once is the only exact de-overlap.
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
    return pd.concat(frames, ignore_index=True)


def relief_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Relief pitches only, tagged with the fielding team, as the engine reads them.

    Same definition as ``bullpen_relief_frame``: the pitching side comes from
    ``inning_topbot``, and any arm that appeared in the first inning of that
    game is that game's starter and is dropped.
    """
    d = df.copy()
    top = d["inning_topbot"].astype(str).str.startswith("Top")
    d["pen"] = np.where(top, d["home_team"], d["away_team"])
    starters = set(
        map(tuple, d.loc[d["inning"] <= 1, ["game_date", "pitcher"]].dropna().to_numpy())
    )
    keys = list(zip(d["game_date"], d["pitcher"], strict=True))
    d = d[[k not in starters for k in keys]]
    return d[d["inning"] >= 6]


def per_team_game(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, date): the counters both pen reads are built on."""
    d = d.copy()
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
    d["two_strike"] = d["strikes"] == 2
    d["two_strike_whiff"] = d["two_strike"] & d["whiff"]
    d["is_fb"] = d["pitch_type"].isin(FASTBALLS)
    d["fb_velo"] = np.where(d["is_fb"], pd.to_numeric(d["release_speed"], errors="coerce"), 0.0)
    d["woba_v"] = pd.to_numeric(d["woba_value"], errors="coerce").fillna(0.0)
    d["woba_d"] = pd.to_numeric(d["woba_denom"], errors="coerce").fillna(0.0)

    return d.groupby(["pen", "game_date"], as_index=False).agg(
        pitches=("csw", "size"),
        n_csw=("csw", "sum"),
        n_whiff=("whiff", "sum"),
        n_swing=("swing", "sum"),
        n_zone=("in_zone", "sum"),
        n_oz_swing=("oz_swing", "sum"),
        n_first=("first", "sum"),
        n_first_strike=("first_strike", "sum"),
        n_two_strike=("two_strike", "sum"),
        n_two_strike_whiff=("two_strike_whiff", "sum"),
        pa=("is_pa", "sum"),
        n_k=("k", "sum"),
        n_bb=("bb", "sum"),
        n_fb=("is_fb", "sum"),
        velo_sum=("fb_velo", "sum"),
        woba_sum=("woba_v", "sum"),
        woba_den=("woba_d", "sum"),
    ).assign(n_out_zone=lambda g: g.pitches - g.n_zone)


def _rates(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name, fn in RATES.items():
        out[prefix + name] = fn(frame)
    return out


def panel(games: pd.DataFrame, days: int) -> pd.DataFrame:
    """Each team-game, paired with that pen's prior ``days`` of relief work.

    The window is also split in half so the *direction* of each signal can be
    tested beside its level.
    """
    rows: list[dict[str, float]] = []
    half = days // 2
    for team, g in games.groupby("pen"):
        g = g.sort_values("game_date").reset_index(drop=True)
        dates = g["game_date"].to_numpy()
        arr = g[SUMS].to_numpy(dtype=float)
        for i in range(len(g)):
            floor = dates[i] - np.timedelta64(days, "D")
            mid = dates[i] - np.timedelta64(half, "D")
            win = (dates < dates[i]) & (dates >= floor)
            if not win.any():
                continue
            row: dict[str, float] = {"pen": team, "game_date": g["game_date"].iloc[i]}
            row.update(dict(zip(SUMS, arr[win].sum(axis=0), strict=True)))
            for label, mask in (("early", win & (dates < mid)), ("late", win & (dates >= mid))):
                row.update(
                    {f"{label}_{c}": v for c, v in zip(SUMS, arr[mask].sum(axis=0), strict=True)}
                )
            row["y_pa"] = float(g["pa"].iloc[i])
            row["y_k"] = float(g["n_k"].iloc[i])
            row["y_bb"] = float(g["n_bb"].iloc[i])
            row["y_woba_sum"] = float(g["woba_sum"].iloc[i])
            row["y_woba_den"] = float(g["woba_den"].iloc[i])
            rows.append(row)
    p = pd.DataFrame(rows)
    p = p[(p.pa >= MIN_PA_WINDOW) & (p.y_pa >= MIN_PA_GAME)].reset_index(drop=True)
    halves = {
        label: _rates(p[[f"{label}_{c}" for c in SUMS]].rename(
            columns=dict(zip([f"{label}_{c}" for c in SUMS], SUMS, strict=True))
        ))
        for label in ("early", "late")
    }
    p = pd.concat([p, _rates(p)], axis=1)
    for name in RATES:
        p["d_" + name] = halves["late"][name].to_numpy() - halves["early"][name].to_numpy()
    p["y_k_pct"] = p.y_k / p.y_pa
    p["y_bb_pct"] = p.y_bb / p.y_pa
    p["y_woba"] = p.y_woba_sum / p.y_woba_den.replace(0, np.nan)
    return p.replace([np.inf, -np.inf], np.nan).sort_values("game_date").reset_index(drop=True)


def k_multiplier(p: pd.DataFrame) -> pd.Series:
    """The shipped ``PitcherRegression.k_multiplier`` on these window rates."""
    m = 1.0 + np.clip((p.csw - BL_CSW) * 2.5, -0.15, 0.20)
    m *= 1.0 + np.clip((p.k_minus_bb - BL_K_MINUS_BB) * 1.5, -0.12, 0.15)
    m *= 1.0 + np.clip((p.two_strike_whiff - BL_TWO_STRIKE_WHIFF) * 0.8, -0.06, 0.08)
    return np.clip(m, 0.75, 1.30)


def wrmse(pred, truth: pd.Series, weight: pd.Series) -> float:
    return float(np.sqrt(np.average((np.asarray(pred) - truth.to_numpy()) ** 2, weights=weight)))


def _fit(train: pd.DataFrame, cols: list[str], target: str, weight: str):
    return sm.WLS(
        train[target], sm.add_constant(train[cols], has_constant="add"), weights=train[weight]
    ).fit()


def _apply(res, test: pd.DataFrame, cols: list[str], clip: tuple[float, float]) -> np.ndarray:
    return np.clip(res.predict(sm.add_constant(test[cols], has_constant="add")).to_numpy(), *clip)


def holdout(
    p: pd.DataFrame, specs: dict[str, list[str]], target: str, clip: tuple[float, float],
    weight: str, label: str, cut: pd.Timestamp,
) -> None:
    d = p.dropna(subset=[target, *{c for cols in specs.values() for c in cols}])
    train, test = d[d.game_date < cut], d[d.game_date >= cut]
    print(f"\n===== {label}: holdout from {cut.date()} (train {len(train)}, test {len(test)}) =====")
    base = float(np.average(train[target], weights=train[weight]))
    print(f"  {'league mean':<38} wRMSE {wrmse(np.full(len(test), base), test[target], test[weight]):.5f}")
    for name, cols in specs.items():
        res = _fit(train, cols, target, weight)
        terms = "  ".join(f"{c} {res.params[c]:+.3f} (t {res.tvalues[c]:+.2f})" for c in cols)
        err = wrmse(_apply(res, test, cols, clip), test[target], test[weight])
        print(f"  {name:<38} wRMSE {err:.5f}   {terms}")


def walk_forward(
    p: pd.DataFrame, specs: dict[str, list[str]], target: str, clip: tuple[float, float],
    weight: str, label: str, min_train: int = 400,
) -> None:
    d = p.dropna(subset=[target, *{c for cols in specs.values() for c in cols}])
    print(f"\n===== {label}: weekly walk-forward =====")
    preds: dict[str, list[np.ndarray]] = {k: [] for k in specs}
    preds["league mean"] = []
    tests: list[pd.DataFrame] = []
    for _, test in d.groupby(d.game_date.dt.to_period("W")):
        train = d[d.game_date < test.game_date.min()]
        if len(train) < min_train or test.empty:
            continue
        tests.append(test)
        preds["league mean"].append(
            np.full(len(test), float(np.average(train[target], weights=train[weight])))
        )
        for name, cols in specs.items():
            preds[name].append(_apply(_fit(train, cols, target, weight), test, cols, clip))
    scored = pd.concat(tests)
    for name, chunks in preds.items():
        print(f"  {name:<38} wRMSE {wrmse(np.concatenate(chunks), scored[target], scored[weight]):.5f}"
              f"  n={len(scored)}")


def multiplier_calibration(p: pd.DataFrame) -> None:
    """Is the shipped strikeout multiplier the right size?"""
    d = p.dropna(subset=["y_k_pct", "k_pct", "csw", "k_minus_bb", "two_strike_whiff"]).copy()
    d["km"] = k_multiplier(d)
    d["pred"] = d.k_pct * d.km
    print("\n===== the shipped k_multiplier on a bullpen =====")
    print(f"  range across pen-games: {d.km.min():.3f}..{d.km.max():.3f}"
          f"  mean {d.km.mean():.3f}  sd {d.km.std():.3f}")
    for name, col in (("window K% alone", "k_pct"), ("window K% x k_multiplier", "pred")):
        res = sm.WLS(d.y_k_pct, sm.add_constant(d[[col]]), weights=d.y_pa).fit()
        print(f"  realised K% on {name:<26} slope {res.params[col]:+.3f}"
              f" (t {res.tvalues[col]:+.2f})  wRMSE {wrmse(d[col], d.y_k_pct, d.y_pa):.5f}")
    res = sm.WLS(
        d.y_k_pct, sm.add_constant(d[["k_pct", "csw", "k_minus_bb", "two_strike_whiff"]]),
        weights=d.y_pa,
    ).fit()
    print("  fitted slopes, K% held fixed (the multiplier's own terms):")
    for c in ("csw", "k_minus_bb", "two_strike_whiff"):
        print(f"    {c:<18} {res.params[c]:+.3f} (t {res.tvalues[c]:+.2f})")
    print("  shipped, as a rate move at league K%: csw +2.5, k_minus_bb +1.5, "
          "two_strike_whiff +0.8 (multiplicative)")


def zone_tripwire(p: pd.DataFrame) -> None:
    """Does the walk trap ever fire, and does it mark anything?"""
    d = p.dropna(subset=["zone", "y_bb_pct"])
    below = d[d.zone < 0.40]
    print("\n===== the Zone% walk-trap tripwire (< .40) =====")
    print(f"  pen-games below .40: {len(below)} of {len(d)} ({100 * len(below) / len(d):.2f}%)"
          f"   Zone% p1 {d.zone.quantile(0.01):.3f}  min {d.zone.min():.3f}"
          f"  median {d.zone.median():.3f}")
    q = pd.qcut(d.zone, 5, duplicates="drop")
    table = d.groupby(q, observed=True).apply(
        lambda x: pd.Series({
            "n": len(x),
            "zone": np.average(x.zone, weights=x.y_pa),
            "next BB%": np.average(x.y_bb_pct, weights=x.y_pa),
        }),
        include_groups=False,
    )
    print(table.round(4).to_string())


def two_strike_units(relief: pd.DataFrame, pitches: pd.DataFrame) -> None:
    """Is ``BL_TWO_STRIKE_WHIFF`` on the same scale as the rate it is compared to?"""
    starters = set(
        map(tuple, pitches.loc[pitches["inning"] <= 1, ["game_date", "pitcher"]].dropna().to_numpy())
    )
    keys = list(zip(pitches["game_date"], pitches["pitcher"], strict=True))
    start_rows = pitches[[k in starters for k in keys]]
    print("\n===== BL_TWO_STRIKE_WHIFF: which rate is it? =====")
    for label, rows in (("starters", start_rows), ("relief", relief)):
        two = rows[rows["strikes"] == 2]
        per_pitch = float(two["description"].isin(WHIFF_DESC).mean())
        per_swing = float(
            two["description"].isin(WHIFF_DESC).sum() / two["description"].isin(SWING_DESC).sum()
        )
        print(f"  {label:<9} whiffs per 2-strike pitch {per_pitch:.3f}"
              f"   per 2-strike swing {per_swing:.3f}   constant {BL_TWO_STRIKE_WHIFF:.3f}")
    floor = BL_TWO_STRIKE_WHIFF - 0.06 / 0.8  # where the term's -0.06 clip binds
    per_arm = start_rows[start_rows["strikes"] == 2].groupby("pitcher")["description"].agg(
        [("rate", lambda s: s.isin(WHIFF_DESC).mean()), ("n", "size")]
    )
    per_arm = per_arm[per_arm["n"] >= 100]
    print(f"  starters pinned to the term's floor (rate <= {floor:.3f}): "
          f"{(per_arm['rate'] <= floor).mean():.1%} of {len(per_arm)} arms")


def trend_terms(p: pd.DataFrame) -> None:
    """Direction, tested against the level it is supposed to add to."""
    print("\n===== does the direction of travel add anything to the level? =====")
    for target, level, deltas in (
        ("y_k_pct", ["k_pct", "csw", "swstr"], ["d_csw", "d_k_pct", "d_swstr", "d_velo"]),
        ("y_bb_pct", ["bb_pct", "zone"], ["d_zone", "d_bb_pct", "d_fstrike"]),
        ("y_woba", ["woba", "csw"], ["d_woba", "d_csw", "d_velo"]),
    ):
        d = p.dropna(subset=[target, *level, *deltas])
        res = sm.WLS(d[target], sm.add_constant(d[level + deltas]), weights=d.y_pa).fit()
        terms = "  ".join(f"{c} {res.params[c]:+.3f} (t {res.tvalues[c]:+.2f})" for c in deltas)
        print(f"  {target:<10} n={len(d):<6} level={'+'.join(level):<22} {terms}")


def main() -> None:
    pitches = load_pitches()
    relief = relief_rows(pitches)
    games = per_team_game(relief)
    print(f"{len(pitches):,} pitches -> {len(relief):,} relief pitches -> "
          f"{len(games):,} team-games, {games.pen.nunique()} pens, "
          f"{games.game_date.min().date()}..{games.game_date.max().date()}")

    panels = {days: panel(games, days) for days in (21, 42)}
    for days, p in panels.items():
        print(f"  {days}d window: {len(p):,} predictable team-games, "
              f"median {p.pa.median():.0f} relief PA in the look-back")

    p21, p42 = panels[21], panels[42]
    cut = p42.game_date.quantile(0.6)

    print("\n################ WINDOW: 21 days (the engine's bullpen_days) ################")
    holdout(p21, K_SPECS, "y_k_pct", (0.05, 0.45), "y_pa", "next-game relief K%", cut)
    print("\n################ WINDOW: 42 days (bullpen_skill_days, shipped OFF) ################")
    holdout(p42, K_SPECS, "y_k_pct", (0.05, 0.45), "y_pa", "next-game relief K%", cut)
    walk_forward(p42, K_SPECS, "y_k_pct", (0.05, 0.45), "y_pa", "next-game relief K% (42d)")

    holdout(p42, BB_SPECS, "y_bb_pct", (0.01, 0.25), "y_pa", "next-game relief BB% (42d)", cut)
    holdout(p42, WOBA_SPECS, "y_woba", (0.10, 0.60), "y_woba_den",
            "next-game relief wOBA allowed (42d)", cut)

    multiplier_calibration(p21)
    two_strike_units(relief, pitches)
    zone_tripwire(p21)
    trend_terms(p42)


if __name__ == "__main__":
    main()
