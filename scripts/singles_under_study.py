"""Fit the singles-Under profile out of time, instead of asserting its weights.

``mlb_engine/features/singles_under.py`` scores five hand-picked flags with
hand-picked weights (2.0 for a TTO profile, 1.5 for fly-ball tilt, ...) taken
from a framework rather than from data. This measures each of them, and the
score they add up to, against what a batter actually did *after* the window:

    unit       one batter-game with at least one plate appearance
    target     the batter records NO single -> the u0.5 singles bet wins
    features   the profile built from a 42-day window ending before the game
    design     8 rolling blocks, features from the window, outcome from the
               7 days after it, so nothing is fitted on its own outcome

Controls matter more here than in the extra-base studies: the singles under is
decided mostly by *opportunity* (a leadoff man gets five cracks at it) and by
the batter's own singles rate, so any flag has to earn its place on top of
those two.
"""

from __future__ import annotations

import glob
import math
import warnings

import pandas as pd
import statsmodels.api as sm

from mlb_engine.data.statcast import dedupe_pitches
from mlb_engine.features.singles_under import (
    MIN_BIP,
    MIN_PA,
    build_singles_under,
    singles_under_score,
)

warnings.filterwarnings("ignore")

CACHE = "/home/ubuntu/.mlb_engine/cache/statcast_*.pkl"
WINDOW = 42
FORWARD = 7
FEATURES = [
    "k_pct", "bb_pct", "z_swing", "avg_la", "barrel", "hard_hit", "pull_rate",
]
CONTROLS = ["singles_rate", "pa_per_game"]


def load() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(CACHE)):
        d = pd.read_pickle(path)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"])
    # The cached ranges overlap, and a pitch carries no id (#108).
    d = dedupe_pitches(d)
    d["game"] = d["home_team"].astype(str) + "|" + d["away_team"].astype(str)
    return d.sort_values("game_date")


def outcomes(d: pd.DataFrame) -> pd.DataFrame:
    """Per batter-game: plate appearances and singles."""
    pa = d[d["events"].notna()]
    g = pa.groupby(["game_date", "game", "batter"])
    return pd.DataFrame(
        {
            "pa": g.size(),
            "singles": g["events"].apply(lambda s: int((s == "single").sum())),
        }
    ).reset_index()


def profiles(win: pd.DataFrame) -> pd.DataFrame:
    """The engine's own profile for every batter with enough window sample."""
    rows = []
    pa_all = win[win["events"].notna()]
    for bid, bdf in win.groupby("batter"):
        stand = bdf["stand"].mode()
        prof = build_singles_under(bdf, stand.iloc[0] if len(stand) else None)
        if not prof.has_data or prof.bip < MIN_BIP:
            continue
        ev = pa_all[pa_all["batter"] == bid]
        n_pa = len(ev)
        games = ev.groupby(["game_date", "game"]).ngroups
        score, _ = singles_under_score(prof)
        rows.append(
            {
                "batter": bid,
                "k_pct": prof.k_pct,
                "bb_pct": prof.bb_pct,
                "z_swing": prof.z_swing,
                "avg_la": prof.avg_la,
                "barrel": prof.barrel,
                "hard_hit": prof.hard_hit,
                "pull_rate": prof.pull_rate,
                "singles_rate": float((ev["events"] == "single").sum() / n_pa),
                "pa_per_game": float(n_pa / games) if games else float("nan"),
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def build(d: pd.DataFrame) -> pd.DataFrame:
    days = sorted(d["game_date"].unique())
    start, end = days[0], days[-1]
    blocks = []
    cut = start + pd.Timedelta(days=WINDOW)
    while cut + pd.Timedelta(days=FORWARD) <= end:
        blocks.append(cut)
        cut = cut + pd.Timedelta(days=FORWARD)
    out = []
    for i, cut in enumerate(blocks):
        win = d[(d["game_date"] >= cut - pd.Timedelta(days=WINDOW)) & (d["game_date"] < cut)]
        fwd = d[(d["game_date"] >= cut) & (d["game_date"] < cut + pd.Timedelta(days=FORWARD))]
        prof = profiles(win)
        if prof.empty:
            continue
        y = outcomes(fwd)
        m = y.merge(prof, on="batter", how="inner")
        m["block"] = i
        out.append(m)
        print(
            f"  block {i}: window to {cut.date()}  batters {len(prof)}  "
            f"batter-games {len(m)}"
        )
    return pd.concat(out, ignore_index=True)


def fit(d: pd.DataFrame, cols: list[str]) -> pd.Series | None:
    x = d[cols].astype(float)
    x = sm.add_constant(x)
    y = d["no_single"].astype(float)
    ok = x.notna().all(axis=1)
    try:
        res = sm.Logit(y[ok], x[ok]).fit(disp=0)
    except Exception as exc:  # pragma: no cover - study script
        print("   fit failed:", exc)
        return None
    return pd.Series({"n": int(ok.sum()), **{c: res.params[c] for c in cols},
                      **{f"z_{c}": res.tvalues[c] for c in cols}})


def z_score(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = d.copy()
    for c in cols:
        s = d[c].astype(float)
        z[c] = (s - s.mean()) / (s.std() or 1.0)
    return z


def main() -> None:
    print("loading statcast cache ...")
    d = load()
    print(f"{len(d):,} pitches, {d['game_date'].min().date()} .. {d['game_date'].max().date()}")
    print(f"blocks (window {WINDOW}d -> forward {FORWARD}d), min {MIN_PA} PA / {MIN_BIP} BIP:")
    m = build(d)
    m = m[m["pa"] >= 1].copy()
    m["no_single"] = (m["singles"] == 0).astype(float)
    print(
        f"\n{len(m):,} batter-games, {m['batter'].nunique()} batters, "
        f"{m['block'].nunique()} blocks"
    )
    print(f"base rate: no single in {m['no_single'].mean() * 100:.1f}% of batter-games\n")

    zs = z_score(m, FEATURES + CONTROLS)

    print("-- each flag on its own (standardized, no controls)")
    print(f"{'feature':<14}{'coef':>9}{'z':>8}{'p':>9}")
    for c in FEATURES + CONTROLS:
        r = fit(zs, [c])
        if r is None:
            continue
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(r[f"z_{c}"]) / math.sqrt(2))))
        print(f"{c:<14}{r[c]:>+9.4f}{r[f'z_{c}']:>8.2f}{p:>9.4f}")

    print("\n-- controlled for opportunity and the batter's own singles rate")
    print(f"{'feature':<14}{'coef':>9}{'z':>8}{'p':>9}")
    for c in FEATURES:
        r = fit(zs, [c, *CONTROLS])
        if r is None:
            continue
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(r[f"z_{c}"]) / math.sqrt(2))))
        print(f"{c:<14}{r[c]:>+9.4f}{r[f'z_{c}']:>8.2f}{p:>9.4f}")

    print("\n-- all together (the coats-off fit)")
    r = fit(zs, FEATURES + CONTROLS)
    if r is not None:
        print(f"{'feature':<14}{'coef':>9}{'z':>8}")
        for c in FEATURES + CONTROLS:
            print(f"{c:<14}{r[c]:>+9.4f}{r[f'z_{c}']:>8.2f}")

    print("\n-- sign stability across blocks (joint fit, per block)")
    signs: dict[str, list[float]] = {c: [] for c in FEATURES}
    for _blk, g in zs.groupby("block"):
        rb = fit(g, FEATURES + CONTROLS)
        if rb is None:
            continue
        for c in FEATURES:
            signs[c].append(float(rb[c]))
    for c in FEATURES:
        v = signs[c]
        pos = sum(1 for x in v if x > 0)
        print(f"{c:<14} {pos}/{len(v)} blocks positive   " + " ".join(f"{x:+.2f}" for x in v))

    print("\n-- the current hand-weighted score, out of time")
    print(f"{'score':<10}{'n':>7}{'no-single%':>12}{'vs base':>9}")
    base = m["no_single"].mean() * 100
    for s, g in m.groupby(m["score"].round(1)):
        if len(g) < 200:
            continue
        r = g["no_single"].mean() * 100
        print(f"{s:<10}{len(g):>7}{r:>12.1f}{r - base:>+9.1f}")
    strong = m[m["score"] >= 3.0]
    rest = m[m["score"] < 3.0]
    print(
        f"\nstrong (score>=3): n={len(strong)} no-single {strong['no_single'].mean() * 100:.1f}%"
        f"   rest: n={len(rest)} {rest['no_single'].mean() * 100:.1f}%"
    )

    print("\n-- each flag as the screen actually fires it")
    flags = {
        "TTO (K>25 & BB>12)": (m["k_pct"] > 0.25) & (m["bb_pct"] > 0.12),
        "high K% (>25)": m["k_pct"] > 0.25,
        "passive Z-Swing (<60)": m["z_swing"] < 0.60,
        "fly-ball tilt (LA>20)": m["avg_la"] > 20.0,
        "power contact": (m["barrel"] > 0.15) & (m["hard_hit"] > 0.48),
        "pull grounders": (m["pull_rate"] > 0.45) & (m["avg_la"] < 5.0),
    }
    print(f"{'flag':<24}{'n':>7}{'no-single%':>12}{'vs base':>9}")
    for name, mask in flags.items():
        g = m[mask.fillna(False)]
        if len(g) < 100:
            print(f"{name:<24}{len(g):>7}{'--':>12}{'thin':>9}")
            continue
        r = g["no_single"].mean() * 100
        print(f"{name:<24}{len(g):>7}{r:>12.1f}{r - base:>+9.1f}")

    m.to_csv("/home/ubuntu/singles_under_fit.csv", index=False)
    print("\nwrote /home/ubuntu/singles_under_fit.csv")


if __name__ == "__main__":
    main()
