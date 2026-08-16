"""What a starter's fastball velocity is worth, and over how many days.

The engine reads ``release_speed`` in exactly one place -- ``features/trend.py``,
which prints an arrow in the slate article and touches no price. This measures
whether that is the right home for it.

Four modes, all built from pitches thrown strictly before the game:

    reliability  what one start measures, by metric: the correlation between
                 consecutive starts by the same pitcher.
    window       which velocity window predicts the next start best, 1 to 8
                 weeks, decayed blends, and a level plus a last-start arrow.
    k            binomial deviance on strikeouts per PA against the starter,
                 the bar every K term in ``k_multiplier`` has to clear.
    hits         the same on hits per PA -- the bar that inverse-BABIP and
                 dxwOBA both failed in ``starter_contact_instrument``.

    python -m scripts.velocity_read_study reliability
    python -m scripts.velocity_read_study window --cache <statcast pickle>

Findings (2026 season through 08-15, 3,256 starts by 253 pitchers):

* **One start measures how the ball leaves the hand, and nothing about what the
  hitters did with it.** Correlation between consecutive starts: release height
  .97, extension .95, velocity .93, spin .91, IVB .84 -- then a cliff to whiff
  per swing .20, K per PA .20, CSW% .15, xwOBA allowed .10, BB per PA .07. A
  velocity read off a single outing is legitimate; a contact read is not, and
  the reason is that one start is ~90 radar-measured fastballs and ~22 results.

* **Shorter is better, monotonically.** Held-out RMSE on the next start's K
  rate, one velocity read added to the levels the engine prices: 7d .10391,
  14d .10399, 21d .10408, 42d .10422. Exponential decay never beats the plain
  7-day read. Best of all is a season level plus the last-start deviation
  (.09987), and the deviation survives controls for a short previous outing,
  the month and days of rest (coefficient moves in the fourth decimal).

* **The deviation is asymmetric.** An arm a mph below his season mean is only
  ~30% of the way down in his next start; a mph above is ~55% of the way up. A
  dip is more often the one-off.

* **Velocity is the only shape metric that pays on both sides.** Added to the
  priced levels, held out: velocity -.00175 on the K forecast and -.00081 on
  xwOBA allowed; spin -.00033 / -.00011; extension and release height at noise.
  IVB *hurts* a K forecast (z -5.3) and carries home runs instead (z +13.1),
  which is the one place the engine already uses it.
"""

from __future__ import annotations

import argparse
import glob
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import dedupe_pitches

CACHE = "/home/ubuntu/.mlb_engine/cache/statcast_*.pkl"
FOUR_SEAM = ("FF", "FA")
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
CALLED = {"called_strike"}
K_EV = ("strikeout", "strikeout_double_play")
HITS = ("single", "double", "triple", "home_run")

MIN_PA = 12  # in the start being predicted
MIN_FB_START = 15  # four-seamers before a start's velocity is readable
MIN_HISTORY = 4  # readable prior starts before a pitcher enters the study
WINDOW = 42  # the level window the engine prices


def load(pattern: str) -> pd.DataFrame:
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(pattern))]
    shared = sorted(set.intersection(*(set(f.columns) for f in frames)))
    d = dedupe_pitches(pd.concat([f[shared] for f in frames], ignore_index=True))
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(d["game_date"]).dt.date,
            "pitcher": d["pitcher"].astype("int64"),
            "inning": pd.to_numeric(d["inning"], errors="coerce"),
            "events": d["events"].astype(str),
            "is_pa": d["events"].notna(),
            "is_fb": d["pitch_type"].astype(str).isin(FOUR_SEAM),
            "speed": pd.to_numeric(d["release_speed"], errors="coerce"),
            "spin": pd.to_numeric(d["release_spin_rate"], errors="coerce"),
            "ext": pd.to_numeric(d["release_extension"], errors="coerce"),
            "relz": pd.to_numeric(d["release_pos_z"], errors="coerce"),
            "pfxz": pd.to_numeric(d["pfx_z"], errors="coerce"),
            "xw": pd.to_numeric(d["estimated_woba_using_speedangle"], errors="coerce"),
        }
    )
    desc = d["description"].astype(str)
    out["csw"] = desc.isin(WHIFF | CALLED)
    out["whiff"] = desc.isin(WHIFF)
    out["swing"] = desc.isin(WHIFF | {"foul", "hit_into_play", "foul_bunt", "missed_bunt"})
    out["is_k"] = out["events"].isin(K_EV)
    out["is_hit"] = out["events"].isin(HITS)
    out["is_bb"] = out["events"].eq("walk")
    out["is_hr"] = out["events"].eq("home_run")
    return out


def per_start(df: pd.DataFrame) -> pd.DataFrame:
    """One row per start a pitcher opened: what he threw, and what it produced."""
    opened = df[df["inning"] == 1].groupby(["date", "pitcher"]).size().index
    game = df.groupby(["date", "pitcher"]).agg(
        pitches=("csw", "size"),
        csw_n=("csw", "sum"),
        whiff_n=("whiff", "sum"),
        swings=("swing", "sum"),
        pa=("is_pa", "sum"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        hits=("is_hit", "sum"),
        hr=("is_hr", "sum"),
        xw=("xw", "mean"),
        bbe=("xw", "count"),
        relz=("relz", "mean"),
    )
    fb = df[df["is_fb"] & df["speed"].notna()]
    shape = fb.groupby(["date", "pitcher"]).agg(
        vfa=("speed", "mean"),
        n_fb=("speed", "size"),
        spin=("spin", "mean"),
        ext=("ext", "mean"),
        ivb=("pfxz", "mean"),
    )
    shape["ivb"] = shape["ivb"] * 12.0  # feet of break to inches
    out = game.join(shape, how="left")
    out = out[out.index.isin(opened)].reset_index()
    return out.sort_values(["pitcher", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# reliability: what does one start measure?
# --------------------------------------------------------------------------- #

RELIABILITY = (
    ("relz", "release height (ft)"),
    ("ext", "release extension (ft)"),
    ("vfa", "four-seam velo (mph)"),
    ("spin", "four-seam spin (rpm)"),
    ("ivb", "four-seam IVB (in)"),
    ("whiff_rate", "whiff / swing"),
    ("k_rate", "K per PA"),
    ("csw_rate", "CSW%"),
    ("xw", "xwOBA allowed"),
    ("bb_rate", "BB per PA"),
)


def reliability(st: pd.DataFrame) -> None:
    st = st[(st["pa"] >= MIN_PA) & (st["pitches"] >= 60)].copy()
    st["whiff_rate"] = st["whiff_n"] / st["swings"]
    st["k_rate"] = st["k"] / st["pa"]
    st["bb_rate"] = st["bb"] / st["pa"]
    st["csw_rate"] = st["csw_n"] / st["pitches"]
    print(f"{len(st):,} starts, {st['pitcher'].nunique()} pitchers\n")
    print(f"{'read off ONE start':<26}{'pairs':>8}{'r':>7}{'sd across':>11}{'sd 1 start':>12}")
    for col, name in RELIABILITY:
        s = st[["pitcher", "date", col]].dropna().sort_values(["pitcher", "date"])
        s["next"] = s.groupby("pitcher")[col].shift(-1)
        p = s.dropna(subset=["next"])
        if len(p) < 50:
            continue
        r = float(np.corrcoef(p[col], p["next"])[0, 1])
        season = st.groupby("pitcher")[col].mean().dropna()
        print(f"{name:<26}{len(p):>8,}{r:>7.2f}{season.std():>11.3f}{p[col].std():>12.3f}")


# --------------------------------------------------------------------------- #
# window: which velocity read predicts the next start?
# --------------------------------------------------------------------------- #

WINDOWS = (7, 14, 21, 28, 42, 56)
HALFLIVES = (7.0, 14.0, 21.0, 35.0)


def _priced_levels(past: pd.DataFrame) -> dict[str, float] | None:
    """What the engine already knows about him: six starts of K%, CSW%, xwOBAcon."""
    s6 = past.tail(6)
    if s6["pa"].sum() < 60 or s6["bbe"].sum() < 20:
        return None
    return {
        "k_pct": float(s6["k"].sum() / s6["pa"].sum()),
        "csw": float(s6["csw_n"].sum() / s6["pitches"].sum()),
        "xwa": float((s6["xw"] * s6["bbe"]).sum() / s6["bbe"].sum()),
    }


def window_rows(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in st.groupby("pitcher"):
        g = g.reset_index(drop=True)
        readable = g["n_fb"] >= MIN_FB_START
        for i in range(len(g)):
            cur = g.iloc[i]
            if cur["pa"] < MIN_PA:
                continue
            past = g.iloc[:i][readable.iloc[:i]]
            if len(past) < MIN_HISTORY:
                continue
            lv = _priced_levels(past)
            if lv is None:
                continue
            rec: dict[str, object] = {
                "date": cur["date"],
                "pitcher": pid,
                "k_rate": cur["k"] / cur["pa"],
                "xwoba": cur["xw"],
                "pa": float(cur["pa"]),
                "bip": float(cur["bbe"]),
                **lv,
            }
            ok = True
            for w in WINDOWS:
                s = past[past["date"] >= cur["date"] - timedelta(days=w)]
                if s["n_fb"].sum() < 20:
                    ok = False
                    break
                rec[f"vfa{w}"] = float((s["vfa"] * s["n_fb"]).sum() / s["n_fb"].sum())
            if not ok:
                continue
            age = np.array([(cur["date"] - d).days for d in past["date"]], dtype=float)
            for hl in HALFLIVES:
                wt = 0.5 ** (age / hl) * past["n_fb"].to_numpy(float)
                rec[f"hl{int(hl)}"] = float((wt * past["vfa"].to_numpy(float)).sum() / wt.sum())
            season = float((past["vfa"] * past["n_fb"]).sum() / past["n_fb"].sum())
            rec["season"] = season
            rec["last"] = float(past.iloc[-1]["vfa"])
            rec["dev"] = rec["last"] - season
            rows.append(rec)
    return pd.DataFrame(rows).dropna()


def wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([np.ones(len(x)), x])
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x, xtw @ y)
    resid = y - x @ beta
    s2 = float((w * resid**2).sum() / (w.sum() - x.shape[1]))
    se = np.sqrt(np.diag(np.linalg.inv(xtw @ x) * s2))
    return beta, se


def holdout_rmse(
    d: pd.DataFrame, cols: list[str], target: str, weight: str
) -> tuple[float, float]:
    dd = d.dropna(subset=cols + [target])
    cut = dd["date"].quantile(0.5)
    tr, te = dd[dd["date"] <= cut], dd[dd["date"] > cut]
    beta, se = wls(
        tr[cols].to_numpy(float), tr[target].to_numpy(float), tr[weight].to_numpy(float)
    )
    x = np.column_stack([np.ones(len(te)), te[cols].to_numpy(float)])
    w = te[weight].to_numpy(float)
    rmse = float(np.sqrt((w * (te[target].to_numpy(float) - x @ beta) ** 2).sum() / w.sum()))
    return rmse, float(beta[-1] / se[-1])


def window(st: pd.DataFrame) -> None:
    d = window_rows(st)
    print(
        f"{len(d):,} starts, {d['pitcher'].nunique()} pitchers, "
        f"{d['date'].min()} to {d['date'].max()}"
    )
    base = ["k_pct", "csw", "xwa"]
    for target, weight in (("k_rate", "pa"), ("xwoba", "bip")):
        print(f"\n=== {target}: one velocity read on top of the priced levels ===")
        r0, _ = holdout_rmse(d, base, target, weight)
        print(f"  {'no velocity':<24} RMSE {r0:.5f}")
        reads = (
            [(f"vfa{w}", f"{w} days") for w in WINDOWS]
            + [(f"hl{int(h)}", f"{int(h)}d half-life") for h in HALFLIVES]
            + [("season", "season to date"), ("last", "last start")]
        )
        for col, name in reads:
            r, z = holdout_rmse(d, base + [col], target, weight)
            print(f"  {name:<24} RMSE {r:.5f} ({r - r0:+.5f})  z {z:+6.2f}")
        r, z = holdout_rmse(d, base + ["season", "dev"], target, weight)
        print(f"  {'season + last-start dev':<24} RMSE {r:.5f} ({r - r0:+.5f})  dev z {z:+6.2f}")

    # How much of a one-start deviation is still there next time out?
    print("\n-- how much of a deviation carries into the next start? --")
    d = d.sort_values(["pitcher", "date"])
    d["next_dev"] = d.groupby("pitcher")["dev"].shift(-1)
    p = d.dropna(subset=["next_dev"])
    for lo, hi, name in ((-9, -0.7, "down >0.7"), (-0.3, 0.3, "flat"), (0.7, 9, "up >0.7")):
        s = p[(p["dev"] > lo) & (p["dev"] <= hi)]
        if len(s):
            print(
                f"  {name:<10} n={len(s):<5} this start {s['dev'].mean():+.2f} mph vs season, "
                f"next {s['next_dev'].mean():+.2f}  (carry {s['next_dev'].mean() / s['dev'].mean():.0%})"
            )


# --------------------------------------------------------------------------- #
# k / hits: the deviance bar a priced term has to clear
# --------------------------------------------------------------------------- #


def deviance(p: np.ndarray, made: np.ndarray, pa: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = made * np.log(p) + (pa - made) * np.log1p(-p)
    return float(-2 * ll.sum() / pa.sum())


def logit_fit(x: np.ndarray, made: np.ndarray, pa: np.ndarray, iters: int = 60) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(made.sum() / (pa.sum() - made.sum()))
    for _ in range(iters):
        eta = x @ beta
        p = 1 / (1 + np.exp(-eta))
        w = pa * p * (1 - p)
        z = eta + (made - pa * p) / np.maximum(w, 1e-9)
        xw = x * w[:, None]
        nxt = np.linalg.solve(x.T @ xw + 1e-9 * np.eye(x.shape[1]), xw.T @ z)
        if np.max(np.abs(nxt - beta)) < 1e-9:
            return nxt
        beta = nxt
    return beta


def event_rows(st: pd.DataFrame) -> pd.DataFrame:
    """One row per start: the event counts, the priced levels, and velocity."""
    rows = []
    for pid, g in st.groupby("pitcher"):
        g = g.reset_index(drop=True)
        readable = g["n_fb"] >= MIN_FB_START
        for i in range(len(g)):
            cur = g.iloc[i]
            if cur["pa"] < MIN_PA:
                continue
            past = g.iloc[:i][readable.iloc[:i]]
            if len(past) < MIN_HISTORY:
                continue
            lv = _priced_levels(past)
            if lv is None:
                continue
            season = float((past["vfa"] * past["n_fb"]).sum() / past["n_fb"].sum())
            rows.append(
                {
                    "date": cur["date"],
                    "pitcher": pid,
                    "pa": float(cur["pa"]),
                    "k": float(cur["k"]),
                    "hits": float(cur["hits"]),
                    "nonk_pa": float(cur["pa"] - cur["k"]),
                    "season": season,
                    "dev": float(past.iloc[-1]["vfa"]) - season,
                    **lv,
                }
            )
    return pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)


def event_study(st: pd.DataFrame, outcome: str) -> None:
    d = event_rows(st)
    made = d[outcome].to_numpy(float)
    # Hits are scored per *non-strikeout* PA: a harder thrower allows fewer hits
    # partly by striking men out, and that half is already priced by the K term.
    pa = d["nonk_pa" if outcome == "hits" else "pa"].to_numpy(float)
    ones = np.ones(len(d))
    ctrl = np.column_stack(
        [
            (d["k_pct"] - d["k_pct"].mean()).to_numpy(float),
            (d["csw"] - d["csw"].mean()).to_numpy(float),
            (d["xwa"] - d["xwa"].mean()).to_numpy(float),
        ]
    )
    season = (d["season"] - d["season"].mean()).to_numpy(float)
    dev = d["dev"].to_numpy(float)
    specs: dict[str, list[np.ndarray]] = {
        "none (priced levels)": [],
        "+ velocity level": [season],
        "+ last-start dev": [dev],
        "+ level and dev": [season, dev],
        # Does a dip need its own coefficient? It carries differently in the
        # velocity itself (30% of a dip, 55% of a spike), so it might here too.
        "+ level, dip only": [season, np.minimum(dev, 0.0)],
        "+ level, spike only": [season, np.maximum(dev, 0.0)],
    }
    cut = int(len(d) * 0.6)
    print(
        f"\n{len(d):,} starts, {int(pa.sum()):,} PA, {d['date'].min()} to {d['date'].max()}"
        f"  (train {cut} / holdout {len(d) - cut})\n"
    )
    print(f"{'term':<22}{'coef':>9}{'t':>7}{'train dev':>12}{'holdout dev':>13}")
    for name, cols in specs.items():
        x = np.column_stack([ones, ctrl, *cols]) if cols else np.column_stack([ones, ctrl])
        b = logit_fit(x[:cut], made[:cut], pa[:cut])
        tr = deviance(1 / (1 + np.exp(-x[:cut] @ b)), made[:cut], pa[:cut])
        ho = deviance(1 / (1 + np.exp(-x[cut:] @ b)), made[cut:], pa[cut:])
        if cols:
            b_all = logit_fit(x, made, pa)
            p = 1 / (1 + np.exp(-(x @ b_all)))
            w = pa * p * (1 - p)
            cov = np.linalg.inv(x.T @ (x * w[:, None]))
            i = x.shape[1] - 1
            print(
                f"{name:<22}{b_all[i]:>9.4f}{b_all[i] / np.sqrt(cov[i, i]):>7.2f}"
                f"{tr:>12.5f}{ho:>13.5f}"
            )
        else:
            print(f"{name:<22}{'--':>9}{'--':>7}{tr:>12.5f}{ho:>13.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("reliability", "window", "k", "hits"))
    # ``hits`` scores hits per non-strikeout PA, so the two terms do not
    # double-count the strikeouts a harder fastball prevents.
    ap.add_argument("--cache", default=CACHE, help="glob of Statcast pickles")
    args = ap.parse_args()
    st = per_start(load(args.cache))
    if args.mode == "reliability":
        reliability(st)
    elif args.mode == "window":
        window(st)
    else:
        event_study(st, "k" if args.mode == "k" else "hits")


if __name__ == "__main__":
    main()
