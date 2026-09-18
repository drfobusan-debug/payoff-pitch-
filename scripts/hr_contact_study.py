"""Which contact metric tells you the home runs a starter is about to allow.

The HR multiplier in ``features/regression.py`` has been driven by barrel rate
allowed since it was written, on the grounds that barrels have the highest PPV
for HR/9. That is a statement about what a home run *is*, not about what a
pitcher *owns*: a barrel is mostly the hitter, and a starter sees ~16 batted
balls a night. This measures the whole family the same way -- barrel, hard-hit,
fly-ball rate, HR/FB and his own HR rate -- and asks which of them survives.

Three modes, every feature built from starts strictly before the game scored:

    reliability  split a pitcher's own starts odd against even: does the metric
                 describe him, or the hitters he happened to face?
    window       how many starts of fly-ball rate to read -- 1 to season, and
                 the same in calendar days, against the shipped six weeks.
    terms        binomial deviance per PA on the home runs he allows, one read
                 at a time on top of the levels the engine already prices.

    python -m scripts.hr_contact_study reliability
    python -m scripts.hr_contact_study terms --cache <statcast pickle>

Findings (2026 season through 08-15, 3,341 starts by 262 pitchers, 2,494 of them
predicted from earlier starts by the same arm):

* **Barrel% allowed does describe him, but weakly and only at sample.** Split-
  half across his own starts, each side holding 60+ batted balls: K/PA .64,
  fly-ball rate .52, CSW% .34, xwOBA allowed .30, hard-hit .28, barrel .24,
  HR/PA .21 -- and HR/FB never reaches 40 qualifying arms at all. Let arms with
  ~30 batted balls a side in and barrel collapses to .02 while fly-ball rate
  holds; the sensitivity is the finding.

* **And it does not forecast.** Added to the priced levels (K%, CSW%, xwOBA
  allowed), predicting the next start's HR per PA on a chronological 60/40
  holdout: barrel fits at t=+3.3 with the right sign and moves held-out deviance
  **+.00009 -- the wrong way**. HR/FB is t=+0.4 for nothing. His own prior HR/PA
  is t=+2.9 for -.00002. Fitting is not forecasting.

* **Where he lets the ball go is the read that works.** Fly-ball rate: t=+4.2,
  holdout -.00033 over four starts and -.00051 over three, against -.00017 on
  the six-week window the rest of the starter line uses and -.00002 at sixty
  days. Ground-ball rate is the same signal with the sign flipped (-.00042).
  Adding barrel *beside* fly-ball rate makes the pair worse than fly-ball rate
  alone (-.00014); adding hard-hit beside it helps a little (-.00040).

* **There is no arrow, unlike velocity.** A season fly-ball level plus a
  last-start deviation scores the deviation at z=+0.05. One start is ~16 batted
  balls: his last outing's fly-ball rate is the hitters, not him.

* **It is a small term either way.** ~.0003 of deviance is real and repeatable,
  but the park, the hitters and the four-seamer's ride are all larger, and none
  of this is ROI until the ledger has graded it.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from mlb_engine.data.statcast import dedupe_pitches

CACHE = "/home/ubuntu/.mlb_engine/cache/statcast_*.pkl"
MIN_PA = 12  # plate appearances before a start is scored as a start
MIN_HISTORY = 2  # prior starts before he can be predicted at all
MIN_BBE = 20  # batted balls in the control window
STARTS = (1, 2, 3, 4, 6, 10, 99)  # 99 == season to date
DAYS = (14, 21, 42, 60)
AIR = ("fly_ball", "popup", "line_drive")
FLY = ("fly_ball", "popup")


def load(pattern: str = CACHE) -> pd.DataFrame:
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(pattern))]
    if not frames:
        raise SystemExit(f"no Statcast pickles matched {pattern}")
    shared = sorted(set.intersection(*(set(f.columns) for f in frames)))
    d = dedupe_pitches(pd.concat([f[shared] for f in frames], ignore_index=True))
    events = d["events"].astype(str)
    bb_type = d["bb_type"].astype(str)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(d["game_date"]).dt.date,
            "pitcher": d["pitcher"].astype("int64"),
            "inning": pd.to_numeric(d["inning"], errors="coerce"),
            "is_pa": d["events"].notna(),
            "is_k": events.isin(("strikeout", "strikeout_double_play")),
            "is_hr": events.eq("home_run"),
            "barrel": pd.to_numeric(d.get("launch_speed_angle"), errors="coerce").eq(6),
            "hard": pd.to_numeric(d.get("launch_speed"), errors="coerce") >= 95.0,
            "xw": pd.to_numeric(d["estimated_woba_using_speedangle"], errors="coerce"),
            "csw": d["description"]
            .astype(str)
            .isin(
                (
                    "swinging_strike",
                    "swinging_strike_blocked",
                    "foul_tip",
                    "called_strike",
                )
            ),
        }
    )
    out["batted"] = bb_type.isin(("ground_ball", "line_drive", *FLY))
    out["air"] = bb_type.isin(AIR)
    out["fly"] = bb_type.isin(FLY)
    return out


def per_start(df: pd.DataFrame) -> pd.DataFrame:
    """One row per start, counts only -- rates are formed over windows later."""
    opened = df[df["inning"] == 1].groupby(["date", "pitcher"]).size().index
    game = df.groupby(["date", "pitcher"]).agg(
        pitches=("csw", "size"),
        csw_n=("csw", "sum"),
        pa=("is_pa", "sum"),
        k=("is_k", "sum"),
        hr=("is_hr", "sum"),
    )
    batted = df[df["batted"]].groupby(["date", "pitcher"]).agg(
        bbe=("barrel", "size"),
        barrels=("barrel", "sum"),
        hard=("hard", "sum"),
        fb=("fly", "sum"),
        air=("air", "sum"),
        xw_sum=("xw", "sum"),
        xw_n=("xw", "count"),
    )
    out = game.join(batted, how="left").fillna(0.0)
    out = out[out.index.isin(opened)].reset_index()
    out = out[out["pa"] >= MIN_PA]
    return out.sort_values(["pitcher", "date"]).reset_index(drop=True)


# --- reliability -----------------------------------------------------------
METRICS = (
    ("K per PA", "k", "pa"),
    ("fly-ball rate allowed", "fb", "bbe"),
    ("CSW%", "csw_n", "pitches"),
    ("HR per PA", "hr", "pa"),
    ("hard-hit% allowed", "hard", "bbe"),
    ("xwOBA allowed", "xw_sum", "xw_n"),
    ("barrel% allowed", "barrels", "bbe"),
    ("HR/FB% allowed", "hr", "fb"),
)


def reliability(st: pd.DataFrame, min_denom: int = 60) -> None:
    """Odd starts against even starts by the same arm: is this him?

    ``min_denom`` is required of *each* half, not of the season, and the answer
    moves a lot with it -- barrel% allowed reads .18 at 60 batted balls a side
    and .02 when arms with 30 a side are let in. That sensitivity is itself the
    finding: it is a rate that needs more sample than a starter's season gives.
    """
    st = st.copy()
    st["idx"] = st.groupby("pitcher").cumcount()
    print(f"\n  each half >= {min_denom} in the denominator")
    print("  metric                    n   split-half   full-season")
    for name, num, den in METRICS:
        rates, enough = [], []
        for parity in (0, 1):
            g = st[st["idx"] % 2 == parity].groupby("pitcher")[[num, den]].sum()
            rates.append((g[num] / g[den].replace(0, np.nan)).rename(parity))
            enough.append(g[den] >= min_denom)
        both = pd.concat(rates, axis=1).dropna()
        keep = enough[0].reindex(both.index).fillna(False) & enough[1].reindex(
            both.index
        ).fillna(False)
        both = both[keep.to_numpy()]
        if len(both) < 40:
            print(f"  {name:<22}{len(both):>5}{'thin':>13}{'':>14}")
            continue
        r = float(np.corrcoef(both[0], both[1])[0, 1])
        # Spearman-Brown: what the split-half r implies for a whole season.
        print(f"  {name:<22}{len(both):>5}{r:>13.2f}{2 * r / (1 + r):>14.2f}")


# --- prediction ------------------------------------------------------------
def rows(st: pd.DataFrame) -> pd.DataFrame:
    """One row per predicted start, features from his earlier starts only."""
    recs: list[dict[str, float]] = []
    for pid, g in st.groupby("pitcher"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            cur = g.iloc[i]
            past = g.iloc[:i]
            if len(past) < MIN_HISTORY:
                continue
            s6 = past.tail(6)
            if s6["pa"].sum() < 60 or s6["bbe"].sum() < MIN_BBE or s6["xw_n"].sum() < MIN_BBE:
                continue
            bbe6 = float(s6["bbe"].sum())
            rec = {
                "date": cur["date"],
                "pitcher": float(pid),
                "pa": float(cur["pa"]),
                "hr": float(cur["hr"]),
                "k_pct": float(s6["k"].sum() / s6["pa"].sum()),
                "csw": float(s6["csw_n"].sum() / s6["pitches"].sum()),
                "xwa": float(s6["xw_sum"].sum() / s6["xw_n"].sum()),
                "barrel": float(s6["barrels"].sum()) / bbe6,
                "hard": float(s6["hard"].sum()) / bbe6,
                "hrfb": float(s6["hr"].sum()) / max(float(s6["fb"].sum()), 1.0),
                "hr_prior": float(s6["hr"].sum() / s6["pa"].sum()),
            }
            ok = True
            for n in STARTS:
                s = past.tail(n)
                if s["bbe"].sum() < 5:
                    ok = False
                    break
                rec[f"fb{n}"] = float(s["fb"].sum() / s["bbe"].sum())
                rec[f"gb{n}"] = float((s["bbe"].sum() - s["air"].sum()) / s["bbe"].sum())
            for w in DAYS:
                s = past[past["date"] >= cur["date"] - pd.Timedelta(days=w)]
                if not len(s) or s["bbe"].sum() < 5:
                    ok = False
                    break
                rec[f"fbd{w}"] = float(s["fb"].sum() / s["bbe"].sum())
            if ok:
                recs.append(rec)
    return pd.DataFrame(recs).dropna().sort_values("date").reset_index(drop=True)


def _deviance(p: np.ndarray, made: np.ndarray, pa: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-2 * (made * np.log(p) + (pa - made) * np.log1p(-p)).sum() / pa.sum())


def _fit(x: np.ndarray, made: np.ndarray, pa: np.ndarray, iters: int = 80) -> np.ndarray:
    """Binomial IRLS: ``made`` successes out of ``pa`` trials per row."""
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


def _score(d: pd.DataFrame, cols: list[str]) -> tuple[float, float, float]:
    """Holdout deviance, plus the full-sample coefficient and t of the last term."""
    made = d["hr"].to_numpy(float)
    pa = d["pa"].to_numpy(float)

    def centred(name: str) -> np.ndarray:
        return (d[name] - d[name].mean()).to_numpy(float)

    parts = [np.ones(len(d)), centred("k_pct"), centred("csw"), centred("xwa")]
    parts += [centred(c) for c in cols]
    x = np.column_stack(parts)
    cut = int(len(d) * 0.6)
    beta = _fit(x[:cut], made[:cut], pa[:cut])
    ho = _deviance(1 / (1 + np.exp(-x[cut:] @ beta)), made[cut:], pa[cut:])
    if not cols:
        return ho, float("nan"), float("nan")
    full = _fit(x, made, pa)
    p = 1 / (1 + np.exp(-(x @ full)))
    w = pa * p * (1 - p)
    cov = np.linalg.inv(x.T @ (x * w[:, None]))
    i = x.shape[1] - 1
    return ho, float(full[i]), float(full[i] / np.sqrt(cov[i, i]))


def _header(d: pd.DataFrame) -> None:
    print(
        f"\n  {len(d):,} starts, {int(d['pa'].sum()):,} PA, {int(d['hr'].sum()):,} HR, "
        f"{d['date'].min()} to {d['date'].max()}; chronological 60/40 holdout\n"
    )


def terms(d: pd.DataFrame) -> None:
    """One contact read at a time, on top of the levels the engine prices."""
    _header(d)
    base, _, _ = _score(d, [])
    specs = {
        "fly-ball rate (4 starts)": ["fb4"],
        "fly-ball rate (6 weeks)": ["fbd42"],
        "hard-hit% allowed": ["hard"],
        "barrel% allowed (ships)": ["barrel"],
        "HR/FB% allowed": ["hrfb"],
        "his own HR per PA": ["hr_prior"],
        "fly-ball and barrel": ["barrel", "fb4"],
        "fly-ball and hard-hit": ["hard", "fb4"],
    }
    print(f"  {'added read':<26}{'coef':>9}{'t':>7}{'holdout':>11}{'vs base':>10}")
    print(f"  {'priced levels only':<26}{'--':>9}{'--':>7}{base:>11.5f}{'--':>10}")
    for name, cols in specs.items():
        ho, coef, t = _score(d, cols)
        print(f"  {name:<26}{coef:>9.3f}{t:>7.2f}{ho:>11.5f}{ho - base:>+10.5f}")


def window(d: pd.DataFrame) -> None:
    """How much of his fly-ball history to read, and whether it has an arrow."""
    _header(d)
    base, _, _ = _score(d, [])
    print(f"  {'fly-ball rate read over':<26}{'t':>7}{'holdout':>11}{'vs base':>10}")
    print(f"  {'no batted-ball read':<26}{'--':>7}{base:>11.5f}{'--':>10}")
    for n in STARTS:
        label = "season to date" if n == 99 else f"last {n} start{'s' if n > 1 else ''}"
        ho, _, t = _score(d, [f"fb{n}"])
        print(f"  {label:<26}{t:>7.2f}{ho:>11.5f}{ho - base:>+10.5f}")
    for w in DAYS:
        ho, _, t = _score(d, [f"fbd{w}"])
        print(f"  {str(w) + ' days':<26}{t:>7.2f}{ho:>11.5f}{ho - base:>+10.5f}")
    ho, _, t = _score(d, ["fb99", "fb1"])
    print(f"  {'season level + last start':<26}{t:>7.2f}{ho:>11.5f}{ho - base:>+10.5f}")
    ho, _, t = _score(d, ["gb4"])
    print(f"  {'(ground-ball rate, 4)':<26}{t:>7.2f}{ho:>11.5f}{ho - base:>+10.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("reliability", "window", "terms"))
    ap.add_argument("--cache", default=CACHE, help="glob of Statcast pickles")
    args = ap.parse_args()
    st = per_start(load(args.cache))
    print(f"{len(st):,} starts, {st['pitcher'].nunique()} pitchers")
    if args.mode == "reliability":
        reliability(st)
        return
    d = rows(st)
    if args.mode == "window":
        window(d)
    else:
        terms(d)


if __name__ == "__main__":
    main()
