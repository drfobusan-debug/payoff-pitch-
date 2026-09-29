"""Does a de-correlated totals model beat the sheet's additive SUM against the close?

Analysis only; nothing here touches the production totals sheet.  The study
rebuilds, as of each game date, one factor per source of information the sheet
uses -- starters, bullpens, lineups, park, umpire -- from the SP-study artifacts
(``~/.mlb_engine/audit/sp_study``: as-of starter/bullpen tables, finals, closing
game totals) and the season Statcast cache (team batting).  It then compares:

* ``sum_like``  -- the sheet's method rebuilt from the same inputs: every band
  column added (SP SIERA/xERA/CSW + K-BB, RP SIERA/xERA/CSW + K-BB, lineup wOBA
  + Barrel%, park, umpire).  Weather, book splits, fatigue and BsR are not
  available historically, so it is the sheet minus those columns.
* ``orth``      -- one continuous factor per source, each residualised on the
  closing total (so it only carries what the market has not priced), equal
  weight, fitted on prior dates only.
* ``ols``       -- runs minus close regressed on the close and the five factors,
  expanding window, predicting each date from the dates before it.

Every model is graded the same way: sign of the score vs the closing total,
out of sample, with the slate's own over rate as the yardstick.

Usage::

    .venv/bin/python -m scripts.totals_decorrelated_study
    .venv/bin/python -m scripts.totals_decorrelated_study --study-dir DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from mlb_engine.data.parks import get_park
from mlb_engine.output.totals_sheet import (
    UMP_BAND_RUNS,
    UMP_MIN_GAMES,
    barrel_pts,
    csw_pts,
    kbb_pts,
    park_pts,
    siera_pts,
    woba_pts,
    xera_pts,
)

log = logging.getLogger("totals_decorrelated_study")

STUDY_DIR = Path.home() / ".mlb_engine" / "audit" / "sp_study"
CACHE_DIR = Path.home() / ".mlb_engine" / "cache"
OUT_DIR = Path.home() / ".mlb_engine" / "audit" / "totals_decor"

ARM_METRICS = ["SIERA", "xERA", "CSW%", "K-BB%"]
FACTORS = ["sp", "pen", "off", "park", "ump"]
SHEET_COLS = [
    "sp_a", "sp_h", "kbb_sp_a", "kbb_sp_h", "rp_a", "rp_h", "kbb_rp_a", "kbb_rp_h",
    "off_a", "off_h", "park_pts", "ump_pts",
]
MIN_TRAIN = 150
LINE_LO, LINE_HI = 5.5, 12.5
BOOT = 2000


# --- inputs ------------------------------------------------------------------------


def closing_totals(closing_dir: Path) -> dict[tuple[str, str], float]:
    """(date, 'A @ H') -> the game-total line closed nearest even money."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for path in sorted(closing_dir.glob("closing_*.json")):
        day = path.stem.split("_", 1)[1]
        for q in json.loads(path.read_text()):
            if q.get("market") != "game_total" or not str(q.get("selection", "")).startswith("Over "):
                continue
            try:
                line = float(q["selection"].split()[1])
            except (IndexError, ValueError):
                continue
            if not LINE_LO <= line <= LINE_HI:  # in-play captures (e.g. Over 3.5 / 16.0)
                continue
            d = abs(float(q["no_vig_prob"]) - 0.5)
            key = (day, q["matchup"])
            if key not in out or d < out[key][0]:
                out[key] = (d, line)
    return {k: line for k, (_, line) in out.items()}


def team_batting_asof(pitches: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date team wOBA and Barrel% before each date: columns date, team, woba, barrel."""
    df = pitches[["game_date", "inning_topbot", "away_team", "home_team",
                  "woba_value", "woba_denom", "launch_speed_angle", "bb_type"]].copy()
    df["team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])
    df["bbe"] = df["bb_type"].notna().astype(int)
    df["barrel"] = (df["launch_speed_angle"].astype("float").fillna(0.0) == 6).astype(int)
    df["woba_value"] = pd.to_numeric(df["woba_value"], errors="coerce").fillna(0.0).astype(float)
    df["woba_denom"] = pd.to_numeric(df["woba_denom"], errors="coerce").fillna(0.0).astype(float)
    day = df.groupby(["team", "game_date"], as_index=False)[["woba_value", "woba_denom", "bbe", "barrel"]].sum()
    day = day.sort_values(["team", "game_date"])
    cum = day.groupby("team")[["woba_value", "woba_denom", "bbe", "barrel"]].cumsum()
    day["woba"] = cum["woba_value"] / cum["woba_denom"].replace(0, np.nan)
    day["barrel"] = 100 * cum["barrel"] / cum["bbe"].replace(0, np.nan)
    day["date"] = pd.to_datetime(day["game_date"])
    return day[["team", "date", "woba", "barrel"]]


def batting_before(table: pd.DataFrame, team: str, day: pd.Timestamp) -> tuple[float, float]:
    t = table[(table.team == team) & (table.date < day)]
    if t.empty:
        return math.nan, math.nan
    last = t.iloc[-1]
    return float(last.woba), float(last.barrel)


def umpire_asof(boxes: list[dict]) -> pd.DataFrame:
    """Per game: home-plate umpire's runs/game over his prior games minus league, and his n."""
    rows = sorted((b for b in boxes if b.get("hp") and b.get("runs") is not None), key=lambda b: b["date"])
    seen: dict[str, list[float]] = {}
    lg: list[float] = []
    out = []
    for b in rows:
        prior = seen.get(b["hp"], [])
        out.append({
            "game_pk": b["pk"],
            "ump_n": len(prior),
            "ump_diff": (np.mean(prior) - np.mean(lg)) if prior and lg else math.nan,
        })
        seen.setdefault(b["hp"], []).append(float(b["runs"]))
        lg.append(float(b["runs"]))
    return pd.DataFrame(out)


def ump_pts(n: int, diff: float) -> int:
    if n < UMP_MIN_GAMES or math.isnan(diff):
        return 0
    return 1 if diff >= UMP_BAND_RUNS else -1 if diff <= -UMP_BAND_RUNS else 0


# --- the frame ----------------------------------------------------------------------


def _arm_pts(row: pd.Series | None) -> tuple[int, int]:
    """Sheet points for one arm: (SIERA + xERA + CSW, K-BB); unknown arm = 0."""
    if row is None or pd.isna(row.get("SIERA")):
        return 0, 0
    pts = siera_pts(float(row["SIERA"]))
    if not pd.isna(row.get("CSW%")):
        pts += csw_pts(float(row["CSW%"]) * 100)
    if not pd.isna(row.get("xERA")):
        pts += xera_pts(float(row["xERA"]))
    kbb = row.get("K-BB%")
    return pts, kbb_pts(float(kbb) * 100) if not pd.isna(kbb) else 0


def _softness(row: pd.Series | None) -> float:
    """One number per arm: mean z over the four sheet metrics, sign flipped so soft > 0."""
    if row is None:
        return math.nan
    zs = [row.get(f"z_{m}") for m in ARM_METRICS]
    zs = [float(z) for z in zs if z is not None and not pd.isna(z)]
    return -float(np.mean(zs)) if zs else math.nan


def build_frame(
    games: pd.DataFrame,
    starters: pd.DataFrame,
    pens: pd.DataFrame,
    closes: dict[tuple[str, str], float],
    batting: pd.DataFrame,
    venues: dict[str, dict],
    umps: pd.DataFrame,
) -> pd.DataFrame:
    """One row per priced, non-doubleheader game with both sides' factors as of the date."""
    sp_idx = starters.set_index(["date", "pitcher"])
    bp_idx = pens.set_index(["date", "team"])
    ump_idx = umps.set_index("game_pk") if not umps.empty else None
    rows = []
    for g in games.itertuples(index=False):
        if g.doubleheader:
            continue
        label = f"{g.away} @ {g.home}"
        close = closes.get((g.date, label))
        if close is None:
            continue
        day = pd.Timestamp(g.date)
        sides = {}
        for side, team, sp_id in (("a", g.away, g.away_sp_id), ("h", g.home, g.home_sp_id)):
            sp = sp_idx.loc[(g.date, int(sp_id))] if (g.date, int(sp_id)) in sp_idx.index else None
            if isinstance(sp, pd.DataFrame):
                sp = sp.iloc[0]
            bp = bp_idx.loc[(g.date, team)] if (g.date, team) in bp_idx.index else None
            if isinstance(bp, pd.DataFrame):
                bp = bp.iloc[0]
            woba, barrel = batting_before(batting, team, day)
            sp_p, kbb_sp = _arm_pts(sp)
            rp_p, kbb_rp = _arm_pts(bp)
            sides[side] = {
                f"sp_{side}": sp_p, f"kbb_sp_{side}": kbb_sp,
                f"rp_{side}": rp_p, f"kbb_rp_{side}": kbb_rp,
                f"off_{side}": (woba_pts(woba) if not math.isnan(woba) else 0)
                + (barrel_pts(barrel) if not math.isnan(barrel) else 0),
                f"sp_soft_{side}": _softness(sp),
                f"pen_soft_{side}": _softness(bp),
                f"woba_{side}": woba, f"barrel_{side}": barrel,
            }
        v = venues.get(str(g.game_pk), {})
        park = get_park(int(v["venue_id"])) if v.get("venue_id") else None
        pf = park.park_factor if park else 100.0
        u = ump_idx.loc[int(g.game_pk)] if ump_idx is not None and int(g.game_pk) in ump_idx.index else None
        ump_n = int(u.ump_n) if u is not None else 0
        ump_diff = float(u.ump_diff) if u is not None else math.nan
        runs = int(g.away_runs) + int(g.home_runs)
        rows.append({
            "date": g.date, "game_pk": int(g.game_pk), "game": label,
            "close": float(close), "runs": runs, "resid": runs - float(close),
            "result": "over" if runs > close else "under" if runs < close else "push",
            "park_factor": pf, "roof": park.roof if park else "open",
            "park_pts": park_pts(pf), "ump_pts": ump_pts(ump_n, ump_diff),
            "ump_diff": 0.0 if math.isnan(ump_diff) or ump_n < UMP_MIN_GAMES else ump_diff,
            **sides["a"], **sides["h"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # a game with neither starter qualified has no starter factor at all
    df = df[df[["sp_soft_a", "sp_soft_h"]].notna().any(axis=1)].copy()
    df["sum_like"] = df[SHEET_COLS].sum(axis=1)
    # one continuous factor per source, before any market adjustment
    df["sp"] = df[["sp_soft_a", "sp_soft_h"]].mean(axis=1)
    df["pen"] = df[["pen_soft_a", "pen_soft_h"]].mean(axis=1)
    df["off"] = _lineup_factor(df)
    df["park"] = (df["park_factor"] - 100.0) / 3.0
    df["ump"] = df["ump_diff"]
    return df.sort_values(["date", "game_pk"]).reset_index(drop=True)


def _lineup_factor(df: pd.DataFrame) -> pd.Series:
    """Mean over both lineups of z(wOBA) + z(Barrel%), z taken across the frame."""
    parts = []
    for col in ("woba", "barrel"):
        both = pd.concat([df[f"{col}_a"], df[f"{col}_h"]])
        mu, sd = both.mean(), both.std(ddof=0)
        if not sd or math.isnan(sd):
            parts.append(pd.Series(0.0, index=df.index))
            continue
        parts.append(((df[f"{col}_a"] - mu) / sd + (df[f"{col}_h"] - mu) / sd) / 2)
    return sum(parts).fillna(0.0) / len(parts)


# --- analysis -----------------------------------------------------------------------


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(beta, se, r2) for y ~ X (X already has its constant column)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = float(e @ e) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tss = float(((y - y.mean()) ** 2).sum())
    return beta, se, 1 - float(e @ e) / tss if tss else 0.0


def ols_table(y: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    Xm = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    beta, se, r2 = _ols(y, Xm)
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = 2 * sps.t.sf(np.abs(t), df=max(len(y) - Xm.shape[1], 1))
    out = pd.DataFrame({"term": ["const", *X.columns], "coef": beta, "se": se, "t": t, "p": p})
    out.attrs["r2"] = r2
    return out


def redundancy(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Correlation matrix of the sheet's columns and the effective number of independent ones."""
    cols = [c for c in SHEET_COLS if df[c].std(ddof=0) > 0]
    corr = df[cols].corr()
    eig = np.linalg.eigvalsh(corr.to_numpy())
    eig = np.clip(eig, 0, None)
    n_eff = float(eig.sum() ** 2 / (eig**2).sum()) if eig.sum() else float(len(cols))
    return corr, n_eff


def factor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per factor: r with the close, r with the residual, and its partial r once the close is held fixed."""
    rows = []
    for f in ["sum_like", *FACTORS]:
        x = df[f].astype(float)
        if x.std(ddof=0) == 0:
            rows.append({"factor": f, "r_close": 0.0, "r_resid": 0.0, "partial_r_resid": 0.0, "p_partial": 1.0})
            continue
        r_close = float(np.corrcoef(x, df["close"])[0, 1])
        r_resid = float(np.corrcoef(x, df["resid"])[0, 1])
        Xc = np.column_stack([np.ones(len(df)), df["close"]])
        bx, *_ = np.linalg.lstsq(Xc, x, rcond=None)
        by, *_ = np.linalg.lstsq(Xc, df["resid"], rcond=None)
        ex, ey = x - Xc @ bx, df["resid"] - Xc @ by
        pr = float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else 0.0
        n = len(df)
        t = pr * math.sqrt(max(n - 3, 1) / max(1 - pr**2, 1e-12))
        rows.append({"factor": f, "r_close": r_close, "r_resid": r_resid,
                     "partial_r_resid": pr, "p_partial": 2 * sps.t.sf(abs(t), df=max(n - 3, 1))})
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Out-of-sample scores for every game after the first ``min_train``: ols and orth."""
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
    ols_score = np.full(len(df), np.nan)
    orth_score = np.full(len(df), np.nan)
    for day in df["date"].unique():
        test = df.index[df["date"] == day]
        train = df.index[df["date"] < day]
        if len(train) < min_train:
            continue
        tr, te = df.loc[train], df.loc[test]
        X_tr = np.column_stack([np.ones(len(tr)), tr["close"], tr[FACTORS]])
        X_te = np.column_stack([np.ones(len(te)), te["close"], te[FACTORS]])
        beta, *_ = np.linalg.lstsq(X_tr, tr["resid"].to_numpy(dtype=float), rcond=None)
        ols_score[test] = X_te @ beta
        # equal-weight sum of factors orthogonalised on the close
        Xc_tr = np.column_stack([np.ones(len(tr)), tr["close"]])
        Xc_te = np.column_stack([np.ones(len(te)), te["close"]])
        total = np.zeros(len(te))
        for f in FACTORS:
            b, *_ = np.linalg.lstsq(Xc_tr, tr[f].to_numpy(dtype=float), rcond=None)
            e_tr = tr[f].to_numpy(dtype=float) - Xc_tr @ b
            sd = e_tr.std()
            if sd <= 0:
                continue
            sign = np.sign(np.corrcoef(e_tr, tr["resid"])[0, 1]) or 1.0
            total += sign * (te[f].to_numpy(dtype=float) - Xc_te @ b) / sd
        orth_score[test] = total
    out = df.copy()
    out["ols"] = ols_score
    out["orth"] = orth_score
    return out


def _wilson(h: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    p = h / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - half, c + half


def grade_sign(df: pd.DataFrame, score: str, threshold: float = 0.0) -> dict:
    """Sign-of-score record on decided games with |score| > threshold, plus the slate's over rate."""
    d = df[df[score].notna() & (df[score].abs() > threshold) & (df["result"] != "push")]
    lean = np.where(d[score] > 0, "over", "under")
    hits = int((lean == d["result"]).sum())
    n = len(d)
    lo, hi = _wilson(hits, n)
    overs = int((d["result"] == "over").sum())
    return {"score": score, "threshold": threshold, "n": n, "hits": hits, "misses": n - hits,
            "rate": hits / n if n else math.nan, "ci_lo": lo, "ci_hi": hi,
            "over_leans": int((lean == "over").sum()),
            "over_lean_rate": float(((lean == "over") & (d["result"] == "over")).sum() / max((lean == "over").sum(), 1)),
            "under_lean_rate": float(((lean == "under") & (d["result"] == "under")).sum() / max((lean == "under").sum(), 1)),
            "slate_over_rate": overs / n if n else math.nan,
            "r_resid": float(np.corrcoef(d[score], d["resid"])[0, 1]) if n > 2 and d[score].std() > 0 else 0.0}


def band_table(df: pd.DataFrame, score: str, edges: list[float]) -> pd.DataFrame:
    """Sign-of-score record by |score| band (edges are the lower bounds)."""
    d = df[df[score].notna() & (df["result"] != "push")]
    rows = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else math.inf
        b = d[(d[score].abs() >= lo) & (d[score].abs() < hi)]
        if b.empty:
            continue
        lean = np.where(b[score] > 0, "over", "under")
        hits = int((lean == b["result"]).sum())
        rows.append({"score": score, "band": f"{lo:g}..{hi:g}" if hi != math.inf else f">={lo:g}",
                     "n": len(b), "hits": hits, "rate": hits / len(b),
                     "mean_close": float(b["close"].mean()), "mean_runs": float(b["runs"].mean()),
                     "over_rate": float((b["result"] == "over").mean())})
    return pd.DataFrame(rows)


def paired_boot(df: pd.DataFrame, a: str, b: str, boot: int = BOOT, seed: int = 7) -> tuple[float, float, float]:
    """Difference in sign hit rate (a minus b) with a paired bootstrap CI over games."""
    d = df[df[a].notna() & df[b].notna() & (df["result"] != "push")]
    ha = (np.where(d[a] > 0, "over", "under") == d["result"]).astype(float).to_numpy()
    hb = (np.where(d[b] > 0, "over", "under") == d["result"]).astype(float).to_numpy()
    diff = ha - hb
    if len(diff) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- report -------------------------------------------------------------------------


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def ledger_overlap(wf: pd.DataFrame, ledger: pd.DataFrame, boot: int = BOOT) -> tuple[pd.DataFrame, dict[str, tuple[float, float, float]]]:
    """Grade the production sheet's own SUM against orth/ols on the games both have scored."""
    led = ledger[ledger["result"].isin(["over", "under", "push"])][["date", "game", "sum_pts"]]
    m = wf.merge(led, on=["date", "game"]).rename(columns={"sum_pts": "sheet_sum"})
    m = m[m["ols"].notna()]
    grades = pd.DataFrame([grade_sign(m, s) for s in ("sheet_sum", "sum_like", "orth", "ols")])
    diffs = {s: paired_boot(m, s, "sheet_sum", boot) for s in ("orth", "ols")}
    return grades, diffs


def write_report(frame: pd.DataFrame, out: Path, boot: int = BOOT, ledger: pd.DataFrame | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    corr, n_eff = redundancy(frame)
    corr.to_csv(out / "sheet_column_correlations.csv")
    factors = factor_table(frame)
    factors.to_csv(out / "factor_table.csv", index=False)
    full = ols_table(frame["resid"].to_numpy(dtype=float), frame[["close", *FACTORS]])
    full.to_csv(out / "ols_in_sample.csv", index=False)
    wf = walk_forward(frame)
    wf.to_csv(out / "games_scored.csv", index=False)
    grades = pd.DataFrame([grade_sign(wf, s) for s in ("sum_like", "orth", "ols")]
                          + [grade_sign(wf, "ols", 0.5), grade_sign(wf, "orth", 1.0), grade_sign(wf, "sum_like", 4)])
    grades.to_csv(out / "sign_records.csv", index=False)
    bands = pd.concat([
        band_table(wf, "sum_like", [0, 1, 5, 10, 15]),
        band_table(wf, "orth", [0, 0.5, 1.0, 2.0]),
        band_table(wf, "ols", [0, 0.25, 0.5, 1.0]),
    ], ignore_index=True)
    bands.to_csv(out / "bands.csv", index=False)
    oos = wf[wf["ols"].notna()]
    d_orth = paired_boot(oos, "orth", "sum_like", boot)
    d_ols = paired_boot(oos, "ols", "sum_like", boot)

    lines = [
        "# Totals sheet: additive SUM vs a de-correlated model",
        "",
        f"Window {frame['date'].min()} to {frame['date'].max()}, {len(frame)} priced games "
        f"(single-admission, closing total on record). Out-of-sample section starts after the first "
        f"{MIN_TRAIN} games: {len(oos)} games, {oos['date'].min()} to {oos['date'].max()}.",
        "",
        "Inputs rebuilt as of each date: starter and bullpen SIERA / xERA / CSW% / K-BB% (Statcast, "
        "same tables as the SP study), lineup wOBA and Barrel% to date (Statcast), park factor, home-plate "
        "umpire runs/game. Not available historically, so not in any model here: weather, Circa/DK "
        "splits, fatigue, BsR. `sum_like` is therefore the sheet minus those columns.",
        "",
        "## 1. How redundant are the sheet's columns?",
        "",
        f"Effective number of independent columns among the {len(corr)} band columns: **{n_eff:.1f}** "
        f"(eigenvalue participation ratio; {len(corr)} would mean no overlap).",
        "",
        "Largest pairwise correlations:",
        "",
    ]
    pairs = []
    for i, a in enumerate(corr.columns):
        for b_ in corr.columns[i + 1:]:
            pairs.append((abs(corr.loc[a, b_]), a, b_, corr.loc[a, b_]))
    for _, a, b_, r in sorted(pairs, reverse=True)[:8]:
        lines.append(f"- {a} vs {b_}: r = {r:+.2f}")
    lines += [
        "",
        "## 2. What does each factor carry once the close is held fixed?",
        "",
        "| factor | r(close) | r(runs - close) | partial r vs residual | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in factors.itertuples(index=False):
        lines.append(f"| {r.factor} | {r.r_close:+.3f} | {r.r_resid:+.3f} | {r.partial_r_resid:+.3f} | {r.p_partial:.3f} |")
    lines += [
        "",
        f"In-sample OLS of runs - close on the close and the five factors: R^2 = {full.attrs['r2']:.4f}.",
        "",
        "| term | coef | se | t | p |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in full.itertuples(index=False):
        lines.append(f"| {r.term} | {r.coef:+.3f} | {r.se:.3f} | {r.t:+.2f} | {r.p:.3f} |")
    lines += [
        "",
        "## 3. Out of sample: sign of the score vs the closing total",
        "",
        "Each model scored every game from the dates before it only. `orth` = equal-weight sum of the five "
        "factors after each is residualised on the close; `ols` = fitted residual. Slate over rate is the "
        "yardstick: a model that only says 'Over' scores it.",
        "",
        "| model | filter | n | record | rate | 95% CI | Over leans | Over-lean rate | Under-lean rate | slate over | r(score, resid) |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for g in grades.itertuples(index=False):
        filt = f"abs > {g.threshold:g}" if g.threshold else "all"
        lines.append(
            f"| {g.score} | {filt} | {g.n} | {g.hits}-{g.misses} | {_pct(g.rate)} | {_pct(g.ci_lo)}-{_pct(g.ci_hi)} "
            f"| {g.over_leans} | {_pct(g.over_lean_rate)} | {_pct(g.under_lean_rate)} | {_pct(g.slate_over_rate)} | {g.r_resid:+.3f} |"
        )
    lines += [
        "",
        f"Paired bootstrap ({boot} resamples), hit rate minus `sum_like` on the same games: "
        f"orth {d_orth[0]:+.3f} [{d_orth[1]:+.3f}, {d_orth[2]:+.3f}]; ols {d_ols[0]:+.3f} [{d_ols[1]:+.3f}, {d_ols[2]:+.3f}].",
        "",
        "## 4. Does the extreme end hold up?",
        "",
        "| model | abs band | n | record | rate | mean close | mean runs | over rate |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for b in bands.itertuples(index=False):
        lines.append(f"| {b.score} | {b.band} | {b.n} | {b.hits}-{b.n - b.hits} | {_pct(b.rate)} | {b.mean_close:.2f} | {b.mean_runs:.2f} | {_pct(b.over_rate)} |")
    if ledger is not None:
        lg, ld = ledger_overlap(wf, ledger, boot)
        lines += [
            "",
            "## 4b. Same games as the production ledger",
            "",
            f"{int(lg['n'].max())} games are both in the sheet's ledger (its real SUM, weather and splits included) and "
            "in the out-of-sample section here.",
            "",
            "| model | n | record | rate | 95% CI | Over leans | slate over |",
            "|---|---:|---|---:|---|---:|---:|",
        ]
        for g in lg.itertuples(index=False):
            lines.append(f"| {g.score} | {g.n} | {g.hits}-{g.misses} | {_pct(g.rate)} | {_pct(g.ci_lo)}-{_pct(g.ci_hi)} | {g.over_leans} | {_pct(g.slate_over_rate)} |")
        lines.append("")
        lines.append("Paired difference vs the sheet's SUM: " + "; ".join(
            f"{s} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" for s, d in ld.items()) + ".")
    lines += ["", "## 5. Reading", ""]
    lines += _reading(frame, factors, full, grades, bands, n_eff, d_orth, d_ols)
    path = out / "totals_decorrelated_study.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _reading(frame: pd.DataFrame, factors: pd.DataFrame, full: pd.DataFrame, grades: pd.DataFrame,
             bands: pd.DataFrame, n_eff: float, d_orth: tuple, d_ols: tuple) -> list[str]:
    f = factors.set_index("factor")
    g = grades[grades["threshold"] == 0].set_index("score")
    sig = [t for t in full.itertuples(index=False) if t.term not in ("const", "close") and t.p < 0.05]
    out = [
        f"- The sheet's {len(SHEET_COLS)} band columns behave like about {n_eff:.0f} independent inputs; the rest is the "
        "same starter or lineup counted through several correlated metrics.",
        f"- `sum_like` correlates {f.loc['sum_like', 'r_close']:+.2f} with the close and "
        f"{f.loc['sum_like', 'partial_r_resid']:+.3f} with the result once the close is held fixed "
        f"(p = {f.loc['sum_like', 'p_partial']:.2f}).",
    ]
    if sig:
        out.append("- Factors with p < 0.05 against the residual in sample: "
                   + ", ".join(f"{t.term} ({t.coef:+.2f} runs per unit)" for t in sig) + ".")
    else:
        out.append("- No single factor is significant against the residual in sample once the close is in the model.")
    out.append(
        f"- Out of sample: sum_like {_pct(g.loc['sum_like', 'rate'])}, orth {_pct(g.loc['orth', 'rate'])}, "
        f"ols {_pct(g.loc['ols', 'rate'])} against a slate over rate of {_pct(g.loc['sum_like', 'slate_over_rate'])}; "
        f"the paired difference vs sum_like is orth {d_orth[0]:+.1%} [{d_orth[1]:+.1%}, {d_orth[2]:+.1%}], "
        f"ols {d_ols[0]:+.1%} [{d_ols[1]:+.1%}, {d_ols[2]:+.1%}]."
    )
    top = bands[(bands["score"] == "sum_like")].tail(2)
    if not top.empty:
        out.append("- sum_like's biggest bands: " + "; ".join(
            f"{b.band}: {b.hits}-{b.n - b.hits} ({_pct(b.rate)}) on a mean close of {b.mean_close:.2f}" for b in top.itertuples(index=False)) + ".")
    ci_ok = [d for d in (("orth", d_orth), ("ols", d_ols)) if d[1][1] > 0]
    if ci_ok:
        out.append("- Verdict: " + " and ".join(d[0] for d in ci_ok) + " beat the additive SUM with a CI that excludes zero.")
    else:
        out.append("- Verdict: neither de-correlated version beats the additive SUM by a margin whose CI excludes zero on this "
                   "window; de-correlating removes the double counting but does not create information the close lacks.")
    return out


# --- driver -------------------------------------------------------------------------


def load_inputs(study_dir: Path, cache_dir: Path) -> pd.DataFrame:
    games = pd.read_csv(study_dir / "games.csv")
    starters = pd.read_csv(study_dir / "starter_scores.csv")
    pens = pd.read_csv(study_dir / "bullpen_scores.csv")
    closes = closing_totals(study_dir / "prices" / "closing")
    dates = sorted({d for d, _ in closes})
    if not dates:
        raise SystemExit(f"no closing totals under {study_dir / 'prices' / 'closing'}")
    games = games[(games.date >= dates[0]) & (games.date <= dates[-1])]
    season = int(dates[0][:4])
    log.info("loading Statcast %d for team batting", season)
    pitches = pd.concat(
        [pd.read_pickle(p) for p in sorted(cache_dir.glob(f"statcast_{season}-*.pkl"))],
        ignore_index=True,
    )
    batting = team_batting_asof(pitches)
    venues_path = study_dir / "venues.json"
    venues = json.loads(venues_path.read_text()) if venues_path.exists() else {}
    box_path = cache_dir / f"totals_boxscores_{season}.json"
    umps = umpire_asof(json.loads(box_path.read_text())) if box_path.exists() else pd.DataFrame(columns=["game_pk", "ump_n", "ump_diff"])
    return build_frame(games, starters, pens, closes, batting, venues, umps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study-dir", type=Path, default=STUDY_DIR)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".mlb_engine" / "audit" / "totals_ledger.csv",
                    help="production totals ledger; overlap section is skipped when absent")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = load_inputs(args.study_dir, args.cache_dir)
    log.info("%d priced games in the frame", len(frame))
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "frame.csv", index=False)
    ledger = pd.read_csv(args.ledger) if args.ledger.exists() else None
    path = write_report(frame, args.out, args.boot, ledger)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
