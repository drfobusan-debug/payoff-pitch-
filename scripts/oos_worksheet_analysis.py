"""Out-of-sample test of the daily worksheet's metric scores against pre-game prices.

Joins the per-game scores from ``oos_worksheet_build.py`` to the Odds API
pre-game prices from ``oos_odds_history.py`` and runs, with 2,000-draw
game-level bootstrap intervals:

1. per-metric tables (11 SP, 11 BP, totals, whiff/contact families), pooled and
   per stratum: side accuracy vs no-vig implied, residual, ML/RL units, r(win),
   r(run diff), r(runs innings 1-5 / 6+);
2. the pre-registered 2026 leaners as out-of-sample hypotheses with Holm
   correction;
3. the whiff-vs-contact family split;
4. a market-controlled logit per metric (outcome ~ gap + market logit);
5. rolling 400-game residuals for the top leaners;
6. (price-quality check against closing prices, when a closing file is given);
7. the n needed at 80% power to confirm each pooled residual.

Plus the batting "bat vs hand" add-on (H-V0..H-V6) as a separate Holm family.

    .venv/bin/python scripts/oos_worksheet_analysis.py \
        --games ~/.mlb_engine/audit/oos_games_2024.csv ~/.mlb_engine/audit/oos_games_2025.csv \
        --prices ~/.mlb_engine/audit/oos_prices.csv --out-dir ~/.mlb_engine/audit/oos
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.output.daily_worksheet import BAT_COLS, BP_COLS, SP_COLS

N_BOOT = 2000
SEED = 20260923
SP_LABELS = [c[0] for c in SP_COLS]
BP_LABELS = [c[0] for c in BP_COLS]
BAT_LABELS = [c[0] for c in BAT_COLS]
WHIFF = {"K%", "K-BB%", "CSW%", "Stuff+", "O-Swing%", "SqUp Con%"}
CONTACT = {"xERA", "xFIP", "SIERA", "HardHit%", "FB%", "Barrel%"}
EARLY_LATE_2026 = pd.Timestamp("2026-07-19")
PRICE_BUCKETS = (("dog +150+", 150, math.inf), ("dog +100..+150", 100, 150),
                 ("fav -100..-150", -150, -100), ("fav -150+", -math.inf, -150))


# --- helpers ---------------------------------------------------------------------------


def american_to_prob(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    return np.where(p > 0, 100.0 / (p + 100.0), -p / (-p + 100.0))


def american_profit(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    return np.where(p > 0, p / 100.0, 100.0 / -p)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def boot_ci(x: np.ndarray, stat: str = "mean", rng: np.random.Generator | None = None,
            n_boot: int = N_BOOT) -> tuple[float, float, float, float]:
    """(point, lo, hi, two-sided bootstrap p that the statistic is zero)."""
    rng = rng or np.random.default_rng(SEED)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return math.nan, math.nan, math.nan, math.nan
    f = np.mean if stat == "mean" else np.sum
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = f(x[idx], axis=1)
    point = float(f(x))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return point, float(lo), float(hi), float(max(p, 1 / n_boot))


def boot_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Bootstrap CI / p for mean(a) - mean(b), independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return math.nan, math.nan, math.nan, math.nan
    da = a[rng.integers(0, len(a), (N_BOOT, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (N_BOOT, len(b)))].mean(1)
    d = da - db
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(a.mean() - b.mean()), float(lo), float(hi), float(max(p, 1 / N_BOOT))


def power_n(effect: float, sd: float, power_z: float = 0.8416, alpha_z: float = 1.96) -> float:
    if not effect or math.isnan(effect) or math.isnan(sd):
        return math.nan
    return float(((alpha_z + power_z) * sd / abs(effect)) ** 2)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: (math.isnan(pvals[i]), pvals[i]))
    out = [math.nan] * m
    running = 0.0
    for rank, i in enumerate(order):
        if math.isnan(pvals[i]):
            continue
        running = max(running, (m - rank) * pvals[i])
        out[i] = min(1.0, running)
    return out


def logistic_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Newton-Raphson logistic regression: (beta, se)."""
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se


def verdict(sign_expected: int, point: float, lo: float, hi: float, n: int, needed: float) -> str:
    if math.isnan(point):
        return "not testable"
    same = (point > 0) == (sign_expected > 0)
    if same and (lo > 0 or hi < 0):
        return "confirmed OOS"
    if same:
        return "sign replicates but underpowered" if n < needed else "does not replicate"
    if (lo > 0 or hi < 0):
        return "reversed"
    return "does not replicate"


# --- data ------------------------------------------------------------------------------


def load_games(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["commence_ts"] = pd.to_datetime(df["commence"], utc=True)
    df["stratum"] = df["date"].dt.year.astype(str)
    is26 = df["date"].dt.year.eq(2026)
    df.loc[is26 & (df["date"] < EARLY_LATE_2026), "stratum"] = "2026-early"
    df.loc[is26 & (df["date"] >= EARLY_LATE_2026), "stratum"] = "2026-late"
    return df


def load_prices(path: Path) -> pd.DataFrame:
    px = pd.read_csv(path)
    px["commence_ts"] = pd.to_datetime(px["commence"], utc=True)
    px["price_date"] = (px["commence_ts"] - pd.Timedelta(hours=5)).dt.tz_localize(None).dt.normalize()
    return px


def merge_prices(games: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Join on (home, away, local date), then keep the price row whose first pitch is
    nearest the schedule's (so doubleheader games each take their own row) and
    within 3 hours of it."""
    m = games.merge(px.drop(columns=["commence"]).rename(columns={"commence_ts": "px_commence"}),
                    left_on=["home", "away", "date"], right_on=["home", "away", "price_date"], how="inner")
    m["_dt"] = (m["px_commence"] - m["commence_ts"]).abs()
    m = m[m["_dt"] <= pd.Timedelta(hours=3)].sort_values("_dt")
    m = m.drop_duplicates(subset=["game_pk"])
    m = m.drop_duplicates(subset=["home", "away", "px_commence"])
    df = m.drop(columns=["_dt"]).sort_values("commence_ts").reset_index(drop=True).copy()
    df["home_win"] = (df["home_runs"] > df["away_runs"]).astype(float)
    df["run_diff"] = df["home_runs"] - df["away_runs"]
    df["rdiff15"] = df["home_r15"] - df["away_r15"]
    df["rdiff6"] = df["home_r6"] - df["away_r6"]
    df["p_home"] = df["p_home_book"]
    df["mkt_logit"] = logit(df["p_home"].to_numpy())
    if "rl_home" in df:
        ph = american_to_prob(df["rl_home"].fillna(-110))
        pa = american_to_prob(df["rl_away"].fillna(-110))
        df["p_home_cover"] = ph / (ph + pa)
        df["home_cover"] = np.sign(df["run_diff"] + df["rl_home_pt"].fillna(-1.5))
    return df


# --- metric evaluation ------------------------------------------------------------------


@dataclass
class SideBets:
    """Every game where the metric took a side, from that side's point of view."""

    n: int
    win: np.ndarray
    implied: np.ndarray
    ml_profit: np.ndarray
    cover: np.ndarray
    cover_implied: np.ndarray
    rl_profit: np.ndarray
    gap: np.ndarray
    home_win: np.ndarray
    run_diff: np.ndarray
    rdiff15: np.ndarray
    rdiff6: np.ndarray
    frame: pd.DataFrame


def side_bets(df: pd.DataFrame, gap: pd.Series) -> SideBets:
    d = df.loc[gap.notna() & gap.ne(0)].copy()
    g = gap.loc[d.index]
    home_side = g > 0
    d["side_home"] = home_side
    d["gap"] = g
    win = np.where(home_side, d["home_win"], 1 - d["home_win"])
    implied = np.where(home_side, d["p_home"], 1 - d["p_home"])
    price = np.where(home_side, d["ml_home"], d["ml_away"])
    ml_profit = np.where(win == 1, american_profit(pd.Series(price)), -1.0)
    d["side_price"] = price
    d["win"] = win
    d["implied"] = implied
    d["ml_profit"] = ml_profit
    if "rl_home" in d and d["rl_home"].notna().any():
        has = d["rl_home"].notna() & d["rl_away"].notna()
        cover_home = d["home_cover"].to_numpy()
        cover = np.where(home_side, cover_home, -cover_home).astype(float)
        cover = np.where(cover == 0, np.nan, (cover > 0).astype(float))
        rl_price = np.where(home_side, d["rl_home"], d["rl_away"])
        rl_profit = np.where(np.isnan(cover), 0.0, np.where(cover == 1, american_profit(pd.Series(rl_price)), -1.0))
        cover_implied = np.where(home_side, d["p_home_cover"], 1 - d["p_home_cover"])
        cover = np.where(has, cover, np.nan)
        rl_profit = np.where(has, rl_profit, np.nan)
        cover_implied = np.where(has, cover_implied, np.nan)
    else:
        cover = rl_profit = cover_implied = np.full(len(d), np.nan)
    d["cover"] = cover
    d["rl_profit"] = rl_profit
    d["cover_implied"] = cover_implied
    return SideBets(len(d), win.astype(float), implied.astype(float), ml_profit.astype(float), cover,
                    cover_implied, rl_profit, g.to_numpy(float), d["home_win"].to_numpy(float),
                    d["run_diff"].to_numpy(float), d["rdiff15"].to_numpy(float), d["rdiff6"].to_numpy(float), d)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return math.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def evaluate(df: pd.DataFrame, gap: pd.Series, rng: np.random.Generator) -> dict:
    sb = side_bets(df, gap)
    if sb.n < 20:
        return {"n": sb.n}
    resid = sb.win - sb.implied
    r_pt, r_lo, r_hi, r_p = boot_ci(resid * 100, "mean", rng)
    u_pt, u_lo, u_hi, u_p = boot_ci(sb.ml_profit, "sum", rng)
    rl_ok = ~np.isnan(sb.cover)
    rl_resid = (sb.cover - sb.cover_implied)[rl_ok] * 100
    rlr_pt, rlr_lo, rlr_hi, rlr_p = boot_ci(rl_resid, "mean", rng)
    rlu_pt, rlu_lo, rlu_hi, rlu_p = boot_ci(sb.rl_profit[rl_ok], "sum", rng)
    # market-controlled logit, home perspective
    X = np.column_stack([np.ones(sb.n), sb.gap, logit(df.loc[sb.frame.index, "p_home"].to_numpy())])
    try:
        beta, se = logistic_fit(X, sb.home_win)
        b, s = float(beta[1]), float(se[1])
    except np.linalg.LinAlgError:
        b, s = math.nan, math.nan
    return {
        "n": sb.n, "side_win_pct": 100 * sb.win.mean(), "implied_pct": 100 * sb.implied.mean(),
        "resid_pts": r_pt, "resid_lo": r_lo, "resid_hi": r_hi, "resid_p": r_p,
        "ml_units": u_pt, "ml_units_lo": u_lo, "ml_units_hi": u_hi, "ml_p": u_p,
        "ml_roi_pct": 100 * u_pt / sb.n,
        "rl_n": int(rl_ok.sum()), "rl_cover_pct": 100 * np.nanmean(sb.cover) if rl_ok.any() else math.nan,
        "rl_resid_pts": rlr_pt, "rl_resid_lo": rlr_lo, "rl_resid_hi": rlr_hi, "rl_resid_p": rlr_p,
        "rl_units": rlu_pt, "rl_units_lo": rlu_lo, "rl_units_hi": rlu_hi, "rl_p": rlu_p,
        "r_win": _corr(sb.gap, sb.home_win), "r_run_diff": _corr(sb.gap, sb.run_diff),
        "r_runs_1_5": _corr(sb.gap, sb.rdiff15), "r_runs_6plus": _corr(sb.gap, sb.rdiff6),
        "logit_beta": b, "logit_se": s, "logit_lo": b - 1.96 * s, "logit_hi": b + 1.96 * s,
        "logit_p": 2 * (1 - _norm_cdf(abs(b / s))) if s and not math.isnan(s) else math.nan,
        "resid_sd_pts": float(np.std(resid * 100)),
        "n_80pct_power": power_n(r_pt, float(np.std(resid * 100))),
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def gap_series(df: pd.DataFrame, prefix: str, label: str | None, kind: str) -> pd.Series:
    """Home minus away for one metric (or the total when label is None).

    kind: 'pts' (production compressed), 'dec' (decile 1..10 of percentile rank), 'z'.
    """
    if label is None:
        return df[f"home_{prefix}_total"] - df[f"away_{prefix}_total"]
    if kind == "pts":
        return df[f"home_{prefix}_{label}_pts"] - df[f"away_{prefix}_{label}_pts"]
    if kind == "dec":
        h = np.ceil(df[f"home_{prefix}_{label}_pct"] * 10).clip(1, 10)
        a = np.ceil(df[f"away_{prefix}_{label}_pct"] * 10).clip(1, 10)
        return h - a
    return df[f"home_{prefix}_{label}_z"] - df[f"away_{prefix}_{label}_z"]


def family_gap(df: pd.DataFrame, prefix: str, labels: list[str], fam: set[str], kind: str = "pts") -> pd.Series:
    cols = [label for label in labels if label in fam and f"home_{prefix}_{label}_pts" in df]
    return sum((gap_series(df, prefix, label, kind) for label in cols), pd.Series(0.0, index=df.index))


def metric_tables(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    strata = ["pooled", *sorted(df["stratum"].unique())]
    specs: list[tuple[str, str, str | None]] = []
    for label in SP_LABELS:
        specs.append(("sp", "SP", label))
    specs.append(("sp", "SP", None))
    for label in BP_LABELS:
        specs.append(("bp", "BP", label))
    specs.append(("bp", "BP", None))
    for st in strata:
        sub = df if st == "pooled" else df[df["stratum"].eq(st)]
        for prefix, group, lab in specs:
            for kind in ("pts", "dec", "z"):
                if lab is None and kind != "pts":
                    continue
                if f"home_{prefix}_total" not in sub:
                    continue
                g = gap_series(sub, prefix, lab, kind)
                res = evaluate(sub, g, rng)
                rows.append({"stratum": st, "group": group, "metric": lab or "TOTAL", "scoring": kind, **res})
        for prefix, group in (("sp", "SP"), ("bp", "BP")):
            labels = SP_LABELS if prefix == "sp" else BP_LABELS
            for fam_name, fam in (("Whiff", WHIFF), ("Contact", CONTACT)):
                g = family_gap(sub, prefix, labels, fam)
                res = evaluate(sub, g, rng)
                rows.append({"stratum": st, "group": group, "metric": f"FAMILY {fam_name}", "scoring": "pts", **res})
    return pd.DataFrame(rows)


# --- pre-registered hypotheses ---------------------------------------------------------


def band_eval(df: pd.DataFrame, lo: float, hi: float, rng: np.random.Generator) -> dict:
    g = gap_series(df, "sp", None, "pts")
    mask = g.abs().between(lo, hi)
    return evaluate(df[mask], g[mask], rng)


def f5_under(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Blind F5 Under when both starters are top-tercile Contact, against a proxy F5
    line derived from the full-game total (4.5 / 5 / 5.5 for totals <=8 / 8.5-9 / >=9.5)."""
    d = df[df["total_pt"].notna()].copy()
    contact = [label for label in SP_LABELS if label in CONTACT]
    for side in ("home", "away"):
        d[f"{side}_contact_pct"] = d[[f"{side}_sp_{c}_pct" for c in contact]].mean(axis=1)
    d = d[d["home_contact_pct"].notna() & d["away_contact_pct"].notna()]
    line = np.where(d["total_pt"] <= 8, 4.5, np.where(d["total_pt"] <= 9, 5.0, 5.5))
    f5 = d["home_r15"] + d["away_r15"]
    under = np.where(f5 < line, 1.0, np.where(f5 > line, 0.0, np.nan))
    both = (d["home_contact_pct"] >= 2 / 3) & (d["away_contact_pct"] >= 2 / 3)
    base = under[~np.isnan(under)]
    sel = under[both.to_numpy() & ~np.isnan(under)]
    pt, lo, hi, p = boot_diff(sel, base, rng)
    return {"n": int(len(sel)), "under_pct": 100 * sel.mean() if len(sel) else math.nan,
            "baseline_under_pct": 100 * base.mean(), "resid_pts": 100 * pt, "resid_lo": 100 * lo,
            "resid_hi": 100 * hi, "resid_p": p, "n_80pct_power": power_n(100 * pt, 100 * float(np.std(sel))) if len(sel) else math.nan}


def hypotheses(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []

    def add(hid: str, text: str, sign: int, res: dict, key: str = "resid_pts", extra: str = "") -> None:
        pt, lo, hi, p, n = (res.get(key, math.nan), res.get(f"{key}_lo" if key != "resid_pts" else "resid_lo", math.nan),
                            res.get(f"{key}_hi" if key != "resid_pts" else "resid_hi", math.nan),
                            res.get(f"{key.split('_')[0]}_p" if key != "resid_pts" else "resid_p", math.nan),
                            res.get("n", 0))
        need = res.get("n_80pct_power", math.nan)
        rows.append({"id": hid, "hypothesis": text, "expected_sign": sign, "n": n, "point": pt, "lo": lo, "hi": hi,
                     "raw_p": p, "ml_units": res.get("ml_units", math.nan), "rl_units": res.get("rl_units", math.nan),
                     "n_80pct_power": need, "verdict": verdict(sign, pt, lo, hi, n, need), "note": extra})

    add("H1", "BP HardHit% gap: ML residual > 0", +1, evaluate(df, gap_series(df, "bp", "HardHit%", "pts"), rng))
    add("H2", "BP O-Swing% gap: ML residual > 0", +1, evaluate(df, gap_series(df, "bp", "O-Swing%", "pts"), rng))
    add("H3", "BP xERA gap: ML residual > 0 (also RL)", +1, evaluate(df, gap_series(df, "bp", "xERA", "pts"), rng))
    add("H4", "BP K% gap: ML residual < 0 (fade strikeout pens)", -1, evaluate(df, gap_series(df, "bp", "K%", "pts"), rng))
    add("H5", "SP K-BB% gap: ML residual > 0", +1, evaluate(df, gap_series(df, "sp", "K-BB%", "pts"), rng))
    add("H6", "SP total gap 5-9: side residual > 0", +1, band_eval(df, 5, 9, rng),
        extra="SP total here sums 11 metrics incl. Stuff+ (FanGraphs)")
    add("H7", "Blind F5 Under, both starters top-tercile Contact (proxy F5 line)", +1, f5_under(df, rng))
    bp_tot = evaluate(df, gap_series(df, "bp", None, "pts"), rng)
    sp_tot = evaluate(df, gap_series(df, "sp", None, "pts"), rng)
    add("H8a", "BP total gap: RL units > 0", +1, bp_tot, key="rl_units")
    add("H8b", "SP total gap: RL units < 0", -1, sp_tot, key="rl_units")
    out = pd.DataFrame(rows)
    # Holm over the eight pre-registered hypotheses (H8 counted once: its worse half).
    fam = out[~out["id"].eq("H8b")].copy()
    h8 = out[out["id"].isin(["H8a", "H8b"])]["raw_p"].max()
    fam.loc[fam["id"].eq("H8a"), "raw_p"] = h8
    fam["holm_p"] = holm(fam["raw_p"].tolist())
    out = out.merge(fam[["id", "holm_p"]], on="id", how="left")
    out.loc[out["id"].eq("H8b"), "holm_p"] = out.loc[out["id"].eq("H8a"), "holm_p"].values
    # secondary leaners (not in the Holm family)
    sec = []
    for hid, prefix, label, sign, text in (
        ("S1", "bp", "CSW%", -1, "BP CSW% gap: ML residual < 0"),
        ("S2", "bp", "K-BB%", -1, "BP K-BB% gap: ML residual < 0"),
        ("S3", "sp", "xERA", +1, "SP xERA gap: ML residual > 0"),
    ):
        res = evaluate(df, gap_series(df, prefix, label, "pts"), rng)
        sec.append({"id": hid, "hypothesis": text, "expected_sign": sign, "n": res.get("n", 0),
                    "point": res.get("resid_pts", math.nan), "lo": res.get("resid_lo", math.nan),
                    "hi": res.get("resid_hi", math.nan), "raw_p": res.get("resid_p", math.nan),
                    "ml_units": res.get("ml_units", math.nan), "rl_units": res.get("rl_units", math.nan),
                    "n_80pct_power": res.get("n_80pct_power", math.nan),
                    "verdict": verdict(sign, res.get("resid_pts", math.nan), res.get("resid_lo", math.nan),
                                       res.get("resid_hi", math.nan), res.get("n", 0), res.get("n_80pct_power", math.nan)),
                    "note": "secondary, not Holm-corrected", "holm_p": math.nan})
    band10 = band_eval(df, 10, math.inf, rng)
    sec.append({"id": "S4", "hypothesis": "SP total gap >= 10: side residual < 0", "expected_sign": -1,
                "n": band10.get("n", 0), "point": band10.get("resid_pts", math.nan), "lo": band10.get("resid_lo", math.nan),
                "hi": band10.get("resid_hi", math.nan), "raw_p": band10.get("resid_p", math.nan),
                "ml_units": band10.get("ml_units", math.nan), "rl_units": band10.get("rl_units", math.nan),
                "n_80pct_power": band10.get("n_80pct_power", math.nan),
                "verdict": verdict(-1, band10.get("resid_pts", math.nan), band10.get("resid_lo", math.nan),
                                   band10.get("resid_hi", math.nan), band10.get("n", 0), band10.get("n_80pct_power", math.nan)),
                "note": "secondary, not Holm-corrected", "holm_p": math.nan})
    return pd.concat([out, pd.DataFrame(sec)], ignore_index=True)


# --- stability -------------------------------------------------------------------------


def rolling(df: pd.DataFrame, window: int = 400, step: int = 100) -> pd.DataFrame:
    rows = []
    for name, prefix, label in (("bp HardHit%", "bp", "HardHit%"), ("bp O-Swing%", "bp", "O-Swing%"),
                                ("bp xERA", "bp", "xERA"), ("bp K% (fade)", "bp", "K%")):
        sb = side_bets(df.sort_values("commence_ts"), gap_series(df, prefix, label, "pts"))
        fr = sb.frame.sort_values("commence_ts")
        resid = (fr["win"] - fr["implied"]).to_numpy() * 100
        if label == "K%":
            resid = -resid  # the fade: residual of betting against the side
        dates = fr["date"].to_numpy()
        for start in range(0, max(len(fr) - window + 1, 1), step):
            seg = resid[start:start + window]
            rows.append({"leaner": name, "window_start": pd.Timestamp(dates[start]).date(),
                         "window_end": pd.Timestamp(dates[min(start + window, len(fr)) - 1]).date(),
                         "n": len(seg), "resid_pts": seg.mean(), "se": seg.std() / math.sqrt(len(seg))})
    return pd.DataFrame(rows)


# --- batting add-on ----------------------------------------------------------------------


def _bat_decile_total(df: pd.DataFrame, side: str) -> pd.Series:
    cols = [f"{side}_bat_hand_{label}_pct" for label in BAT_LABELS if f"{side}_bat_hand_{label}_pct" in df]
    return sum((np.ceil(df[c] * 10).clip(1, 10) for c in cols), pd.Series(0.0, index=df.index))


def batting(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "home_bat_hand_total" not in df:
        return pd.DataFrame(), pd.DataFrame()
    d = df[df["home_bat_hand_total"].notna() & df["away_bat_hand_total"].notna()].copy()
    gap = d["home_bat_hand_total"] - d["away_bat_hand_total"]
    gap_dec = _bat_decile_total(d, "home") - _bat_decile_total(d, "away")
    tables = []
    for st in ["pooled", *sorted(d["stratum"].unique())]:
        sub = d if st == "pooled" else d[d["stratum"].eq(st)]
        for kind, g in (("pts", gap), ("dec", gap_dec)):
            tables.append({"stratum": st, "scoring": kind, **evaluate(sub, g.loc[sub.index], rng)})
    sb = side_bets(d, gap)
    fr = sb.frame
    fr["resid"] = fr["win"] - fr["implied"]
    fr["rl_resid"] = fr["cover"] - fr["cover_implied"]
    side_team = np.where(fr["side_home"], "home", "away")
    fr["side_pa60"] = [fr.at[i, f"{s}_bat_hand_pa60"] for i, s in zip(fr.index, side_team, strict=True)]
    fr["side_rank_gap"] = [abs(fr.at[i, f"{s}_bat_hand_rank"] - fr.at[i, f"{s}_bat_ovr_rank"]) for i, s in zip(fr.index, side_team, strict=True)]
    fr["side_opp_hand"] = [fr.at[i, f"{s}_opp_hand"] for i, s in zip(fr.index, side_team, strict=True)]
    fr["side_platoon"] = [abs(fr.at[i, f"{s}_bat_vsl_total"] - fr.at[i, f"{s}_bat_vsr_total"]) for i, s in zip(fr.index, side_team, strict=True)]
    fr["half"] = np.where(fr["date"].dt.month * 100 + fr["date"].dt.day <= 715, "1st half", "2nd half")
    q75 = fr["side_platoon"].quantile(0.75)

    for name, lo, hi in PRICE_BUCKETS:
        m = fr["side_price"].gt(lo) & fr["side_price"].le(hi) if lo > -math.inf else fr["side_price"].le(hi)
        if lo == 100:
            m = fr["side_price"].ge(100) & fr["side_price"].lt(150)
        elif lo == 150:
            m = fr["side_price"].ge(150)
        elif hi == -100:
            m = fr["side_price"].lt(0) & fr["side_price"].gt(-150)
        elif hi == -150:
            m = fr["side_price"].le(-150)
        tables.append({"stratum": f"price {name}", "scoring": "pts", **evaluate(d.loc[fr.index[m]], gap.loc[fr.index[m]], rng)})
    for half in ("1st half", "2nd half"):
        m = fr["half"].eq(half)
        tables.append({"stratum": half, "scoring": "pts", **evaluate(d.loc[fr.index[m]], gap.loc[fr.index[m]], rng)})

    hyp = []
    full = evaluate(d, gap, rng)

    def add(hid: str, text: str, pt: float, lo: float, hi: float, p: float, n: int, need: float, note: str = "") -> None:
        hyp.append({"id": hid, "hypothesis": text, "n": n, "point": pt, "lo": lo, "hi": hi, "raw_p": p,
                    "n_80pct_power": need, "verdict": verdict(+1, pt, lo, hi, n, need), "note": note})

    add("H-V0", "vs-hand gap side: ML residual > 0 (units ML/RL in table)", full["resid_pts"], full["resid_lo"],
        full["resid_hi"], full["resid_p"], full["n"], full["n_80pct_power"],
        f"ML {full['ml_units']:+.1f}u ROI {full['ml_roi_pct']:+.1f}% [{100*full['ml_units_lo']/full['n']:+.1f},{100*full['ml_units_hi']/full['n']:+.1f}]; RL {full['rl_units']:+.1f}u")

    def subgroup(hid: str, text: str, mask: pd.Series, note: str = "") -> None:
        a, b = fr.loc[mask, "resid"].to_numpy() * 100, fr.loc[~mask, "resid"].to_numpy() * 100
        pt, lo, hi, p = boot_diff(a, b, rng)
        add(hid, text, pt, lo, hi, p, int(mask.sum()), power_n(pt, float(np.std(a))) if len(a) else math.nan,
            f"in-group resid {a.mean():+.1f} (n={len(a)}) vs out {b.mean():+.1f} (n={len(b)}); {note}")

    subgroup("H-V1", "residual positive only when side's split PA (60d) >= 800", fr["side_pa60"].ge(800))
    subgroup("H-V2", "profit only from dogs (side price > 0)", fr["side_price"].gt(0))
    rl_ok = fr["rl_resid"].notna()
    diff = (fr.loc[rl_ok, "rl_resid"] - fr.loc[rl_ok, "resid"]).to_numpy() * 100
    pt, lo, hi, p = boot_ci(diff, "mean", rng)
    add("H-V3", "RL residual > ML residual (paired)", pt, lo, hi, p, int(rl_ok.sum()), power_n(pt, float(np.std(diff))))
    subgroup("H-V4", "value only when vs-hand rank disagrees with Overall rank by >= 5", fr["side_rank_gap"].ge(5))
    subgroup("H-V5", "effect stronger vs LHP starters", fr["side_opp_hand"].eq("L"))
    subgroup("H-V6", "driven by top-quartile |vsL - vsR| platoon spread", fr["side_platoon"].ge(q75),
             f"q75={q75:.0f} pts")
    h = pd.DataFrame(hyp)
    h["holm_p"] = holm(h["raw_p"].tolist())
    return pd.DataFrame(tables), h


# --- price quality -----------------------------------------------------------------------


def price_quality(df: pd.DataFrame, closing: Path | None) -> dict | None:
    if closing is None or not closing.exists():
        return None
    cl = pd.read_csv(closing)
    need = {"home", "away", "date", "p_home_close"}
    if not need.issubset(cl.columns):
        return None
    cl["date"] = pd.to_datetime(cl["date"])
    m = df.merge(cl, on=["home", "away", "date"], how="inner")
    if m.empty:
        return None
    d = (m["p_home"] - m["p_home_close"]).abs()
    return {"n": len(m), "mean_abs_diff_pts": 100 * d.mean(), "median_abs_diff_pts": 100 * d.median()}


# --- report ------------------------------------------------------------------------------


def _fmt(x: float, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def _p(x: float) -> str:
    return "—" if math.isnan(x) else ("<.001" if x < 0.001 else f"{x:.3f}")


def _metric_md(t: pd.DataFrame, stratum: str, group: str) -> str:
    sub = t[t["stratum"].eq(stratum) & t["group"].eq(group) & t["scoring"].eq("pts")]
    lines = ["| metric | n | side win% | implied% | resid (pts) [95% CI] | p | ML u | RL cover% | RL u | r(win) | r(RD) | r(R 1-5) | r(R 6+) | logit β [CI] | n@80% |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sub.itertuples():
        if r.n < 20:
            lines.append(f"| {r.metric} | {r.n} | — | | | | | | | | | | | | |")
            continue
        lines.append(
            f"| {r.metric} | {r.n} | {r.side_win_pct:.1f} | {r.implied_pct:.1f} | {_fmt(r.resid_pts)} [{_fmt(r.resid_lo)}, {_fmt(r.resid_hi)}] | {_p(r.resid_p)} "
            f"| {_fmt(r.ml_units)} | {r.rl_cover_pct:.1f} | {_fmt(r.rl_units)} | {_fmt(r.r_win, 3)} | {_fmt(r.r_run_diff, 3)} | {_fmt(r.r_runs_1_5, 3)} | {_fmt(r.r_runs_6plus, 3)} "
            f"| {_fmt(r.logit_beta, 3)} [{_fmt(r.logit_lo, 3)}, {_fmt(r.logit_hi, 3)}] | {r.n_80pct_power:,.0f} |")
    return "\n".join(lines)


def write_report(out_dir: Path, df: pd.DataFrame, tables: pd.DataFrame, hyp: pd.DataFrame, hyp_late: pd.DataFrame,
                 roll: pd.DataFrame, bat_tables: pd.DataFrame, bat_hyp: pd.DataFrame, pq: dict | None,
                 meta: dict) -> Path:
    L: list[str] = []
    L.append("# Worksheet SP/BP metric scores vs pre-game prices: 2024-2026 out-of-sample test\n")
    L.append(meta.get("preamble", ""))
    L.append("\n## Sample\n")
    L.append("| stratum | priced games | with SP scores both sides | with BP scores | median h to first pitch |\n|---|---|---|---|---|")
    for st, sub in df.groupby("stratum"):
        L.append(f"| {st} | {len(sub)} | {int((sub['home_sp_total'].notna() & sub['away_sp_total'].notna()).sum())} "
                 f"| {int(sub['home_bp_total'].notna().sum())} | {sub['hours_to_pitch'].median():.2f} |")
    L.append(f"| **pooled** | {len(df)} | {int((df['home_sp_total'].notna() & df['away_sp_total'].notna()).sum())} "
             f"| {int(df['home_bp_total'].notna().sum())} | {df['hours_to_pitch'].median():.2f} |")
    L.append("\nPrices are **pre-game** snapshots (median hours to first pitch above), not closing. "
             "Book: first available of pinnacle/fanduel/draftkings/betmgm (column `book`); implied = no-vig on that book's two sides.\n")

    L.append("\n## 2. Pre-registered 2026 leaners, tested on 2024 + 2025 + 2026-early only\n")
    L.append("| id | hypothesis | n | point (pts) [95% CI] | raw p | Holm p | ML u | RL u | n@80% | verdict |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in hyp.itertuples():
        L.append(f"| {r.id} | {r.hypothesis} | {r.n} | {_fmt(r.point)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_p(r.raw_p)} | {_p(r.holm_p) if not math.isnan(r.holm_p) else '—'} "
                 f"| {_fmt(r.ml_units)} | {_fmt(r.rl_units)} | {r.n_80pct_power:,.0f} | **{r.verdict}** |" if not math.isnan(r.n_80pct_power) else
                 f"| {r.id} | {r.hypothesis} | {r.n} | {_fmt(r.point)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_p(r.raw_p)} | — | {_fmt(r.ml_units)} | {_fmt(r.rl_units)} | — | **{r.verdict}** |")
    L.append("\nH8a/H8b are one Holm slot (the worse p of the pair). H6/S4 use the SP total; H7 uses a proxy F5 line "
             "(4.5/5/5.5 by full-game total) because F5 prices were not fetched -- it tests the *subset* Under rate "
             "against the all-games Under rate at the same proxy lines.\n")
    if not hyp_late.empty:
        L.append("\n### Same hypotheses on 2026-late (2026-07-19..) with Odds API pre-game prices -- the original study's window, for reference only\n")
        L.append("| id | n | point (pts) [95% CI] | raw p | ML u | RL u | sign |\n|---|---|---|---|---|---|---|")
        for r in hyp_late.itertuples():
            L.append(f"| {r.id} | {r.n} | {_fmt(r.point)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_p(r.raw_p)} | {_fmt(r.ml_units)} | {_fmt(r.rl_units)} | "
                     f"{'same' if (not math.isnan(r.point)) and (r.point > 0) == (r.expected_sign > 0) else 'opposite'} |")

    for st in ["pooled", *sorted(df["stratum"].unique())]:
        L.append(f"\n## 1. Per-metric tables -- {st} (compressed +2/+1/-2 scoring, home-minus-away gap)\n")
        L.append("### Starters\n")
        L.append(_metric_md(tables, st, "SP"))
        L.append("\n### Bullpens\n")
        L.append(_metric_md(tables, st, "BP"))
    L.append("\nDecile and z-score alternates for every metric are in `oos_metric_tables.csv` (`scoring` = dec / z).\n")

    L.append("\n## 3. Whiff vs Contact family split (pooled)\n")
    fam = tables[tables["stratum"].eq("pooled") & tables["metric"].str.startswith("FAMILY")]
    L.append("| group | family | n | resid (pts) [CI] | ML u | RL cover% | RL u | RL resid |\n|---|---|---|---|---|---|---|---|")
    for r in fam.itertuples():
        L.append(f"| {r.group} | {r.metric.replace('FAMILY ', '')} | {r.n} | {_fmt(r.resid_pts)} [{_fmt(r.resid_lo)}, {_fmt(r.resid_hi)}] | {_fmt(r.ml_units)} | {r.rl_cover_pct:.1f} | {_fmt(r.rl_units)} | {_fmt(r.rl_resid_pts)} |")

    L.append("\n## 4. Market-controlled logit (home win ~ gap + market logit), pooled\n")
    L.append("Coefficient on the gap (per point of home-minus-away score) with 95% CI; the `logit β` column of the per-metric tables above. Decile/z variants in the CSV.\n")
    pooled = tables[tables["stratum"].eq("pooled") & tables["scoring"].eq("pts") & ~tables["metric"].str.startswith("FAMILY")]
    L.append("| group | metric | n | β | 95% CI | p |\n|---|---|---|---|---|---|")
    for r in pooled.itertuples():
        if r.n >= 20:
            L.append(f"| {r.group} | {r.metric} | {r.n} | {_fmt(r.logit_beta, 3)} | [{_fmt(r.logit_lo, 3)}, {_fmt(r.logit_hi, 3)}] | {_p(r.logit_p)} |")

    L.append("\n## 5. Stability: rolling 400-game ML residual (step 100), top leaners\n")
    for name, sub in roll.groupby("leaner", sort=False):
        flips = int((np.sign(sub["resid_pts"]).diff().abs() > 0).sum())
        L.append(f"\n**{name}** -- {len(sub)} windows, {int((sub['resid_pts'] > 0).sum())} positive, {flips} sign changes\n")
        L.append("| window | n | resid (pts) | se |\n|---|---|---|---|")
        for r in sub.itertuples():
            L.append(f"| {r.window_start} .. {r.window_end} | {r.n} | {_fmt(r.resid_pts)} | {r.se:.1f} |")

    L.append("\n## 6. Price quality: pre-game snapshot vs closing\n")
    if pq:
        L.append(f"n={pq['n']} overlapping games; mean |Δ implied| = {pq['mean_abs_diff_pts']:.2f} pts, median {pq['median_abs_diff_pts']:.2f} pts.")
    else:
        L.append(meta.get("pq_note", "Not testable: no closing-price file available."))

    L.append("\n## 7. Power\n")
    L.append("`n@80%` in every table is the number of side-bets needed to detect the observed pooled residual at its point estimate with 80% power (two-sided α=.05), using the observed SD of (win − implied).\n")

    if not bat_hyp.empty:
        L.append("\n## Add-on: batting 'bat vs hand' gap (H-V0..H-V6), test set 2024 + 2025 + 2026-early\n")
        L.append("| id | hypothesis | n | point (pts) [CI] | raw p | Holm p | n@80% | verdict | note |\n|---|---|---|---|---|---|---|---|---|")
        for r in bat_hyp.itertuples():
            L.append(f"| {r.id} | {r.hypothesis} | {r.n} | {_fmt(r.point)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_p(r.raw_p)} | {_p(r.holm_p)} | {r.n_80pct_power:,.0f} | **{r.verdict}** | {r.note} |"
                     if not math.isnan(r.n_80pct_power) else
                     f"| {r.id} | {r.hypothesis} | {r.n} | {_fmt(r.point)} [{_fmt(r.lo)}, {_fmt(r.hi)}] | {_p(r.raw_p)} | {_p(r.holm_p)} | — | **{r.verdict}** | {r.note} |")
        L.append("\n### vs-hand gap by stratum, price bucket and half-season\n")
        L.append("| slice | scoring | n | side win% | implied% | resid [CI] | ML u | ROI% | RL cover% | RL u | RL resid |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for r in bat_tables.itertuples():
            if r.n < 20:
                L.append(f"| {r.stratum} | {r.scoring} | {r.n} | — | | | | | | | |")
                continue
            L.append(f"| {r.stratum} | {r.scoring} | {r.n} | {r.side_win_pct:.1f} | {r.implied_pct:.1f} | {_fmt(r.resid_pts)} [{_fmt(r.resid_lo)}, {_fmt(r.resid_hi)}] "
                     f"| {_fmt(r.ml_units)} | {_fmt(r.ml_roi_pct)} | {r.rl_cover_pct:.1f} | {_fmt(r.rl_units)} | {_fmt(r.rl_resid_pts)} |")

    L.append("\n## Not tested / caveats\n")
    L.append(meta.get("caveats", ""))
    L.append("\n## Artifacts\n")
    for p in sorted(out_dir.glob("*.csv")):
        L.append(f"- `{p}`")
    path = out_dir / "oos_worksheet_report.md"
    path.write_text("\n".join(L))
    return path


def run(games_paths: list[Path], prices_path: Path, out_dir: Path, closing: Path | None = None,
        meta: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    games = load_games(games_paths)
    px = load_prices(prices_path)
    df = merge_prices(games, px)
    df.to_csv(out_dir / "oos_priced_games.csv", index=False)
    test = df[~df["stratum"].eq("2026-late")]
    late = df[df["stratum"].eq("2026-late")]
    tables = metric_tables(df, rng)
    tables.to_csv(out_dir / "oos_metric_tables.csv", index=False)
    hyp = hypotheses(test, rng)
    hyp.to_csv(out_dir / "oos_hypotheses.csv", index=False)
    hyp_late = hypotheses(late, rng) if len(late) >= 100 else pd.DataFrame()
    if not hyp_late.empty:
        hyp_late.to_csv(out_dir / "oos_hypotheses_2026_late.csv", index=False)
    roll = rolling(df)
    roll.to_csv(out_dir / "oos_rolling_400.csv", index=False)
    bat_tables, bat_hyp = batting(test, rng)
    if not bat_hyp.empty:
        bat_tables.to_csv(out_dir / "oos_batting_tables.csv", index=False)
        bat_hyp.to_csv(out_dir / "oos_batting_hypotheses.csv", index=False)
    pq = price_quality(df, closing)
    return write_report(out_dir, df, tables, hyp, hyp_late, roll, bat_tables, bat_hyp, pq, meta or {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=Path, nargs="+", required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / ".mlb_engine" / "audit" / "oos")
    ap.add_argument("--closing", type=Path, default=None,
                    help="CSV with home, away, date, p_home_close for the price-quality check")
    ap.add_argument("--meta", type=Path, default=None,
                    help="JSON with optional 'preamble', 'pq_note', 'caveats' Markdown blocks for the report")
    args = ap.parse_args()
    meta = json.loads(args.meta.read_text()) if args.meta else None
    path = run(args.games, args.prices, args.out_dir, args.closing, meta)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
