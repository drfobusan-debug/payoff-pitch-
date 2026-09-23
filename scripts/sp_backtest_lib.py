"""Pure pandas/numpy half of the SP/BP scoring backtest.

Nothing in here talks to the network, so every routine can be exercised on a
synthetic frame (see ``tests/test_sp_scoring_backtest.py``).  The driver in
``scripts/sp_scoring_backtest.py`` fetches the inputs and calls into this module.

Conventions used throughout:

* A metric *gap* is away minus home after each metric has been signed so that a
  larger number is better for the pitcher (``lower_is_better`` metrics are
  negated).  ``gap > 0`` favours the away side.
* A *residual* is ``won - no_vig_prob`` for the side the signal favours; the
  market's own price is the yardstick, never the raw win rate.
* *Units* are the profit of one unit staked on the favoured side at the recorded
  American price (``+150`` returns 1.5, ``-150`` returns 0.667, a push 0).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from mlb_engine.output.daily_worksheet import BOT, BP_COLS, K_TEAMS, MID, SP_COLS, TOP

SP_LABELS = [c[0] for c in SP_COLS]
SP_LOWER = {c[0]: c[3] for c in SP_COLS}
BP_LABELS = [c[0] for c in BP_COLS]
BP_LOWER = {c[0]: c[4] for c in BP_COLS}
CONTACT = ["xERA", "xFIP", "SIERA", "FB%", "Barrel%", "HardHit%"]
WHIFF = ["K%", "K-BB%", "CSW%", "Stuff+", "O-Swing%"]
BP_KEY = "bp"
SCHEMES = ("cmp", "dec", "z")
Z_CLIP = 3.0


# --- scoring ---------------------------------------------------------------------------


def _signed(values: pd.Series, lower_is_better: bool) -> pd.Series:
    return -values if lower_is_better else values


def compressed_points(values: pd.Series, lower_is_better: bool, k: int) -> pd.Series:
    """``daily_worksheet.score`` for one metric: best k +2, worst k -2, rest +1, NaN worst."""
    s = _signed(values.astype(float), lower_is_better)
    order = s.fillna(-np.inf).rank(method="first", ascending=False)  # 1 = best
    n = len(s)
    pts = pd.Series(MID, index=s.index, dtype=int)
    pts[order <= k] = TOP
    pts[order > n - k] = BOT
    return pts


def decile_points(values: pd.Series, lower_is_better: bool) -> pd.Series:
    """0..9 by percentile rank of the signed metric (9 = best decile); NaN scores 0."""
    s = _signed(values.astype(float), lower_is_better)
    pct = s.rank(method="average", pct=True)
    pts = np.floor(pct * 10).clip(0, 9)
    return pts.fillna(0).astype(int)


def z_points(values: pd.Series, lower_is_better: bool) -> pd.Series:
    """Standard score of the signed metric, clipped to +-3; NaN scores the minimum."""
    s = _signed(values.astype(float), lower_is_better)
    sd = s.std(ddof=0)
    z = (s - s.mean()) / sd if sd and not math.isnan(sd) else pd.Series(0.0, index=s.index)
    z = z.clip(-Z_CLIP, Z_CLIP)
    return z.fillna(z.min() if z.notna().any() else 0.0)


def _score(df: pd.DataFrame, labels: list[str], lower: dict[str, bool], k: int) -> pd.DataFrame:
    out = df.copy()
    for label in labels:
        v = out[label] if label in out else pd.Series(np.nan, index=out.index)
        out[f"cmp_{label}"] = compressed_points(v, lower[label], k)
        out[f"dec_{label}"] = decile_points(v, lower[label])
        out[f"z_{label}"] = z_points(v, lower[label])
    for scheme in SCHEMES:
        out[f"{scheme}_total"] = out[[f"{scheme}_{m}" for m in labels]].sum(axis=1)
    contact = [m for m in CONTACT if m in labels]
    whiff = [m for m in WHIFF if m in labels]
    out["cmp_contact"] = out[[f"cmp_{m}" for m in contact]].sum(axis=1)
    out["cmp_whiff"] = out[[f"cmp_{m}" for m in whiff]].sum(axis=1)
    out["dec_contact"] = out[[f"dec_{m}" for m in contact]].sum(axis=1)
    out["dec_whiff"] = out[[f"dec_{m}" for m in whiff]].sum(axis=1)
    return out


def score_starters(day: pd.DataFrame) -> pd.DataFrame:
    """Score one date's qualifying starters; k is ~10% of the table, floor 3."""
    k = max(3, round(len(day) * 0.10))
    out = _score(day, SP_LABELS, SP_LOWER, k)
    out["k"] = k
    out["tercile"] = pd.qcut(out["cmp_total"].rank(method="first"), 3, labels=[0, 1, 2]).astype(int) \
        if len(out) >= 3 else 1
    out["contact_tercile"] = (
        pd.qcut(out["cmp_contact"].rank(method="first"), 3, labels=[0, 1, 2]).astype(int)
        if len(out) >= 3 else 1
    )
    return out


def score_bullpens(day: pd.DataFrame) -> pd.DataFrame:
    return _score(day, BP_LABELS, BP_LOWER, K_TEAMS)


# --- prices ----------------------------------------------------------------------------


def american_to_prob(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def profit(american: float) -> float:
    """Profit on a 1-unit winning stake."""
    return american / 100 if american > 0 else 100 / -american


_LINE_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*$")


def _parse_day(entries: list[dict]) -> dict[str, dict]:
    """matchup -> {ml, rl, total, f5_ml, f5_total} from one closing/board file."""
    out: dict[str, dict] = {}
    for e in entries:
        m = e.get("market")
        if m not in ("game_ml", "game_rl", "game_total", "f5_ml", "f5_total"):
            continue
        g = out.setdefault(e["matchup"], {"ml": {}, "rl": {}, "total": {}, "f5_ml": {}, "f5_total": {}})
        sel, am, nv = str(e["selection"]), float(e["american"]), e.get("no_vig_prob")
        nv = float(nv) if nv is not None else american_to_prob(am)
        team = sel.split()[0]
        if m == "game_ml":
            g["ml"][team] = (am, nv)
        elif m == "f5_ml":
            g["f5_ml"][team] = (am, nv)
        elif m == "game_rl":
            mt = _LINE_RE.search(sel)
            if mt:
                g["rl"][(team, float(mt.group(1)))] = (am, nv)
        else:
            mt = _LINE_RE.search(sel)
            if not mt:
                continue
            line = float(mt.group(1))
            ou = "over" if "over" in sel.lower() else "under"
            g[m if m == "f5_total" else "total"].setdefault(line, {})[ou] = (am, nv)
    return out


def _main_total(tbl: dict[float, dict]) -> tuple[float, float, float, float, float] | None:
    """Line whose over/under are closest to a pick; returns (line, over_am, under_am, p_over, p_under)."""
    best = None
    for line, sides in tbl.items():
        if "over" not in sides or "under" not in sides:
            continue
        dist = abs(sides["over"][1] - 0.5)
        if best is None or dist < best[0]:
            best = (dist, line, sides["over"][0], sides["under"][0], sides["over"][1], sides["under"][1])
    return None if best is None else best[1:]


ESPN_ABBR = {"ARI": "AZ", "CHW": "CWS"}


def _devig(a: float, b: float) -> tuple[float, float]:
    pa, pb = american_to_prob(a), american_to_prob(b)
    return pa / (pa + pb), pb / (pa + pb)


def _am(node: dict | None) -> float | None:
    """American price out of an ESPN odds leaf ({'american': '-175', ...})."""
    if not node:
        return None
    raw = str(node.get("american", "")).replace("EVEN", "+100").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def espn_entries(away: str, home: str, item: dict) -> list[dict]:
    """Flatten one ESPN competition-odds item into engine-state closing-file entries.

    Uses the provider's ``close`` snapshot (moneyline, run line, total); games
    without a closing snapshot are skipped rather than priced off the opener.
    Returns ``game_ml`` / ``game_rl`` / ``game_total`` rows with no-vig probabilities.
    """
    matchup = f"{away} @ {home}"
    rows: list[dict] = []
    ao, ho = item.get("awayTeamOdds") or {}, item.get("homeTeamOdds") or {}
    ac, hc = ao.get("close") or {}, ho.get("close") or {}
    ml_a, ml_h = _am(ac.get("moneyLine")), _am(hc.get("moneyLine"))
    if ml_a is not None and ml_h is not None:
        pa, ph = _devig(ml_a, ml_h)
        rows += [{"matchup": matchup, "market": "game_ml", "selection": f"{away} ML", "american": ml_a, "no_vig_prob": pa},
                 {"matchup": matchup, "market": "game_ml", "selection": f"{home} ML", "american": ml_h, "no_vig_prob": ph}]
    rl_a, rl_h = _am(ac.get("spread")), _am(hc.get("spread"))
    ln_a, ln_h = _am(ac.get("pointSpread")), _am(hc.get("pointSpread"))
    if rl_a is not None and rl_h is not None and ln_a is not None and ln_h is not None and ln_a == -ln_h:
        pa, ph = _devig(rl_a, rl_h)
        rows += [{"matchup": matchup, "market": "game_rl", "selection": f"{away} {ln_a:+.1f}", "american": rl_a, "no_vig_prob": pa},
                 {"matchup": matchup, "market": "game_rl", "selection": f"{home} {ln_h:+.1f}", "american": rl_h, "no_vig_prob": ph}]
    close = item.get("close") or {}
    ov, un = _am(close.get("over")), _am(close.get("under"))
    tot = (close.get("total") or {}).get("alternateDisplayValue")
    if ov is not None and un is not None and tot not in (None, "", "0"):
        line = float(tot)
        po, pu = _devig(ov, un)
        rows += [{"matchup": matchup, "market": "game_total", "selection": f"Over {line}", "american": ov, "no_vig_prob": po},
                 {"matchup": matchup, "market": "game_total", "selection": f"Under {line}", "american": un, "no_vig_prob": pu}]
    return rows


def load_prices(closing_dir: Path, board_dir: Path | None = None,
                espn_dir: Path | None = None) -> dict[tuple[str, str], dict]:
    """(date, matchup) -> parsed prices; engine-state closing > opening board > ESPN close."""
    out: dict[tuple[str, str], dict] = {}
    sources = [("espn", espn_dir), ("board", board_dir), ("closing", closing_dir)]
    for src, d in sources:
        if d is None or not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            day = p.stem.split("_")[-1]
            entries = json.loads(p.read_text())
            for matchup, g in _parse_day(entries).items():
                g["src"] = src
                out[(day, matchup)] = g  # closing overwrites board
    return out


def price_columns(row: pd.Series, g: dict) -> dict:
    away, home = row["away"], row["home"]
    cols: dict = {"price_src": g.get("src", "")}
    ml = g.get("ml", {})
    if away in ml and home in ml:
        cols["ml_away"], cols["ml_prob_away"] = ml[away]
        cols["ml_home"], cols["ml_prob_home"] = ml[home]
    f5 = g.get("f5_ml", {})
    if away in f5 and home in f5:
        cols["f5ml_away"], cols["f5ml_prob_away"] = f5[away]
        cols["f5ml_home"], cols["f5ml_prob_home"] = f5[home]
    rl = g.get("rl", {})
    for team, tag in ((away, "away"), (home, "home")):
        for line, lbl in ((-1.5, "m15"), (1.5, "p15")):
            if (team, line) in rl:
                cols[f"rl_{tag}_{lbl}"], cols[f"rl_prob_{tag}_{lbl}"] = rl[(team, line)]
    tot = _main_total(g.get("total", {}))
    if tot:
        cols["total_line"], cols["over_am"], cols["under_am"], cols["over_prob"], cols["under_prob"] = tot
    f5t = _main_total(g.get("f5_total", {}))
    if f5t:
        (cols["f5_line"], cols["f5_over_am"], cols["f5_under_am"],
         cols["f5_over_prob"], cols["f5_under_prob"]) = f5t
    return cols


# --- game frame ------------------------------------------------------------------------

PRICE_COLS = [
    "price_src", "ml_away", "ml_prob_away", "ml_home", "ml_prob_home",
    "f5ml_away", "f5ml_prob_away", "f5ml_home", "f5ml_prob_home",
    "rl_away_m15", "rl_prob_away_m15", "rl_away_p15", "rl_prob_away_p15",
    "rl_home_m15", "rl_prob_home_m15", "rl_home_p15", "rl_prob_home_p15",
    "total_line", "over_am", "under_am", "over_prob", "under_prob",
    "f5_line", "f5_over_am", "f5_under_am", "f5_over_prob", "f5_under_prob",
]


def _side_cols(sp: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [c for c in sp.columns if c not in ("date", "pitcher", "name", "team", "IP")]
    ren = {c: f"{prefix}_{c}" for c in keep}
    ren["pitcher"] = f"{prefix}_id"
    return sp[["date", "pitcher", *keep]].rename(columns=ren)


def build_frame(
    games: pd.DataFrame, sp_all: pd.DataFrame, bp_all: pd.DataFrame,
    prices: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    """One row per game: results, both starters' and pens' scores, and the prices."""
    df = games.copy()
    df["date"] = df["date"].astype(str)
    sp_all = sp_all.copy()
    sp_all["date"] = sp_all["date"].astype(str)
    bp_all = bp_all.copy()
    bp_all["date"] = bp_all["date"].astype(str)
    for side in ("away", "home"):
        df = df.merge(_side_cols(sp_all, f"{side}_sp"), on=["date", f"{side}_sp_id"], how="left")
        bkeep = [c for c in bp_all.columns if c not in ("date", "team")]
        b = bp_all[["date", "team", *bkeep]].rename(
            columns={**{c: f"{side}_bp_{c}" for c in bkeep}, "team": side})
        df = df.merge(b, on=["date", side], how="left")
    df["matchup"] = df["away"] + " @ " + df["home"]
    dh_count = df.groupby(["date", "matchup"])["game_pk"].transform("count")
    df["dh_collision"] = dh_count > 1
    price_rows = []
    for _, r in df.iterrows():
        g = prices.get((r["date"], r["matchup"]))
        price_rows.append(price_columns(r, g) if g and not r["dh_collision"] else {})
    pr = pd.DataFrame(price_rows, index=df.index)
    for c in PRICE_COLS:
        if c not in pr:
            pr[c] = np.nan
    df = pd.concat([df, pr], axis=1)
    df["run_diff"] = df["away_runs"] - df["home_runs"]
    df = df[df["run_diff"] != 0].copy()  # no ties in MLB: 0-0 rows are postponed/suspended games
    df["away_win"] = (df["run_diff"] > 0).astype(int)
    df["total_runs"] = df["away_runs"] + df["home_runs"]
    df["f5_runs"] = df["away_f5"] + df["home_f5"]
    df["both_scored"] = df["away_sp_cmp_total"].notna() & df["home_sp_cmp_total"].notna()
    for scheme in SCHEMES:
        df[f"sp_gap_{scheme}"] = df[f"away_sp_{scheme}_total"] - df[f"home_sp_{scheme}_total"]
        df[f"bp_gap_{scheme}"] = df[f"away_bp_{scheme}_total"] - df[f"home_bp_{scheme}_total"]
    df["sp_gap_contact"] = df["away_sp_cmp_contact"] - df["home_sp_cmp_contact"]
    df["sp_gap_whiff"] = df["away_sp_cmp_whiff"] - df["home_sp_cmp_whiff"]
    df["sp_gap_contact_dec"] = df["away_sp_dec_contact"] - df["home_sp_dec_contact"]
    df["sp_gap_whiff_dec"] = df["away_sp_dec_whiff"] - df["home_sp_dec_whiff"]
    for label in SP_LABELS:
        sgn = -1.0 if SP_LOWER[label] else 1.0
        df[f"gap_sp_{label}"] = sgn * (df[f"away_sp_{label}"] - df[f"home_sp_{label}"])
    for label in BP_LABELS:
        sgn = -1.0 if BP_LOWER[label] else 1.0
        df[f"gap_bp_{label}"] = sgn * (df[f"away_bp_{label}"] - df[f"home_bp_{label}"])
    return df


# --- statistics --------------------------------------------------------------------------


def boot_mean_ci(x: np.ndarray, boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI (paired: rows resampled together)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean_test_p(x: np.ndarray) -> float:
    """One-sample t-test of mean != 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return math.nan
    return float(sps.ttest_1samp(x, 0.0).pvalue)


def corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 4 or x[m].std() == 0 or y[m].std() == 0:
        return math.nan, math.nan, n
    r, p = sps.pearsonr(x[m], y[m])
    return float(r), float(p), n


def n_for_mean(delta: float, sd: float = 0.5, power_z: float = 0.84) -> int:
    """Games needed to detect a mean residual ``delta`` at alpha .05, 80% power."""
    return int(math.ceil(((1.96 + power_z) * sd / delta) ** 2)) if delta else 0


def n_for_corr(r: float) -> int:
    return int(math.ceil(((1.96 + 0.84) / math.atanh(abs(r))) ** 2 + 3)) if r else 0


def side_outcomes(df: pd.DataFrame, gap: pd.Series) -> pd.DataFrame:
    """For the side favoured by ``gap`` (>0 away, <0 home): win, market prob, ML/RL units, RL cover.

    The favoured side's run line is -1.5 when it is the moneyline favourite, else
    +1.5, at the price recorded for that line.  Rows with gap == 0 or missing
    prices are dropped from the price columns but kept for the win column.
    """
    away = gap > 0
    home = gap < 0
    out = pd.DataFrame(index=df.index)
    out["decided"] = away | home
    out["won"] = np.where(away, df["away_win"], 1 - df["away_win"]).astype(float)
    out["margin"] = np.where(away, df["run_diff"], -df["run_diff"]).astype(float)
    out["mkt"] = np.where(away, df.get("ml_prob_away", np.nan), df.get("ml_prob_home", np.nan))
    ml_price = np.where(away, df.get("ml_away", np.nan), df.get("ml_home", np.nan))
    out["ml_units"] = [
        (profit(a) if w == 1 else -1.0) if not (isinstance(a, float) and math.isnan(a)) else math.nan
        for a, w in zip(ml_price, out["won"], strict=True)
    ]
    out["residual"] = out["won"] - out["mkt"]
    fav = out["mkt"] >= 0.5
    line = np.where(fav, -1.5, 1.5)
    tag = np.where(fav, "m15", "p15")
    rl_price, rl_prob = [], []
    for i, (is_away, t) in enumerate(zip(away, tag, strict=True)):
        side = "away" if is_away else "home"
        rl_price.append(df[f"rl_{side}_{t}"].iloc[i] if f"rl_{side}_{t}" in df else math.nan)
        rl_prob.append(df[f"rl_prob_{side}_{t}"].iloc[i] if f"rl_prob_{side}_{t}" in df else math.nan)
    out["rl_line"] = line
    out["rl_price"] = rl_price
    out["rl_mkt"] = rl_prob
    out["rl_cover"] = np.where(out["margin"] + out["rl_line"] > 0, 1.0,
                               np.where(out["margin"] + out["rl_line"] < 0, 0.0, np.nan))
    out["rl_units"] = [
        math.nan if (math.isnan(p) or math.isnan(c)) else (profit(p) if c == 1 else -1.0)
        for p, c in zip(out["rl_price"], out["rl_cover"], strict=True)
    ]
    out.loc[~out["decided"], ["won", "margin", "mkt", "ml_units", "residual", "rl_cover", "rl_units"]] = np.nan
    return out


def signal_summary(df: pd.DataFrame, gap: pd.Series, boot: int) -> dict:
    """The per-metric/per-signal line: r's, side record, market prob, residual, units."""
    o = side_outcomes(df, gap)
    r_win, p_win, n_win = corr(gap, df["away_win"])
    r_rd, _, _ = corr(gap, df["run_diff"])
    r_mkt, _, _ = corr(gap, df["ml_prob_away"]) if "ml_prob_away" in df else (math.nan, math.nan, 0)
    dec = o[o["decided"]]
    priced = dec[dec["mkt"].notna()]
    res_m, res_lo, res_hi = boot_mean_ci(priced["residual"].to_numpy(), boot)
    rl = dec[dec["rl_units"].notna()]
    rl_m, rl_lo, rl_hi = boot_mean_ci(rl["rl_units"].to_numpy(), boot)
    return {
        "n": n_win, "r_win": r_win, "p_r_win": p_win, "r_run_diff": r_rd, "r_market": r_mkt,
        "n_decided": len(dec), "side_win_pct": float(dec["won"].mean()) if len(dec) else math.nan,
        "n_priced": len(priced), "side_win_pct_priced": float(priced["won"].mean()) if len(priced) else math.nan,
        "market_prob": float(priced["mkt"].mean()) if len(priced) else math.nan,
        "residual": res_m, "residual_lo": res_lo, "residual_hi": res_hi,
        "p_residual": mean_test_p(priced["residual"].to_numpy()),
        "ml_units": float(priced["ml_units"].sum()) if len(priced) else math.nan,
        "ml_roi": float(priced["ml_units"].mean()) if len(priced) else math.nan,
        "n_rl": len(rl), "rl_cover_pct": float(rl["rl_cover"].mean()) if len(rl) else math.nan,
        "rl_units": float(rl["rl_units"].sum()) if len(rl) else math.nan,
        "rl_roi": rl_m, "rl_roi_lo": rl_lo, "rl_roi_hi": rl_hi,
    }


def metric_table(df: pd.DataFrame, prefix: str, labels: list[str], boot: int) -> pd.DataFrame:
    rows = []
    for label in labels:
        col = f"gap_{prefix}_{label}"
        if col not in df:
            continue
        s = signal_summary(df, df[col], boot)
        s["metric"] = label
        s["group"] = "contact" if label in CONTACT else "whiff" if label in WHIFF else "other"
        rows.append(s)
    return pd.DataFrame(rows).set_index("metric") if rows else pd.DataFrame()


def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Plain OLS with an intercept; returns coef, se, t, p per column."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * sps.t.sf(np.abs(t), dof)
    return pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p}, index=["const", *names])


def _standardize(X: pd.DataFrame) -> pd.DataFrame:
    return (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)


def market_regressions(df: pd.DataFrame, labels: list[str], prefix: str = "sp") -> dict[str, pd.DataFrame]:
    """H3: (a) market prob on standardized gaps, (b) win / run diff on gaps + market logit."""
    cols = [f"gap_{prefix}_{m}" for m in labels if f"gap_{prefix}_{m}" in df]
    need = [*cols, "ml_prob_away", "away_win", "run_diff"]
    d = df.dropna(subset=need)
    d = d[(d["ml_prob_away"] > 0) & (d["ml_prob_away"] < 1)]
    if len(d) < len(cols) + 5:
        return {}
    X = _standardize(d[cols]).fillna(0).to_numpy()
    names = [c.replace(f"gap_{prefix}_", "") for c in cols]
    logit = np.log(d["ml_prob_away"] / (1 - d["ml_prob_away"])).to_numpy()
    out = {
        "market_on_gaps": ols(d["ml_prob_away"].to_numpy(), X, names),
        "win_on_gaps_and_market": ols(
            d["away_win"].to_numpy(float), np.column_stack([X, logit]), [*names, "market_logit"]),
        "run_diff_on_gaps_and_market": ols(
            d["run_diff"].to_numpy(float), np.column_stack([X, logit]), [*names, "market_logit"]),
        "win_on_market_only": ols(d["away_win"].to_numpy(float), logit[:, None], ["market_logit"]),
    }
    for k in out:
        out[k].attrs["n"] = len(d)
    return out


def pairing_cells(df: pd.DataFrame, boot: int, min_n: int = 8) -> pd.DataFrame:
    """H4: bucket games by (elite starter pts, poor starter pts); residual of the elite side."""
    d = df[df["both_scored"] & df["ml_prob_away"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    a, h = d["away_sp_cmp_total"], d["home_sp_cmp_total"]
    d["elite_pts"] = np.maximum(a, h).astype(int)
    d["poor_pts"] = np.minimum(a, h).astype(int)
    gap = d["sp_gap_cmp"]
    o = side_outcomes(d, gap)
    d = pd.concat([d, o[["won", "mkt", "residual", "ml_units", "rl_cover", "rl_units"]]], axis=1)
    d = d[gap != 0]
    gap = gap[d.index]
    rows = []
    for (e, p), g in d.groupby(["elite_pts", "poor_pts"]):
        if len(g) < min_n:
            continue
        m, lo, hi = boot_mean_ci(g["residual"].to_numpy(), boot)
        rows.append({"elite_pts": e, "poor_pts": p, "gap": e - p, "n": len(g),
                     "elite_wins": int(g["won"].sum()), "win_pct": g["won"].mean(),
                     "market_prob": g["mkt"].mean(), "residual": m, "residual_lo": lo, "residual_hi": hi,
                     "p": mean_test_p(g["residual"].to_numpy()),
                     "ml_units": g["ml_units"].sum(), "rl_units": g["rl_units"].sum()})
    cells = pd.DataFrame(rows)
    bands = []
    for lbl, lo_g, hi_g in (("1-4", 1, 5), ("5-9", 5, 10), ("10-14", 10, 15), ("15+", 15, 99)):
        g = d[(gap.abs() >= lo_g) & (gap.abs() < hi_g)]
        if len(g) == 0:
            continue
        m, lo, hi = boot_mean_ci(g["residual"].to_numpy(), boot)
        bands.append({"gap_band": lbl, "n": len(g), "win_pct": g["won"].mean(), "market_prob": g["mkt"].mean(),
                      "residual": m, "residual_lo": lo, "residual_hi": hi,
                      "p": mean_test_p(g["residual"].to_numpy()),
                      "ml_units": g["ml_units"].sum(), "rl_units": g["rl_units"].sum()})
    cells.attrs["bands"] = pd.DataFrame(bands)
    return cells


def starter_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scored starter appearance: points under each scheme and what he allowed."""
    rows = []
    for side in ("away", "home"):
        cols = {f"{side}_sp_{s}_total": f"{s}_total" for s in SCHEMES}
        cols.update({f"{side}_sp_cmp_contact": "cmp_contact", f"{side}_sp_cmp_whiff": "cmp_whiff",
                     f"{side}_sp_dec_contact": "dec_contact", f"{side}_sp_dec_whiff": "dec_whiff",
                     f"{side}_ra_f5": "ra_f5", f"{side}_sp_runs": "sp_runs", f"{side}_sp_outs": "sp_outs",
                     f"{side}_sp_id": "pitcher"})
        for label in SP_LABELS:
            cols[f"{side}_sp_{label}"] = label
            cols[f"{side}_sp_cmp_{label}"] = f"cmp_{label}"
            cols[f"{side}_sp_dec_{label}"] = f"dec_{label}"
        have = [c for c in cols if c in df]
        part = df[["date", "game_pk", *have]].rename(columns=cols)
        part["side"] = side
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    return out[out["cmp_total"].notna()]


def bullpen_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("away", "home"):
        cols = {f"{side}_bp_{s}_total": f"{s}_total" for s in SCHEMES}
        cols.update({f"{side}_bp_runs": "bp_runs", f"{side}_bp_outs": "bp_outs", f"{side}": "team"})
        for label in BP_LABELS:
            cols[f"{side}_bp_{label}"] = label
            cols[f"{side}_bp_cmp_{label}"] = f"cmp_{label}"
        have = [c for c in cols if c in df]
        part = df[["date", "game_pk", *have]].rename(columns=cols)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out[out["cmp_total"].notna() & (out["bp_outs"] > 0)]
    out["bp_ra9"] = out["bp_runs"] / out["bp_outs"] * 27
    return out


# --- candidate strategies ------------------------------------------------------------------

LOPSIDED_GAP = 10
MIN_UNDER = 0.55
MIN_RESIDUAL = 0.03


def strategy_verdict(effect: float, lo: float, hi: float, n: int, need: int) -> str:
    """supported: CI excludes 0 on the predicted side; rejected: powered null or CI on the wrong side."""
    if math.isnan(effect) or n == 0:
        return "not observed"
    if _ci_excludes_zero(lo, hi):
        return "supported" if effect > 0 else "rejected"
    return "rejected" if n >= need else "underpowered"


def _under_row(g: pd.DataFrame, boot: int) -> dict:
    """Blind F5 Under on ``g`` at the recorded line: hit rate, residual vs market, units."""
    under = (g["f5_runs"] < g["f5_line"]).astype(float)
    under[g["f5_runs"] == g["f5_line"]] = np.nan
    resid = under - g["f5_under_prob"]
    units = [
        math.nan if math.isnan(u) else (profit(a) if u == 1 else -1.0)
        for u, a in zip(under, g["f5_under_am"], strict=True)
    ]
    m, lo, hi = boot_mean_ci(resid.to_numpy(), boot)
    return {"n": int(under.notna().sum()), "hit": float(under.mean()) if under.notna().any() else math.nan,
            "market": float(g["f5_under_prob"].mean()) if len(g) else math.nan,
            "effect": m, "lo": lo, "hi": hi, "p": mean_test_p(resid.to_numpy()),
            "units": float(np.nansum(units)) if len(units) else math.nan}


def f5_contact_split(df: pd.DataFrame, boot: int) -> pd.DataFrame:
    """Raw F5 runs by (away, home) Contact-tercile pairing against the season F5 mean (no price)."""
    d = df[df["both_scored"]]
    season = float(d["f5_runs"].mean())
    rows = []
    cells = [("both top", (d["away_sp_contact_tercile"] == 2) & (d["home_sp_contact_tercile"] == 2)),
             ("one top", (d["away_sp_contact_tercile"] == 2) ^ (d["home_sp_contact_tercile"] == 2)),
             ("neither top", (d["away_sp_contact_tercile"] != 2) & (d["home_sp_contact_tercile"] != 2)),
             ("both bottom", (d["away_sp_contact_tercile"] == 0) & (d["home_sp_contact_tercile"] == 0))]
    for name, mask in cells:
        g = d[mask]
        m, lo, hi = boot_mean_ci((g["f5_runs"] - season).to_numpy(), boot)
        lined = g[g["f5_line"].notna()]
        rows.append({"cell": name, "n": len(g), "f5_mean": float(g["f5_runs"].mean()) if len(g) else math.nan,
                     "season_mean": season, "diff": m, "lo": lo, "hi": hi,
                     "p": mean_test_p((g["f5_runs"] - season).to_numpy()),
                     "n_lined": len(lined), "line_mean": float(lined["f5_line"].mean()) if len(lined) else math.nan,
                     "under_pct": float((lined["f5_runs"] < lined["f5_line"]).mean()) if len(lined) else math.nan})
    return pd.DataFrame(rows)


def strategy_table(priced: pd.DataFrame, boot: int) -> pd.DataFrame:
    """The five candidate strategies from the first study, graded against the recorded price.

    ``priced`` is every game with both starters scored and a closing/opening
    moneyline.  Effects are residuals (hit - market prob) so 0 is
    the price; ``need_n`` is the sample needed to detect the pre-registered
    effect at 80% power, ``need_n_obs`` the sample the observed effect would need.
    """
    rows: list[dict] = []
    need = n_for_mean(MIN_RESIDUAL)

    f5 = priced[priced["f5_line"].notna()]
    both = f5[(f5["away_sp_contact_tercile"] == 2) & (f5["home_sp_contact_tercile"] == 2)]
    u = _under_row(both, boot)
    need_u = n_for_mean(MIN_UNDER - 0.5)
    rows.append({"strategy": "S1 F5 Under, both starters top-tercile Contact", "market": "F5 total",
                 "n": u["n"], "hit": u["hit"], "market_prob": u["market"], "effect": u["effect"],
                 "lo": u["lo"], "hi": u["hi"], "p": u["p"], "units": u["units"], "need_n": need_u,
                 "threshold": f"Under >= {MIN_UNDER:.0%} and residual CI > 0",
                 "verdict": strategy_verdict(u["effect"], u["lo"], u["hi"], u["n"], need_u)})

    for name, col in (("S2 ML on the larger K-BB% gap (42d)", "gap_sp_K-BB%"),
                      ("S3 ML on the z-score SP total favourite", "sp_gap_z")):
        s = signal_summary(priced, priced[col], boot)
        rows.append({"strategy": name, "market": "ML", "n": s["n_priced"], "hit": s["side_win_pct_priced"],
                     "market_prob": s["market_prob"], "effect": s["residual"], "lo": s["residual_lo"],
                     "hi": s["residual_hi"], "p": s["p_residual"], "units": s["ml_units"], "need_n": need,
                     "threshold": f"residual >= +{MIN_RESIDUAL:.0%} with CI > 0",
                     "verdict": strategy_verdict(s["residual"], s["residual_lo"], s["residual_hi"], s["n_priced"], need)})

    lop = priced[priced["sp_gap_cmp"].abs() >= LOPSIDED_GAP]
    fade = side_outcomes(lop, -lop["sp_gap_cmp"])
    fade = fade[fade["decided"] & fade["mkt"].notna()]
    m, lo, hi = boot_mean_ci(fade["residual"].to_numpy(), boot)
    rows.append({"strategy": f"S4 fade the SP-favoured side when |compressed gap| >= {LOPSIDED_GAP}",
                 "market": "ML", "n": len(fade), "hit": float(fade["won"].mean()) if len(fade) else math.nan,
                 "market_prob": float(fade["mkt"].mean()) if len(fade) else math.nan, "effect": m, "lo": lo,
                 "hi": hi, "p": mean_test_p(fade["residual"].to_numpy()),
                 "units": float(fade["ml_units"].sum()) if len(fade) else math.nan, "need_n": need,
                 "threshold": f"fade residual >= +{MIN_RESIDUAL:.0%} with CI > 0",
                 "verdict": strategy_verdict(m, lo, hi, len(fade), need)})

    o = side_outcomes(priced, priced["sp_gap_contact"])
    favs = o[o["decided"] & (o["mkt"] >= 0.5) & o["rl_units"].notna()]
    be = favs["rl_price"].map(lambda a: 1 / (1 + profit(a)))
    short = be - favs["rl_cover"]  # > 0 when laying -1.5 loses to its own break-even
    m, lo, hi = boot_mean_ci(short.to_numpy(), boot)
    need_rl = n_for_mean(0.05)
    rows.append({"strategy": "S5 Contact-edge favourites laying -1.5 lose to break-even (RL < ML)",
                 "market": "RL -1.5", "n": len(favs), "hit": float(favs["rl_cover"].mean()) if len(favs) else math.nan,
                 "market_prob": float(be.mean()) if len(favs) else math.nan, "effect": m, "lo": lo, "hi": hi,
                 "p": mean_test_p(short.to_numpy()), "units": float(favs["rl_units"].sum()) if len(favs) else math.nan,
                 "need_n": need_rl, "threshold": "break-even - cover >= +5pt with CI > 0",
                 "verdict": strategy_verdict(m, lo, hi, len(favs), need_rl)})
    out = pd.DataFrame(rows)
    out["need_n_obs"] = [n_for_mean(abs(e)) if not math.isnan(e) and e else 0 for e in out["effect"]]
    return out


# --- report --------------------------------------------------------------------------------


def _f(x: float, nd: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _md_table(df: pd.DataFrame, cols: list[tuple[str, str, int | None]]) -> str:
    """cols: (column, header, decimals | None for pct | -1 for int)."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for idx, r in df.iterrows():
        cells = []
        for c, _, nd in cols:
            v = r[c] if c in r else idx
            if isinstance(v, str):
                cells.append(v)
            elif nd is None:
                cells.append(_pct(float(v)))
            elif nd == -1:
                cells.append("" if pd.isna(v) else str(int(v)))
            else:
                cells.append(_f(float(v), nd))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


SIGNAL_COLS: list[tuple[str, str, int | None]] = [
    ("metric", "signal", 0), ("n", "n", -1), ("r_win", "r win", 3), ("r_run_diff", "r run diff", 3),
    ("r_market", "r mkt", 3), ("side_win_pct_priced", "side acc", None), ("market_prob", "mkt prob", None),
    ("residual", "resid", 4), ("residual_lo", "lo", 4), ("residual_hi", "hi", 4), ("p_residual", "p", 3),
    ("ml_units", "ML u", 2), ("rl_cover_pct", "RL cover", None), ("rl_units", "RL u", 2),
]


def _verdict(ok: bool | None, powered: bool = True) -> str:
    if ok is None or not powered:
        return "underpowered"
    return "supported" if ok else "rejected"


def _ci_excludes_zero(lo: float, hi: float) -> bool:
    return not (math.isnan(lo) or math.isnan(hi)) and (lo > 0 or hi < 0)


def write_report(frame: pd.DataFrame, sp_all: pd.DataFrame, out: Path, boot: int = 2000,
                 bp_key: str = BP_KEY, report_name: str = "sp_scoring_backtest.md") -> str:
    out.mkdir(parents=True, exist_ok=True)
    df = frame[frame["both_scored"]].copy()
    priced = df[df["ml_prob_away"].notna()].copy()
    starters = starter_level(frame)
    pens = bullpen_level(frame)
    md: list[str] = ["# SP / bullpen scoring backtest (H1-H6)\n"]
    dates = sorted(frame["date"].unique())
    pdates = sorted(priced["date"].unique())
    md.append(f"Games with both starters scored: **{len(df)}** ({dates[0]}..{dates[-1]}); "
              f"of which priced (closing no-vig): **{len(priced)}** "
              f"({pdates[0] if pdates else '-'}..{pdates[-1] if pdates else '-'}). "
              f"Starter appearances scored: {len(starters)}; bullpen appearances: {len(pens)}. "
              f"Doubleheader matchups are unpriced (label collision). Bootstrap resamples: {boot}.\n")
    src = priced["price_src"].value_counts() if len(priced) else pd.Series(dtype=int)
    strat = strategy_table(priced, boot)
    strat.to_csv(out / "strategies.csv", index=False)
    split = f5_contact_split(frame, boot)
    split.to_csv(out / "f5_contact_split.csv", index=False)
    md.append("## Candidate strategies (priced window; effect = hit - market prob, 0 = the price)\n")
    md.append(_md_table(strat, [("strategy", "strategy", 0), ("market", "market", 0), ("n", "n", -1),
                                ("hit", "hit", None), ("market_prob", "mkt", None), ("effect", "effect", 4),
                                ("lo", "lo", 4), ("hi", "hi", 4), ("p", "p", 3), ("units", "units", 2),
                                ("need_n", "n needed", -1), ("need_n_obs", "n for obs.", -1),
                                ("verdict", "verdict", 0)]))
    md.append("\n`n needed` = priced games to detect the pre-registered effect at alpha .05 / 80% power "
              "(residual +3pt; F5 Under 55% vs 50%; RL 5pt short of break-even); `n for obs.` = games the "
              "observed effect would need. S5's `hit` is the -1.5 cover rate and `mkt` its break-even.\n")
    md.append("\nF5 runs by Contact-tercile pairing, **all scored games** (no price): diff = cell mean - season mean; "
              "`under` is the share below the recorded F5 line where one exists.\n")
    md.append(_md_table(split, [("cell", "cell", 0), ("n", "n", -1), ("f5_mean", "F5 runs", 3),
                                ("season_mean", "season", 3), ("diff", "diff", 3), ("lo", "lo", 3),
                                ("hi", "hi", 3), ("p", "p", 3), ("n_lined", "n lined", -1),
                                ("line_mean", "line", 2), ("under_pct", "under", None)]))
    md.append("")
    md.append(
        "**Data / method.** Starters identified from the Stats API boxscore (first pitcher listed); "
        "innings 1-5 runs from the linescore; bullpen runs/outs = team pitching line minus the starter. "
        "Postponed/suspended games (Stats API `codedGameState` not F/O, or a 0-0 score) are dropped. "
        "Metrics are FanGraphs leaderboards pulled as-of the day before each game (no look-ahead): "
        "starter skill metrics (K%, K-BB%, CSW%, O-Swing%, Stuff+) over the trailing 42 days with a "
        "150-pitch minimum, the rest (xERA, xFIP, SIERA, HardHit%, Barrel%, FB%) over the season with "
        "20 IP / 300 pitches, mirroring `daily_worksheet.starter_table`; bullpens use team relief lines with "
        "the per-column season/trailing windows of `BP_COLS`. Points: `compressed` = worksheet rule (+2 top-k, -2 bottom-k, +1 "
        "otherwise; k = 10% of qualifying starters, k = 3 of 30 pens), `decile` = 0-9 percentile rank, "
        "`z` = standardized value, all signed so higher is better. Gap = away - home. Price-source mix (games): "
        + ", ".join(f"{k}: {v}" for k, v in src.items()) +
        ". `closing` = `engine-state` no-vig closing lines, `board` = its opening board when no closing "
        "file exists for that date, `espn` = DraftKings closing ML / RL / full-game total from ESPN's "
        "public odds API (used only for dates the engine never priced; ESPN carries no F5 lines, so F5 "
        "market tests are confined to the engine-state window). "
        "Side accuracy / residual / units are for the side favoured by the gap (gap = 0 games skipped); "
        "RL = -1.5 for the favoured side when it is the ML favourite, +1.5 when it is the dog; units are "
        "one-unit stakes at the recorded American price.\n"
    )

    # H1 -------------------------------------------------------------------------------
    h1 = {}
    for name, gap in (("SP total (compressed)", "sp_gap_cmp"), ("Contact gap", "sp_gap_contact"),
                      ("Whiff gap", "sp_gap_whiff"), ("Contact gap (decile)", "sp_gap_contact_dec"),
                      ("Whiff gap (decile)", "sp_gap_whiff_dec"), ("SP total (decile)", "sp_gap_dec"),
                      ("SP total (z)", "sp_gap_z"), ("BP total (compressed)", f"{bp_key}_gap_cmp"),
                      ("BP total (decile)", f"{bp_key}_gap_dec")):
        s = signal_summary(priced, priced[gap], boot)
        s["metric"] = name
        h1[name] = s
    h1df = pd.DataFrame(h1.values()).set_index("metric")
    h1df.to_csv(out / "h1_signals.csv")
    c, w = h1["Contact gap"], h1["Whiff gap"]
    need_n = n_for_mean(0.03)
    c_ok = c["residual"] > 0 and _ci_excludes_zero(c["residual_lo"], c["residual_hi"])
    w_ok = w["residual"] <= 0
    rl_ok = c["rl_cover_pct"] >= 0.524 and w["rl_cover_pct"] <= 0.50
    powered = c["n_priced"] >= need_n
    md.append("## H1 - Contact-suppression edge beats the price; Whiff edge does not\n")
    md.append(_md_table(h1df.reset_index(), SIGNAL_COLS))
    md.append(f"\nPre-registered: Contact residual > 0 with CI excluding 0 **and** Whiff residual <= 0; "
              f"RL: Contact side >= 52.4%, Whiff side <= 50%.\n"
              f"- Contact ML residual {c['residual']:+.4f} [{c['residual_lo']:+.4f}, {c['residual_hi']:+.4f}], "
              f"p={_f(c['p_residual'])}, n={c['n_priced']}; Whiff {w['residual']:+.4f} "
              f"[{w['residual_lo']:+.4f}, {w['residual_hi']:+.4f}].\n"
              f"- RL cover: Contact {_pct(c['rl_cover_pct'])} (n={c['n_rl']}), Whiff {_pct(w['rl_cover_pct'])}.\n"
              f"- To detect a +3pt residual at 80% power you need ~{need_n} priced games; have {c['n_priced']}.\n"
              f"- **Verdict (ML): {_verdict(c_ok and w_ok, powered or c_ok)}**; "
              f"**Verdict (RL): {_verdict(rl_ok, powered or rl_ok)}**.\n")

    # H2 -------------------------------------------------------------------------------
    md.append("## H2 - SP quality predicts innings 1-5 runs and the F5 total more than the full-game total\n")
    st = starters
    r_ra, p_ra, n_ra = corr(st["cmp_contact"], st["ra_f5"])
    r_ra_t, p_ra_t, _ = corr(st["cmp_total"], st["ra_f5"])
    r_ra_w, p_ra_w, _ = corr(st["cmp_whiff"], st["ra_f5"])
    r_ra_d, p_ra_d, _ = corr(st["dec_total"], st["ra_f5"])
    r_sr, p_sr, _ = corr(st["cmp_contact"], st["sp_runs"])
    h2_rows = [
        {"test": "starter Contact pts vs runs allowed inn 1-5", "n": n_ra, "r": r_ra, "p": p_ra},
        {"test": "starter Whiff pts vs runs allowed inn 1-5", "n": n_ra, "r": r_ra_w, "p": p_ra_w},
        {"test": "starter total pts (compressed) vs runs allowed inn 1-5", "n": n_ra, "r": r_ra_t, "p": p_ra_t},
        {"test": "starter total pts (decile) vs runs allowed inn 1-5", "n": n_ra, "r": r_ra_d, "p": p_ra_d},
        {"test": "starter Contact pts vs the starter's own runs", "n": n_ra, "r": r_sr, "p": p_sr},
    ]
    f5 = priced[priced["f5_line"].notna()].copy()
    f5["f5_resid"] = f5["f5_runs"] - f5["f5_line"]
    f5["contact_sum"] = f5["away_sp_cmp_contact"] + f5["home_sp_cmp_contact"]
    f5["total_sum"] = f5["away_sp_cmp_total"] + f5["home_sp_cmp_total"]
    r_f5, p_f5, n_f5 = corr(f5["contact_sum"], f5["f5_resid"])
    r_f5t, p_f5t, _ = corr(f5["total_sum"], f5["f5_resid"])
    fg = priced[priced["total_line"].notna()].copy()
    fg["tot_resid"] = fg["total_runs"] - fg["total_line"]
    fg["contact_sum"] = fg["away_sp_cmp_contact"] + fg["home_sp_cmp_contact"]
    r_fg, p_fg, n_fg = corr(fg["contact_sum"], fg["tot_resid"])
    r_f5_raw, _, _ = corr(f5["contact_sum"], f5["f5_runs"])
    r_fg_raw, _, _ = corr(fg["contact_sum"], fg["total_runs"])
    h2_rows += [
        {"test": "both starters' Contact pts vs F5 total residual (runs - line)", "n": n_f5, "r": r_f5, "p": p_f5},
        {"test": "both starters' total pts vs F5 total residual", "n": n_f5, "r": r_f5t, "p": p_f5t},
        {"test": "both starters' Contact pts vs full-game total residual", "n": n_fg, "r": r_fg, "p": p_fg},
        {"test": "both starters' Contact pts vs raw F5 runs", "n": n_f5, "r": r_f5_raw, "p": math.nan},
        {"test": "both starters' Contact pts vs raw full-game runs", "n": n_fg, "r": r_fg_raw, "p": math.nan},
    ]
    h2df = pd.DataFrame(h2_rows)
    h2df.to_csv(out / "h2_runs.csv", index=False)
    md.append(_md_table(h2df, [("test", "test", 0), ("n", "n", -1), ("r", "r", 3), ("p", "p", 3)]))
    both_top = f5[(f5["away_sp_contact_tercile"] == 2) & (f5["home_sp_contact_tercile"] == 2)]
    under = (both_top["f5_runs"] < both_top["f5_line"]).astype(float)
    under[both_top["f5_runs"] == both_top["f5_line"]] = np.nan
    u_units = [
        math.nan if math.isnan(u) else (profit(a) if u == 1 else -1.0)
        for u, a in zip(under, both_top["f5_under_am"], strict=True)
    ]
    u_m, u_lo, u_hi = boot_mean_ci(under.to_numpy(), boot)
    u_mkt = float(both_top["f5_under_prob"].mean()) if len(both_top) else math.nan
    md.append(f"\nBlind F5 Under when both starters are top-tercile Contact: n={len(both_top)}, "
              f"Under {_pct(u_m)} [{_pct(u_lo)}, {_pct(u_hi)}] vs market Under prob {_pct(u_mkt)}, "
              f"units {_f(float(np.nansum(u_units)), 2)}.\n")
    h2_ok = r_f5 <= -0.15 and u_m >= 0.55 and abs(r_fg) < abs(r_f5)
    md.append(f"Pre-registered: r(Contact sum, F5 residual) <= -0.15 (got {_f(r_f5)}), blind Under >= 55% "
              f"(got {_pct(u_m)}), full-game effect weaker (|r| {_f(abs(r_fg))} vs {_f(abs(r_f5))}). "
              f"n for r=-0.15 at 80% power: {n_for_corr(0.15)}; have {n_f5}.\n"
              f"Runs-allowed part: r(Contact pts, RA 1-5) = {_f(r_ra)} (p={_f(p_ra)}, n={n_ra}).\n"
              f"**Verdict: {_verdict(h2_ok, n_f5 >= n_for_corr(0.15) or h2_ok)}**\n")

    # H3 -------------------------------------------------------------------------------
    md.append("## H3 - The moneyline prices Whiff fully and Contact only partly\n")
    regs = market_regressions(priced, SP_LABELS)
    if regs:
        m = regs["market_on_gaps"].drop(index="const")
        m["group"] = ["contact" if i in CONTACT else "whiff" for i in m.index]
        m["abs_coef"] = m["coef"].abs()
        m = m.sort_values("abs_coef", ascending=False)
        m.to_csv(out / "h3_market_on_gaps.csv")
        md.append(f"(a) No-vig away ML prob on the 11 standardized metric gaps, n={regs['market_on_gaps'].attrs['n']} "
                  f"(coef = change in market prob per 1 SD of gap):\n")
        m2 = m.reset_index().rename(columns={"index": "metric"})
        md.append(_md_table(m2, [("metric", "metric", 0), ("group", "group", 0), ("coef", "coef", 4),
                                 ("se", "se", 4), ("t", "t", 2), ("p", "p", 3)]))
        top3 = list(m.index[:3])
        bot3 = list(m.index[-3:])
        w_reg = regs["win_on_gaps_and_market"].drop(index="const")
        w_reg["group"] = ["contact" if i in CONTACT else "whiff" if i in WHIFF else "-" for i in w_reg.index]
        w_reg.to_csv(out / "h3_win_on_gaps_market.csv")
        rd_reg = regs["run_diff_on_gaps_and_market"].drop(index="const")
        rd_reg.to_csv(out / "h3_rundiff_on_gaps_market.csv")
        md.append("\n(b) Away win on the gaps + market logit (linear probability; coef per 1 SD):\n")
        md.append(_md_table(w_reg.reset_index().rename(columns={"index": "metric"}),
                            [("metric", "metric", 0), ("group", "group", 0), ("coef", "coef", 4),
                             ("se", "se", 4), ("t", "t", 2), ("p", "p", 3)]))
        md.append("\n(c) Run differential on the gaps + market logit:\n")
        rd2 = rd_reg.copy()
        rd2["group"] = w_reg["group"]
        md.append(_md_table(rd2.reset_index().rename(columns={"index": "metric"}),
                            [("metric", "metric", 0), ("group", "group", 0), ("coef", "coef", 4),
                             ("se", "se", 4), ("t", "t", 2), ("p", "p", 3)]))
        contact_keep = [i for i in CONTACT if i in w_reg.index and w_reg.loc[i, "p"] < 0.10]
        whiff_keep = [i for i in WHIFF if i in w_reg.index and w_reg.loc[i, "p"] < 0.10]
        a_ok = all(t in WHIFF for t in top3) and all(b in CONTACT for b in bot3)
        b_ok = len(contact_keep) > 0 and len(whiff_keep) == 0
        md.append(f"\nLargest market coefficients: {', '.join(top3)}; smallest: {', '.join(bot3)}. "
                  f"Expected top-3 all Whiff and bottom-3 all Contact: {'yes' if a_ok else 'no'}.\n"
                  f"Contact metrics keeping p<0.10 on win after the market: {contact_keep or 'none'}; "
                  f"Whiff metrics keeping p<0.10: {whiff_keep or 'none'}.\n"
                  f"**Verdict (a, pricing order): {_verdict(a_ok)}**; "
                  f"**Verdict (b, residual signal): {_verdict(b_ok, len(contact_keep) > 0 or len(whiff_keep) > 0)}**. "
                  f"With ~{regs['market_on_gaps'].attrs['n']} games a per-metric coefficient needs |r| >= "
                  f"{_f(2.8 / math.sqrt(regs['market_on_gaps'].attrs['n']), 2)} to reach p<0.05.\n")
    else:
        md.append("Not enough priced games for the regressions.\n")

    # H4 -------------------------------------------------------------------------------
    md.append("## H4 - 'Elite vs poor' pairings: does any (elite pts, poor pts) cell beat the price?\n")
    cells = pairing_cells(priced, boot)
    if len(cells):
        cells = cells.sort_values("n", ascending=False)
        cells.to_csv(out / "h4_pairing_cells.csv", index=False)
        bands = cells.attrs["bands"]
        bands.to_csv(out / "h4_gap_bands.csv", index=False)
        md.append("SP-gap bands (favoured side = higher compressed SP total):\n")
        md.append(_md_table(bands, [("gap_band", "gap", 0), ("n", "n", -1), ("win_pct", "win", None),
                                    ("market_prob", "mkt", None), ("residual", "resid", 4), ("residual_lo", "lo", 4),
                                    ("residual_hi", "hi", 4), ("p", "p", 3), ("ml_units", "ML u", 2),
                                    ("rl_units", "RL u", 2)]))
        md.append(f"\nCells with n >= 8 (of {len(cells)}), sorted by n:\n")
        md.append(_md_table(cells.head(25), [("elite_pts", "elite", -1), ("poor_pts", "poor", -1), ("n", "n", -1),
                                             ("elite_wins", "W", -1), ("win_pct", "win", None),
                                             ("market_prob", "mkt", None), ("residual", "resid", 4),
                                             ("residual_lo", "lo", 4), ("residual_hi", "hi", 4), ("p", "p", 3),
                                             ("ml_units", "ML u", 2)]))
        sig = cells[(cells["residual_lo"] > 0) | (cells["residual_hi"] < 0)]
        tests = len(cells)
        bonf = cells[cells["p"] < 0.05 / max(tests, 1)]
        eight = cells[(cells["poor_pts"] == 8)]
        md.append(f"\nCells whose 95% CI excludes 0: {len(sig)} of {tests} "
                  f"(expect ~{0.05 * tests:.1f} by chance); surviving Bonferroni (p < {0.05 / max(tests, 1):.4f}): "
                  f"{len(bonf)}. The 'poor = 8' cells over the season: n={int(eight['n'].sum()) if len(eight) else 0}, "
                  f"residual {_f(float((eight['residual'] * eight['n']).sum() / eight['n'].sum()), 4) if len(eight) else '-'}.\n"
                  f"**Verdict: {'rejected (no cell survives multiplicity)' if len(bonf) == 0 else 'not independently supported: ' + ', '.join(f'({int(r.elite_pts)},{int(r.poor_pts)}) n={int(r.n)}' for r in bonf.itertuples()) + ' survives Bonferroni, but with n < 20 and cells chosen after seeing 09-14..22 this is a candidate for out-of-sample re-test, not a finding'}** "
                  f"-- a cell needs ~{n_for_mean(0.10, 0.50):.0f} games to confirm a +10pt residual.\n")
    else:
        md.append("No priced pairings.\n")

    # H5 -------------------------------------------------------------------------------
    md.append("## H5 - Does percentile / z scoring recover signal that +2/+1/-2 compresses away?\n")
    h5_rows = []
    for scheme, lbl in (("cmp", "compressed (+2/+1/-2)"), ("dec", "decile 0-9"), ("z", "z-score")):
        r_w, p_w, n_w = corr(df[f"sp_gap_{scheme}"], df["away_win"])
        r_rd, _, _ = corr(df[f"sp_gap_{scheme}"], df["run_diff"])
        r_ra5, p_ra5, n_ra5 = corr(starters[f"{scheme}_total"], starters["ra_f5"])
        r_own, _, _ = corr(starters[f"{scheme}_total"], starters["sp_runs"])
        r_mk, _, _ = corr(priced[f"sp_gap_{scheme}"], priced["ml_prob_away"])
        h5_rows.append({"table": "starters", "scheme": lbl, "n_games": n_w, "r_win": r_w, "p_win": p_w,
                        "r_run_diff": r_rd, "r_market": r_mk, "n_starts": n_ra5, "r_ra_1_5": r_ra5, "p_ra": p_ra5,
                        "r_own_runs": r_own})
    for scheme, lbl in (("cmp", "compressed (+2/+1/-2)"), ("dec", "decile 0-9"), ("z", "z-score")):
        col = f"{bp_key}_gap_{scheme}"
        r_w, p_w, n_w = corr(df[col], df["away_win"])
        r_rd, _, _ = corr(df[col], df["run_diff"])
        r_bp, p_bp, n_bp = corr(pens[f"{scheme}_total"], pens["bp_ra9"])
        r_mk, _, _ = corr(priced[col], priced["ml_prob_away"])
        h5_rows.append({"table": "bullpens", "scheme": lbl, "n_games": n_w, "r_win": r_w, "p_win": p_w,
                        "r_run_diff": r_rd, "r_market": r_mk, "n_starts": n_bp, "r_ra_1_5": r_bp, "p_ra": p_bp,
                        "r_own_runs": math.nan})
    h5df = pd.DataFrame(h5_rows)
    h5df.to_csv(out / "h5_schemes.csv", index=False)
    md.append("For starters `r_ra` is the correlation of the starter's points with the runs his team allowed "
              "in innings 1-5; for bullpens it is with the pen's runs per 27 outs that night.\n")
    md.append(_md_table(h5df, [("table", "table", 0), ("scheme", "scheme", 0), ("n_games", "games", -1),
                               ("r_win", "r win", 3), ("p_win", "p", 3), ("r_run_diff", "r run diff", 3),
                               ("r_market", "r mkt", 3), ("n_starts", "n appear.", -1), ("r_ra_1_5", "r RA", 3),
                               ("p_ra", "p", 3), ("r_own_runs", "r own runs", 3)]))
    dec = h5df[(h5df["table"] == "starters") & (h5df["scheme"] == "decile 0-9")].iloc[0]
    cmp_ = h5df[(h5df["table"] == "starters") & (h5df["scheme"] == "compressed (+2/+1/-2)")].iloc[0]
    h5_ok = abs(dec["r_win"]) >= 0.15 and dec["r_ra_1_5"] <= -0.25
    md.append(f"\nPre-registered: percentile gap |r(win)| >= 0.15 (got {_f(dec['r_win'])}, compressed {_f(cmp_['r_win'])}) "
              f"and r(RA 1-5) <= -0.25 (got {_f(dec['r_ra_1_5'])}, compressed {_f(cmp_['r_ra_1_5'])}).\n"
              f"**Verdict: {_verdict(h5_ok)}** -- with n={int(dec['n_games'])} games the SE of r is about "
              f"{_f(1 / math.sqrt(max(int(dec['n_games']), 2)), 3)}, so differences between schemes smaller than "
              f"~{_f(2 / math.sqrt(max(int(dec['n_games']), 2)), 2)} are not distinguishable.\n")

    # H6 -------------------------------------------------------------------------------
    md.append("## H6 - Does the run line, not the moneyline, monetise the SP edge?\n")
    o = side_outcomes(priced, priced["sp_gap_contact"])
    o_dec = o[o["decided"]]
    favs = o_dec[(o_dec["mkt"] >= 0.5) & o_dec["rl_units"].notna()]
    dogs = o_dec[(o_dec["mkt"] < 0.5) & o_dec["rl_units"].notna()]
    fav_cov, fav_lo, fav_hi = boot_mean_ci(favs["rl_cover"].to_numpy(), boot)
    fav_u, fav_u_lo, fav_u_hi = boot_mean_ci(favs["rl_units"].to_numpy(), boot)
    dog_cov, dog_lo, dog_hi = boot_mean_ci(dogs["rl_cover"].to_numpy(), boot)
    avg_price = float(favs["rl_price"].mean()) if len(favs) else math.nan
    be = 1 / (1 + profit(avg_price)) if not math.isnan(avg_price) else math.nan
    c = h1["Contact gap"]
    h6_rows = [
        {"slice": "Contact side, ML", "n": c["n_priced"], "hit": c["side_win_pct_priced"], "units": c["ml_units"],
         "roi": c["ml_roi"]},
        {"slice": "Contact side, RL (fav -1.5 / dog +1.5)", "n": c["n_rl"], "hit": c["rl_cover_pct"],
         "units": c["rl_units"], "roi": c["rl_roi"]},
        {"slice": "Contact side favourites laying -1.5", "n": len(favs), "hit": fav_cov, "units": favs["rl_units"].sum(),
         "roi": fav_u},
        {"slice": "Contact side dogs taking +1.5", "n": len(dogs), "hit": dog_cov, "units": dogs["rl_units"].sum(),
         "roi": float(dogs["rl_units"].mean()) if len(dogs) else math.nan},
    ]
    h6df = pd.DataFrame(h6_rows)
    h6df.to_csv(out / "h6_rl_vs_ml.csv", index=False)
    md.append(_md_table(h6df, [("slice", "slice", 0), ("n", "n", -1), ("hit", "hit", None), ("units", "units", 2),
                               ("roi", "ROI/u", 4)]))
    h6_ok = (c["rl_units"] > c["ml_units"]) and fav_cov >= 0.47
    md.append(f"\nFavourites laying -1.5: cover {_pct(fav_cov)} [{_pct(fav_lo)}, {_pct(fav_hi)}], avg price "
              f"{avg_price:+.0f} (break-even {_pct(be)}), ROI {fav_u:+.4f} [{fav_u_lo:+.4f}, {fav_u_hi:+.4f}]. "
              f"Dogs +1.5 cover {_pct(dog_cov)} [{_pct(dog_lo)}, {_pct(dog_hi)}].\n"
              f"Pre-registered: RL units > ML units ({_f(c['rl_units'], 2)} vs {_f(c['ml_units'], 2)}) and fav -1.5 cover >= 47%.\n"
              f"**Verdict: {_verdict(h6_ok, len(favs) >= n_for_mean(0.05))}**\n")

    # per-metric tables -----------------------------------------------------------------
    md.append("## Per-metric signals, starters (priced window; gap = away - home, signed so + is better)\n")
    spm = metric_table(priced, "sp", SP_LABELS, boot)
    spm.to_csv(out / "metric_table_sp.csv")
    md.append(_md_table(spm.reset_index(), SIGNAL_COLS))
    md.append("\n## Per-metric signals, bullpens (priced window)\n")
    bpm = metric_table(priced, "bp", BP_LABELS, boot)
    bpm.to_csv(out / "metric_table_bp.csv")
    md.append(_md_table(bpm.reset_index(), SIGNAL_COLS))
    # starter-level per metric vs runs allowed
    md.append("\n## Per-metric, starter level: raw metric vs runs allowed innings 1-5 (full season)\n")
    rows = []
    for label in SP_LABELS:
        if label not in starters:
            continue
        s = -starters[label] if SP_LOWER[label] else starters[label]
        r1, p1, n1 = corr(s, starters["ra_f5"])
        r2, _, _ = corr(s, starters["sp_runs"])
        r3, p3, _ = corr(starters[f"cmp_{label}"], starters["ra_f5"])
        r4, _, _ = corr(starters[f"dec_{label}"], starters["ra_f5"])
        rows.append({"metric": label, "group": "contact" if label in CONTACT else "whiff", "n": n1,
                     "r_raw_ra15": r1, "p": p1, "r_raw_own_runs": r2, "r_cmp_pts_ra15": r3, "r_dec_pts_ra15": r4})
    spr = pd.DataFrame(rows)
    spr.to_csv(out / "metric_table_sp_runs.csv", index=False)
    md.append(_md_table(spr, [("metric", "metric", 0), ("group", "group", 0), ("n", "n", -1),
                              ("r_raw_ra15", "r raw vs RA1-5", 3), ("p", "p", 3),
                              ("r_raw_own_runs", "r raw vs own runs", 3), ("r_cmp_pts_ra15", "r +2/-2 pts", 3),
                              ("r_dec_pts_ra15", "r decile pts", 3)]))
    rows = []
    for label in BP_LABELS:
        if label not in pens:
            continue
        s = -pens[label] if BP_LOWER[label] else pens[label]
        r1, p1, n1 = corr(s, pens["bp_ra9"])
        r3, _, _ = corr(pens[f"cmp_{label}"], pens["bp_ra9"])
        rows.append({"metric": label, "n": n1, "r_raw_ra9": r1, "p": p1, "r_cmp_pts_ra9": r3})
    bpr = pd.DataFrame(rows)
    bpr.to_csv(out / "metric_table_bp_runs.csv", index=False)
    md.append("\n## Per-metric, bullpen level: raw metric vs the pen's runs per 27 outs that night\n")
    md.append(_md_table(bpr, [("metric", "metric", 0), ("n", "n", -1), ("r_raw_ra9", "r raw vs RA/27", 3),
                              ("p", "p", 3), ("r_cmp_pts_ra9", "r +2/-2 pts", 3)]))
    text = "\n".join(md) + "\n"
    (out / report_name).write_text(text)
    return text
