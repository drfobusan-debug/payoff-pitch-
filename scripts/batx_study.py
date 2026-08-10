"""Evaluate THE BAT X daily projections against our model and against the market.

THE BAT X (Derek Carty) publishes a per-game projected batting line -- PA, AB,
1B, 2B, 3B, HR, BB, K, R, RBI -- which is exactly the input our batter prop
markets need. This script turns that line into over/under probabilities and then
asks the only question worth asking of a new signal: conditional on the market
price, does it carry information our model does not?

The test is deliberately the same one that condemned our own edge: fit

    logit(win) ~ a + b*logit(model) + c*logit(market) + d*logit(batx)

on graded rows. A useful forecast scores a positive coefficient *next to* the
price. Absolute hit rate and ROI are not evidence -- they are dominated by which
props happened to be offered.

Usage::

    # game day, before the slate: turn the BAT X csv into probabilities
    python scripts/batx_study.py price --hitters DKHitters.csv --date 2026-08-10 \\
        --out ~/.mlb_engine/batx/2026-08-10.csv

    # after the slate is graded: join to the ledger and run the head-to-head
    python scripts/batx_study.py grade --probs '~/.mlb_engine/batx/*.csv'

The projections are a *mean* line. Turning a mean into P(over) needs a
distribution, so:

* plate appearances use a two-point distribution straddling the projected mean
  (4.15 PA -> 85% chance of 4, 15% chance of 5), which matches the mean exactly
  and reflects that a hitter takes a whole number of turns;
* hits and total bases come from an exact convolution over those PA of the
  per-PA outcome vector (1B/2B/3B/HR/other), so they share the right joint --
  a home run lifts both;
* runs and RBI have no per-PA decomposition in the feed, so they fall back to a
  Poisson on the projected mean.

``batter_hrr`` (H+R+RBI) is the weakest of the set: it convolves the exact hit
distribution with independent Poissons, and hits/runs/RBI are emphatically not
independent -- a home run scores all three at once. Independence understates the
variance, so P(over) on the combo is biased toward the mean. It is reported, but
it is the one number here not to trust.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

LEDGER = os.path.expanduser("~/.mlb_engine/audit/ledger.csv")

# Per-PA outcome columns we need off the BAT X hitters export, mapped to the
# base value each one is worth. Anything else in the row is a rate, a context
# flag, or a DFS scoring artifact and is ignored.
HIT_BASES = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}

# Ledger market -> the BAT X quantity that prices it.
MARKETS = ("batter_h", "batter_1b", "batter_2b", "batter_hr", "batter_tb", "batter_r", "batter_rbi", "batter_hrr")

# Trailing tokens the engine appends to a player's name in ``selection``.
_SEL_SUFFIX = re.compile(r"\s+(H\+R\+RBI|1B|2B|3B|HR|TB|H|R|RBI)\s+[ou][\d.]+$")


def norm_name(name: str) -> str:
    """Strip accents and case so 'Julio Rodríguez' joins to 'Julio Rodriguez'."""
    n = unicodedata.normalize("NFKD", str(name))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r"[.'`-]", "", n)
    n = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", n.strip().lower())
    return re.sub(r"\s+", " ", n)


def player_from_selection(selection: str) -> str:
    return norm_name(_SEL_SUFFIX.sub("", str(selection)))


def pa_distribution(mean_pa: float) -> dict[int, float]:
    """Two-point distribution over whole plate appearances matching ``mean_pa``."""
    if not math.isfinite(mean_pa) or mean_pa <= 0:
        return {0: 1.0}
    lo = int(math.floor(mean_pa))
    frac = mean_pa - lo
    if frac <= 1e-9:
        return {lo: 1.0}
    return {lo: 1.0 - frac, lo + 1: frac}


def hit_tb_distribution(mean_pa: float, rates: dict[str, float]) -> dict[tuple[int, int], float]:
    """Exact joint distribution of (hits, total bases) over the PA distribution.

    ``rates`` holds per-PA probabilities for 1B/2B/3B/HR; the remainder is any
    non-hit outcome, which advances neither count.
    """
    p_hit = {b: max(0.0, rates.get(k, 0.0)) for k, b in HIT_BASES.items()}
    p_none = max(0.0, 1.0 - sum(p_hit.values()))

    joint: dict[tuple[int, int], float] = {}
    for n_pa, w_pa in pa_distribution(mean_pa).items():
        state: dict[tuple[int, int], float] = {(0, 0): 1.0}
        for _ in range(n_pa):
            nxt: dict[tuple[int, int], float] = {}
            for (h, tb), w in state.items():
                nxt[(h, tb)] = nxt.get((h, tb), 0.0) + w * p_none
                for bases, p in p_hit.items():
                    if p <= 0.0:
                        continue
                    key = (h + 1, tb + bases)
                    nxt[key] = nxt.get(key, 0.0) + w * p
            state = nxt
        for key, w in state.items():
            joint[key] = joint.get(key, 0.0) + w_pa * w
    return joint


def p_at_least(dist: dict[int, float], threshold: int) -> float:
    return float(sum(w for k, w in dist.items() if k >= threshold))


def marginal(joint: dict[tuple[int, int], float], axis: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, w in joint.items():
        out[key[axis]] = out.get(key[axis], 0.0) + w
    return out


def binomial_at_least_one(mean_pa: float, rate: float) -> float:
    """P(count >= 1) for a per-PA Bernoulli, averaged over the PA distribution."""
    if rate <= 0.0:
        return 0.0
    return float(sum(w * (1.0 - (1.0 - min(rate, 1.0)) ** n) for n, w in pa_distribution(mean_pa).items()))


def convolve(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for ka, wa in a.items():
        for kb, wb in b.items():
            out[ka + kb] = out.get(ka + kb, 0.0) + wa * wb
    return out


def poisson_pmf(mean: float, cap: int = 12) -> dict[int, float]:
    mean = max(0.0, float(mean))
    return {k: float(poisson.pmf(k, mean)) for k in range(cap + 1)}


def price_row(row: pd.Series) -> dict[str, float]:
    """All batter-prop over probabilities implied by one BAT X projected line."""
    pa = float(row["PA"])
    rates = {k: (float(row[k]) / pa if pa > 0 else 0.0) for k in HIT_BASES}
    joint = hit_tb_distribution(pa, rates)
    hits = marginal(joint, 0)
    bases = marginal(joint, 1)

    r_mean, rbi_mean = float(row["R"]), float(row["RBI"])
    hrr = convolve(convolve(hits, poisson_pmf(r_mean)), poisson_pmf(rbi_mean))

    return {
        "batter_h@0.5": p_at_least(hits, 1),
        "batter_h@1.5": p_at_least(hits, 2),
        "batter_1b@0.5": binomial_at_least_one(pa, rates["1B"]),
        "batter_2b@0.5": binomial_at_least_one(pa, rates["2B"]),
        "batter_hr@0.5": binomial_at_least_one(pa, rates["HR"]),
        "batter_tb@0.5": p_at_least(bases, 1),
        "batter_tb@1.5": p_at_least(bases, 2),
        "batter_tb@2.5": p_at_least(bases, 3),
        "batter_tb@3.5": p_at_least(bases, 4),
        "batter_r@0.5": p_at_least(poisson_pmf(r_mean), 1),
        "batter_rbi@0.5": p_at_least(poisson_pmf(rbi_mean), 1),
        "batter_hrr@1.5": p_at_least(hrr, 2),
        "batter_hrr@2.5": p_at_least(hrr, 3),
    }


# DraftKings MLB hitter scoring. Used only to verify the export's columns are
# mapped correctly: the projected components must reproduce the projected FPTS.
# A misaligned header silently produces plausible-looking nonsense otherwise.
DK_POINTS = {"1B": 3, "2B": 5, "3B": 8, "HR": 10, "RBI": 2, "R": 2, "BB": 2, "HBP": 2, "SB": 5}


def check_schema(df: pd.DataFrame) -> None:
    """Reconcile projected FPTS against the projected components."""
    if "FPTS" not in df.columns:
        print("no FPTS column -- cannot verify the column mapping")
        return
    cols = [c for c in DK_POINTS if c in df.columns]
    rebuilt = sum(df[c].fillna(0.0) * DK_POINTS[c] for c in cols)
    err = (rebuilt - df["FPTS"].fillna(0.0)).abs()
    ok = float((err < 0.25).mean())
    print(f"schema check: rebuilt DK points from {cols}")
    print(f"  median abs error {err.median():.3f} pts, {ok:.1%} of rows within 0.25")
    if ok < 0.9:
        print("  WARNING: components do not reproduce FPTS -- the columns are probably")
        print("  misaligned, and every probability below would be nonsense. Stopping.")
        raise SystemExit(1)


def read_hitters(path: str) -> pd.DataFrame:
    """Load the BAT X hitters export, tolerating case and whitespace in headers."""
    df = pd.read_csv(os.path.expanduser(path))
    df.columns = [str(c).strip().upper() for c in df.columns]
    name_col = next((c for c in ("NAME", "PLAYER", "PLAYER_NAME") if c in df.columns), None)
    if name_col is None:
        raise SystemExit(f"no player-name column in {path}: {list(df.columns)[:12]}")
    missing = [c for c in ("PA", "R", "RBI", *HIT_BASES) if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing projection columns {missing}")
    df = df.rename(columns={name_col: "NAME"})
    df["player"] = df["NAME"].map(norm_name)
    return df


def cmd_price(args: argparse.Namespace) -> None:
    df = read_hitters(args.hitters)
    check_schema(df)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        if not np.isfinite(row.get("PA", np.nan)) or float(row["PA"]) <= 0:
            continue
        for key, prob in price_row(row).items():
            market, line = key.split("@")
            rows.append(
                {
                    "date": args.date,
                    "player": row["player"],
                    "team": row.get("TEAM", ""),
                    "market": market,
                    "line": float(line),
                    "batx_prob": round(prob, 6),
                }
            )
    out = pd.DataFrame(rows)
    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"{len(df)} hitters -> {len(out)} priced selections -> {dest}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain logistic regression with an intercept. Returns (coefs, std errors)."""
    design = np.column_stack([np.ones(len(y)), x])

    def nll(beta: np.ndarray) -> float:
        z = design @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    fit = minimize(nll, np.zeros(design.shape[1]), method="BFGS")
    pred = 1.0 / (1.0 + np.exp(-(design @ fit.x)))
    w = pred * (1 - pred)
    cov = np.linalg.pinv(design.T @ (design * w[:, None]))
    return fit.x, np.sqrt(np.diag(cov))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def cmd_grade(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(os.path.expanduser(args.probs)))
    if not paths:
        raise SystemExit(f"no BAT X probability files match {args.probs}")
    batx = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["date", "player", "market", "line"]
    dupes = int(batx.duplicated(key).sum())
    if dupes:
        # Same player twice on a date (doubleheaders, or overlapping exports).
        # Keeping both would double-count those rows in the fit.
        print(f"dropping {dupes} duplicate projections on {key}")
        batx = batx.drop_duplicates(key)

    led = pd.read_csv(os.path.expanduser(args.ledger))
    led = led[led.market.isin(MARKETS) & led.result.isin(["win", "loss"])].copy()
    led["player"] = led.selection.map(player_from_selection)
    led["y"] = (led.result == "win").astype(float)

    df = led.merge(batx, on=["date", "player", "market", "line"], how="inner")
    if df.empty:
        raise SystemExit("no ledger rows joined to a BAT X projection -- check dates and name normalisation")

    joined_dates = sorted(df.date.unique())
    print(f"joined {len(df)} graded rows over {len(joined_dates)} slates ({joined_dates[0]}..{joined_dates[-1]})")
    print(f"  unmatched ledger rows: {len(led) - len(df)}")

    print("\ncalibration (mean predicted vs actual)")
    print(f"  {'source':<8} {'mean p':>8} {'actual':>8} {'gap':>8} {'brier':>8}")
    for label, col in (("batx", "batx_prob"), ("model", "model_prob"), ("market", "fair_prob")):
        if col not in df.columns:
            continue
        sub = df[np.isfinite(df[col])]
        if sub.empty:
            print(f"  {label:<8} {'--':>8} (no rows)")
            continue
        p, yy = sub[col].to_numpy(), sub.y.to_numpy()
        print(f"  {label:<8} {p.mean():8.3f} {yy.mean():8.3f} {p.mean() - yy.mean():+8.3f} {brier(p, yy):8.4f}")

    priced = df[np.isfinite(df.fair_prob) & np.isfinite(df.model_prob) & np.isfinite(df.batx_prob)]
    print(f"\nhead-to-head on {len(priced)} rows carrying a real market price")
    if len(priced) < 100:
        print("  too few priced rows to read anything into the coefficients")
    if priced.empty:
        return
    cols = ["model_prob", "fair_prob", "batx_prob"]
    x = np.column_stack([logit(priced[c].to_numpy()) for c in cols])
    beta, se = fit_logit(x, priced.y.to_numpy())
    print(f"  {'term':<14} {'coef':>8} {'se':>7}")
    print(f"  {'intercept':<14} {beta[0]:8.3f} {se[0]:7.3f}")
    for name, b, s in zip(cols, beta[1:], se[1:], strict=True):
        print(f"  {name:<14} {b:8.3f} {s:7.3f}")
    print("\n  a forecast with information the others lack scores a positive coefficient here")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("price", help="turn a BAT X hitters export into over probabilities")
    p.add_argument("--hitters", required=True, help="path to the BAT X hitters CSV export")
    p.add_argument("--date", required=True, help="slate date the export covers (YYYY-MM-DD)")
    p.add_argument("--out", required=True, help="where to write the priced selections")
    p.set_defaults(func=cmd_price)

    g = sub.add_parser("grade", help="join priced selections to graded ledger rows")
    g.add_argument("--probs", required=True, help="glob of files written by 'price'")
    g.add_argument("--ledger", default=LEDGER)
    g.set_defaults(func=cmd_grade)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
