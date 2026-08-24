"""Fit and grade the batter-prop over correction that ``models.run_env`` ships.

Offline, ledger-only, no odds credits. Two terms are fitted in over-space by log
loss on earlier slates and scored on later ones:

    logit(p_over') = logit(p_over) - tilt - slope * elevation

``elevation`` is the simulator's implied game-total mean minus the league's run
level. The ledger does not store the simulator's mean, so it is recovered per
game from the four ``game_total`` over-probabilities by inverting a negative
binomial whose dispersion is fitted globally -- the study's one real proxy, and
the reason the shipped defaults sit at the conservative end of the fitted range.

    python -m scripts.run_env_study --ledger ~/.mlb_engine/audit/ledger.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nbinom

from mlb_engine.models.run_env import LEAGUE_TOTAL_BASELINE, MAX_ELEVATION

EPS = 1e-6
DISPERSIONS = (3, 5, 8, 12, 20, 40, 80)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _nb_sf(mu: float, r: float, k: int) -> float:
    return float(nbinom.sf(k - 1, r, r / (r + mu)))


def _fit_mu(lines: list[float], probs: list[float], r: float) -> float:
    def obj(mu: float) -> float:
        return sum(_nb_sf(mu, r, int(np.ceil(ln))) - p for ln, p in zip(lines, probs, strict=True))

    try:
        return brentq(obj, 1.0, 30.0)
    except ValueError:
        return float("nan")


def sim_means(led: pd.DataFrame) -> pd.DataFrame:
    """Simulator game-total mean per (date, matchup), inverted from the card."""
    g = led[led["market"] == "game_total"].copy()
    g["prob"] = g["raw_prob"].fillna(g["model_prob"])
    over = g[g["selection"].str.startswith("Over")]
    over = over[(over["prob"] > 0.001) & (over["prob"] < 0.999)]
    cards = [
        (d, m, list(x["line"]), list(x["prob"]))
        for (d, m), x in over.groupby(["date", "matchup"])
        if len(x) >= 3
    ]
    best_r, best_sse = DISPERSIONS[0], None
    for r in DISPERSIONS:
        sse = 0.0
        for _d, _m, lines, probs in cards:
            mu = _fit_mu(lines, probs, r)
            if np.isnan(mu):
                sse += 1.0
                continue
            sse += sum((_nb_sf(mu, r, int(np.ceil(ln))) - p) ** 2 for ln, p in zip(lines, probs, strict=True))
        if best_sse is None or sse < best_sse:
            best_r, best_sse = r, sse
    print(f"  dispersion r={best_r} (fit SSE {best_sse:.3f} over {len(cards)} cards)")
    rows = [
        {"date": d, "matchup": m, "sim_mu": _fit_mu(lines, probs, best_r)}
        for d, m, lines, probs in cards
    ]
    return pd.DataFrame(rows).dropna()


def batter_props(ledger: Path) -> pd.DataFrame:
    led = pd.read_csv(ledger, low_memory=False)
    mu = sim_means(led).set_index(["date", "matchup"])["sim_mu"]
    p = led[led["market"].str.startswith("batter_", na=False)].copy()
    p = p[p["result"].isin(["win", "loss"])].copy()
    p["y"] = (p["result"] == "win").astype(int)
    p["sim_mu"] = [mu.get(k, np.nan) for k in zip(p["date"], p["matchup"], strict=True)]
    p = p[p["sim_mu"].notna()].copy()
    p["e"] = (p["sim_mu"] - LEAGUE_TOTAL_BASELINE).clip(-MAX_ELEVATION, MAX_ELEVATION)
    p["is_over"] = p["selection"].str.contains(r" o\d", regex=True)
    p = p[p["is_over"] | p["selection"].str.contains(r" u\d", regex=True)].copy()
    p["p_over"] = np.where(p["is_over"], p["model_prob"], 1 - p["model_prob"])
    p["y_over"] = np.where(p["is_over"], p["y"], 1 - p["y"])
    return p


def apply_terms(p_over: np.ndarray, e: np.ndarray, tilt: float, slope: float) -> np.ndarray:
    return _sigmoid(_logit(p_over) - tilt - slope * e)


def fit_terms(tr: pd.DataFrame) -> tuple[float, float]:
    """Grid search both terms by log loss. Coarse on purpose: the fit is refit
    weekly in the walk-forward and only its order of magnitude survives."""
    po, e, y = tr["p_over"].to_numpy(), tr["e"].to_numpy(), tr["y_over"].to_numpy()
    best = (0.0, 0.0, log_loss(po, y))
    for tilt in np.arange(-0.05, 0.51, 0.01):
        for slope in np.arange(-0.02, 0.16, 0.01):
            ll = log_loss(apply_terms(po, e, tilt, slope), y)
            if ll < best[2]:
                best = (float(tilt), float(slope), ll)
    return round(best[0], 2), round(best[1], 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="~/.mlb_engine/audit/ledger.csv")
    ap.add_argument("--block-days", type=int, default=7)
    args = ap.parse_args()

    p = batter_props(Path(args.ledger).expanduser())
    dates = sorted(p["date"].unique())
    print(f"{len(p)} graded batter prop rows over {len(dates)} slates, e sd {p['e'].std():.2f}")

    rows = []
    blocks = [dates[i : i + args.block_days] for i in range(args.block_days, len(dates), args.block_days)]
    for blk in blocks:
        tr, te = p[p["date"] < blk[0]], p[p["date"].isin(blk)]
        if len(tr) < 5000 or len(te) < 500:
            continue
        tilt, slope = fit_terms(tr)
        po, e, y = te["p_over"].to_numpy(), te["e"].to_numpy(), te["y_over"].to_numpy()
        pc = apply_terms(po, e, tilt, slope)
        rows.append(
            {
                "block": blk[0],
                "n": len(te),
                "tilt": tilt,
                "slope": slope,
                "ll_base": log_loss(po, y),
                "ll_corr": log_loss(pc, y),
                "br_base": brier(po, y),
                "br_corr": brier(pc, y),
            }
        )
    w = pd.DataFrame(rows)
    print(w.round(4).to_string(index=False))
    n = w["n"].to_numpy()
    print(
        f"  pooled logloss {np.average(w['ll_base'], weights=n):.4f} -> "
        f"{np.average(w['ll_corr'], weights=n):.4f}, Brier "
        f"{np.average(w['br_base'], weights=n):.4f} -> {np.average(w['br_corr'], weights=n):.4f}, "
        f"better on {int((w['br_corr'] < w['br_base']).sum())}/{len(w)} blocks"
    )

    # Calibration of the side the engine likes, on the second half of the window.
    split = dates[len(dates) // 2]
    tilt, slope = fit_terms(p[p["date"] < split])
    te = p[p["date"] >= split]
    po, e, y = te["p_over"].to_numpy(), te["e"].to_numpy(), te["y"].to_numpy()
    print(f"\nholdout >= {split} (n={len(te)}), tilt={tilt} slope={slope}")
    for name, arr in (("as priced", po), ("corrected", apply_terms(po, e, tilt, slope))):
        row_p = np.where(te["is_over"], arr, 1 - arr)
        for lab, mask in (
            ("overs liked", (row_p >= 0.5) & te["is_over"].to_numpy()),
            ("unders liked", (row_p >= 0.5) & ~te["is_over"].to_numpy()),
        ):
            print(
                f"  {name:10} {lab:12} n={mask.sum():6} said {row_p[mask].mean() * 100:5.1f} "
                f"hit {y[mask].mean() * 100:5.1f} err {(row_p[mask].mean() - y[mask].mean()) * 100:+5.2f}"
            )


if __name__ == "__main__":
    main()
