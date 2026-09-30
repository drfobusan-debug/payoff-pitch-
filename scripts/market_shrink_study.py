"""Fit the weight the model's disagreement with the price deserves, out of time.

Every market the engine bets is optimistic on the side it backs: across 20,853
graded rows with a two-sided price, realized outcomes land 7-13 points under the
model's own probability, and under the devigged price too. That is a calibration
statement, so the candidate fix is a calibration one -- the shrink the engine
already implements as :func:`mlb_engine.market.ev.anchor_to_market`:

    p_shrunk = fair + alpha * (model - fair)      alpha = 1 - market_anchor

This script fits ``alpha`` from the ledger and grades it on rows the fit never
saw. What it does *not* do is turn anything on. Two reasons:

1. **The screen is affine in the probability**, so shrinking scales every
   measured edge by ``alpha``. Against fixed floors that is arithmetically
   identical to demanding ``min_edge / alpha`` -- at the fitted alphas the card's
   own thresholds delete almost every buy. A weight cannot be adopted without
   refitting the edge floor, price band and probability floor beside it.
2. **The gain is in calibration, not in selection.** A better probability on
   20,000 rows the engine never bet is not a better bet list.

The test that decides an alpha is whether the shrunken number beats *the price
alone* out of sample. If it does not, the honest read is alpha = 0 -- the
disagreement carries nothing -- rather than "the model needs a coefficient".
A single 70/30 cut puts that verdict on the last two slates, so the walk-forward
mode refits every day on everything before it and is the one to believe.

``--write-anchors`` persists the fitted weights to the file ``Config.anchor_for``
reads, for an operator who has read the output and wants them live. It is a
separate, explicit step precisely because the numbers below do not justify
themselves; deleting the file restores the packaged defaults.

Usage::

    python scripts/market_shrink_study.py --ledger ~/.mlb_engine/audit/ledger.csv
    python scripts/market_shrink_study.py --split 0.6 --no-walk-forward
    python scripts/market_shrink_study.py --write-anchors   # opt in, then re-run the card
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.audit.grade import LOSS, WIN
from mlb_engine.audit.ledger import ENGINE, LedgerEntry, load_ledger
from mlb_engine.config import Config
from mlb_engine.market.odds import american_to_decimal
from mlb_engine.market.tiers import Tier

EPS = 1e-6
GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)
MIN_MARKET_ROWS = 25
MIN_TRAIN = 60  # under this a market has no business owning a coefficient
PRIOR_ROWS = 120.0  # pseudo-rows of the pooled alpha mixed into every market
BUY_TIERS = frozenset({Tier.STRONG.value, Tier.MODERATE.value})


def frame(entries: list[LedgerEntry]) -> pd.DataFrame:
    """Graded engine rows that carry both probabilities, oldest first.

    A one-way quote is dropped rather than shrunk: with no second side its
    ``fair_prob`` still holds the vig, so the gap being fitted is part price
    error and the coefficient would absorb it.
    """
    rows = [
        {
            "date": e.date,
            "market": e.market,
            "won": 1 if e.result == WIN else 0,
            "model": e.model_prob,
            "fair": e.fair_prob,
            "odds": e.odds,
            "pnl": e.pnl,
            "buy": e.tier in BUY_TIERS,
        }
        for e in entries
        if e.source == ENGINE
        and e.result in (WIN, LOSS)
        and e.fair_prob is not None
        and e.under_odds is not None
    ]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.clip(p, EPS, 1 - EPS) - y) ** 2))


def shrink(model: np.ndarray, fair: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(fair + alpha * (model - fair), EPS, 1 - EPS)


def fit_alpha(df: pd.DataFrame) -> float:
    """Grid-search alpha on log loss.

    A grid rather than a solver because the objective is nearly flat near the
    optimum -- the printed curve says more about how much the data knows than
    the argmin does.
    """
    y = df["won"].to_numpy()
    m, f = df["model"].to_numpy(), df["fair"].to_numpy()
    return float(GRID[int(np.argmin([logloss(shrink(m, f, a), y) for a in GRID]))])


def alphas_for(train: pd.DataFrame, pooled: float) -> dict[str, float]:
    """Per-market alphas, each pulled toward the pooled one by its own sample.

    An alpha fitted on 60 rows is the same overconfidence the alpha is meant to
    cure, one level up, so the coefficient is itself shrunk.
    """
    out: dict[str, float] = {}
    for market, rows in train.groupby("market"):
        if len(rows) < MIN_TRAIN:
            out[str(market)] = pooled
            continue
        w = len(rows) / (len(rows) + PRIOR_ROWS)
        out[str(market)] = round(w * fit_alpha(rows) + (1 - w) * pooled, 3)
    return out


def apply_alphas(df: pd.DataFrame, alphas: dict[str, float], pooled: float) -> np.ndarray:
    a = df["market"].map(alphas).fillna(pooled).to_numpy(dtype=float)
    fair, model = df["fair"].to_numpy(), df["model"].to_numpy()
    return np.clip(fair + a * (model - fair), EPS, 1 - EPS)


def calibration(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of realized on predicted; slope under 1 is optimism."""
    if len(p) < 20 or float(np.std(p)) < 1e-9:
        return float("nan"), float("nan")
    slope, inter = np.polyfit(p, y, 1)
    return float(slope), float(inter)


@dataclass
class MarketRow:
    market: str
    n_train: int
    n_test: int
    alpha_raw: float
    alpha: float
    ll_model: float
    ll_fair: float
    ll_shrunk: float
    br_model: float
    br_fair: float
    br_shrunk: float

    @property
    def best(self) -> str:
        return min(
            (self.ll_model, "model"), (self.ll_fair, "price"), (self.ll_shrunk, "shrunk")
        )[1]


def holdout(df: pd.DataFrame, split: float) -> dict[str, float]:
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * split)]
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    pooled = fit_alpha(train)
    y = train["won"].to_numpy()
    m, f = train["model"].to_numpy(), train["fair"].to_numpy()
    print(
        f"\n=== single split: train {len(train)} rows (< {cut}), "
        f"test {len(test)} (>= {cut}) ===\n"
        f"pooled alpha {pooled:.2f}  ·  train log-loss curve "
        + "  ".join(f"{a:.2f}:{logloss(shrink(m, f, a), y):.4f}" for a in GRID[::4])
    )

    alphas = alphas_for(train, pooled)
    rows: list[MarketRow] = []
    for market, tr in train.groupby("market"):
        te = test[test["market"] == market]
        if len(tr) < MIN_MARKET_ROWS or len(te) < MIN_MARKET_ROWS:
            continue
        y_t = te["won"].to_numpy()
        m_t, f_t = te["model"].to_numpy(), te["fair"].to_numpy()
        s_t = shrink(m_t, f_t, alphas[str(market)])
        rows.append(
            MarketRow(
                str(market),
                len(tr),
                len(te),
                fit_alpha(tr),
                alphas[str(market)],
                logloss(m_t, y_t),
                logloss(f_t, y_t),
                logloss(s_t, y_t),
                brier(m_t, y_t),
                brier(f_t, y_t),
                brier(s_t, y_t),
            )
        )
    hdr = (
        f"{'market':16s}{'ntr':>6}{'nte':>6}{'a_raw':>7}{'alpha':>7}"
        f"{'LL mdl':>9}{'LL px':>9}{'LL shr':>9}{'BR mdl':>9}{'BR px':>9}{'BR shr':>9}  best"
    )
    print(f"\n{hdr}")
    for r in sorted(rows, key=lambda r: -r.n_test):
        print(
            f"{r.market:16s}{r.n_train:6d}{r.n_test:6d}{r.alpha_raw:7.2f}{r.alpha:7.2f}"
            f"{r.ll_model:9.4f}{r.ll_fair:9.4f}{r.ll_shrunk:9.4f}"
            f"{r.br_model:9.4f}{r.br_fair:9.4f}{r.br_shrunk:9.4f}  {r.best}"
        )

    y_t = test["won"].to_numpy()
    s_t = apply_alphas(test, alphas, pooled)
    print(
        f"\npooled holdout (n={len(test)}): log loss model "
        f"{logloss(test['model'].to_numpy(), y_t):.4f} | price "
        f"{logloss(test['fair'].to_numpy(), y_t):.4f} | shrunk {logloss(s_t, y_t):.4f}"
    )
    for name, p in (
        ("model", test["model"].to_numpy()),
        ("price", test["fair"].to_numpy()),
        ("shrunk", s_t),
    ):
        sl, ic = calibration(p, y_t)
        print(f"  calibration {name:7s} slope {sl:+.3f}  intercept {ic:+.3f}")
    _money(test.assign(shrunk=s_t))
    return alphas


def write_anchors(alphas: dict[str, float], ledger: Path) -> None:
    """Persist ``1 - alpha`` as the anchor weight the card will read.

    Weights, not probabilities, so the file survives a re-fit and says what it
    means to the code that consumes it. Stamped with the ledger it came from,
    because an anchor fitted on somebody else's history is worse than none.
    """
    path = Config().market_anchor_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ledger": str(ledger),
                "anchors": {m: round(1.0 - a, 3) for m, a in sorted(alphas.items())},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {len(alphas)} anchor weights to {path} (delete it to revert)")


def _money(test: pd.DataFrame) -> None:
    """What the shrink would do to the bet list, on the rows it was graded on.

    The shrink only ever pulls a probability toward the price, so as a filter it
    can only remove bets -- which makes this checkable against real P/L with no
    invented prices. It is also the number that decides adoption: a filter that
    keeps single digits out of hundreds is not a filter, it is a new threshold.
    """
    buys = test[test["buy"] & test["odds"].notna() & test["pnl"].notna()]
    if buys.empty:
        print("\nno graded buys in the holdout")
        return
    dec = np.array([american_to_decimal(o) for o in buys["odds"]])
    ev_shrunk = buys["shrunk"].to_numpy() * dec - 1.0
    edge = buys["shrunk"].to_numpy() - buys["fair"].to_numpy()
    print(f"\nholdout buys, one unit flat (n={len(buys)}):")
    for label, keep in (
        ("as placed", np.ones(len(buys), dtype=bool)),
        ("shrunk EV > 0", ev_shrunk > 0),
        ("shrunk EV > .05", ev_shrunk > 0.05),
        ("shrunk edge > .03", edge > 0.03),
    ):
        sub = buys[keep]
        if sub.empty:
            print(f"  {label:18s} n=0")
            continue
        print(
            f"  {label:18s} n={len(sub):5d}  units {sub['pnl'].sum():+8.2f}  "
            f"roi {sub['pnl'].mean() * 100:+6.2f}%  won {sub['won'].mean() * 100:5.1f}%"
        )


def walk_forward(df: pd.DataFrame, warmup: float) -> None:
    """Refit on every prior date, grade the next one. Eleven splits, not one."""
    dates = sorted(df["date"].unique())
    start = max(5, int(len(dates) * warmup))
    out: list[dict[str, float]] = []
    for day in dates[start:]:
        past, today = df[df["date"] < day], df[df["date"] == day]
        if len(today) < 20:
            continue
        pooled = fit_alpha(past)
        s = apply_alphas(today, alphas_for(past, pooled), pooled)
        y = today["won"].to_numpy()
        out.append(
            {
                "n": len(today),
                "alpha": pooled,
                "ll_model": logloss(today["model"].to_numpy(), y),
                "ll_fair": logloss(today["fair"].to_numpy(), y),
                "ll_shrunk": logloss(s, y),
                "br_model": brier(today["model"].to_numpy(), y),
                "br_fair": brier(today["fair"].to_numpy(), y),
                "br_shrunk": brier(s, y),
            }
        )
    if not out:
        print("\nnot enough dated history for a walk-forward")
        return
    w = pd.DataFrame(out)
    n = w["n"].sum()
    print(
        f"\n=== walk-forward: {len(w)} slates, {n} rows, refit each morning ===\n"
        f"pooled alpha {w['alpha'].min():.2f}-{w['alpha'].max():.2f} "
        f"(median {w['alpha'].median():.2f})"
    )
    for kind, name in (("ll", "log loss"), ("br", "Brier")):
        vals = {
            k: float((w[f"{kind}_{k}"] * w["n"]).sum() / n)
            for k in ("model", "fair", "shrunk")
        }
        print(
            f"{name:9s} model {vals['model']:.4f} | price {vals['fair']:.4f} | "
            f"shrunk {vals['shrunk']:.4f} | shrunk vs price "
            f"{vals['shrunk'] - vals['fair']:+.4f}"
        )
    print(
        "slates where the shrunken number beat the price: "
        f"{(w['ll_shrunk'] < w['ll_fair']).mean() * 100:.0f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=Path.home() / ".mlb_engine" / "audit" / "ledger.csv",
    )
    ap.add_argument("--split", type=float, default=0.7, help="train fraction by date")
    ap.add_argument("--warmup", type=float, default=0.4, help="walk-forward warm-up")
    ap.add_argument("--no-walk-forward", action="store_true")
    ap.add_argument(
        "--write-anchors",
        action="store_true",
        help="adopt the fitted weights: write Config.market_anchor_file",
    )
    args = ap.parse_args()

    df = frame(load_ledger(args.ledger))
    if df.empty:
        raise SystemExit(f"no graded two-sided engine rows in {args.ledger}")
    print(
        f"{len(df)} graded rows with a two-sided price, "
        f"{df['date'].nunique()} slates, {df['market'].nunique()} markets"
    )
    alphas = holdout(df, args.split)
    if not args.no_walk_forward:
        walk_forward(df, args.warmup)
    if args.write_anchors:
        write_anchors(alphas, args.ledger)


if __name__ == "__main__":
    main()
