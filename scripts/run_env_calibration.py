"""Refit the calibration map on run-environment-corrected cards, and ask whether it survives.

``MLBE_RUN_ENV`` ships off because the isotonic map was fit on uncorrected
probabilities: it is monotone, so feeding it a raw that has already been pulled
down maps it back toward the win rate the old raw predicted. The refit is the
step that was owed. This script does it walk-forward and scores four things on the
same holdout rows, per market and line:

* ``production`` -- the calibrated probability the engine actually logged;
* ``corrected``  -- that probability shifted by what the scale is worth
  (the PR's number, no recalibration);
* ``refit``      -- isotonic refit on the *uncorrected* raw, train dates only;
* ``both``       -- isotonic refit on the corrected raw, train dates only.

``both`` vs ``refit`` is the question: the correction is a per-market log-odds
shift, and isotonic is invariant to any monotone transform of its input, so if the
shift is the same for every row in a market a refit map must undo it exactly. Any
gain then belongs to recalibration, not to the run environment -- and the flag is
not what earns it. The heterogeneity section measures how nearly constant that
shift is across game contexts, which bounds what repricing could ever add.

Nothing here writes a probability or the map; ``mlb-engine calibrate`` is what
writes the map.

    python scripts/run_env_calibration.py [--sims N] [--split DATE] [--target RUNS]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_engine.calibration import Calibrator, IsotonicMap
from mlb_engine.config import load_config
from mlb_engine.features.rolling import LEAGUE_RATES
from mlb_engine.models import run_env
from mlb_engine.pipeline import load_calibrator
from scripts.run_env_study import GRADED, elasticity, logit_deltas, score, shift

log = logging.getLogger("run_env_cal")

MIN_TRAIN = 200
MIN_TEST = 100

# Park/lineup contexts the correction has to survive, as multipliers on the league
# rate vector. A scale is one number for the whole slate, but what it is *worth* to
# a probability depends on the game it lands in, and these bracket the card.
CONTEXTS = {
    "league average": 1.00,
    "cold, pitcher's park": 0.88,
    "hot, hitter's park": 1.12,
}


def rows(ledger: Path, markets: tuple[str, ...]) -> pd.DataFrame:
    """Graded rows carrying the raw probability the map was applied to."""
    d = pd.read_csv(ledger, low_memory=False)
    if "source" in d.columns:
        d = d[(d["source"].isna()) | (d["source"] == "engine")]
    d = d[d["market"].isin(markets) & d["result"].isin(["win", "loss"])].copy()
    d["won"] = (d["result"] == "win").astype(float)
    d["over"] = d["selection"].str.contains("Over| o", case=False, regex=True)
    return d.dropna(subset=["raw_prob", "model_prob", "line", "date"])


def heterogeneity(scale: float, sims: int) -> None:
    """How much the correction's log-odds worth varies with the game it lands in."""
    print(f"\nis the shift constant? scale {scale:.4f} priced in three run environments")
    per_key: dict[str, list[float]] = {}
    for name, mult in CONTEXTS.items():
        env = {k: v * mult if k in run_env.NON_OUT else v for k, v in LEAGUE_RATES.items()}
        env["OUT"] = max(0.05, 1.0 - sum(v for k, v in env.items() if k != "OUT"))
        base = run_env.scale_rates(env, 1.0)
        moved = run_env.scale_rates(env, scale)
        # The rate vector is what the simulator reads, so the shift's dependence on
        # context is the dependence of these two vectors' difference on it.
        d = float(np.log(moved["HR"] / base["HR"]) - np.log(moved["OUT"] / base["OUT"]))
        per_key.setdefault("hr log-odds", []).append(d)
        print(f"  {name:<22} d log(HR/OUT) {d:+.4f}")
    spread = max(per_key["hr log-odds"]) - min(per_key["hr log-odds"])
    print(f"  spread across contexts: {spread:.4f} log odds")


def grade(
    sub: pd.DataFrame, delta: float, split: str, market: str, line: str, prod: Calibrator
) -> None:
    train, test = sub[sub["date"] <= split], sub[sub["date"] > split]
    if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
        return
    corr_train = shift(train["raw_prob"], delta, train["over"])
    corr_test = shift(test["raw_prob"], delta, test["over"])
    refit = IsotonicMap.fit(list(zip(train["raw_prob"], train["won"].astype(int), strict=True)))
    both = IsotonicMap.fit(list(zip(corr_train, train["won"].astype(int), strict=True)))

    print(
        f"\n  {market} {line}: train {len(train)}, holdout {len(test)}"
        f" (realized {test['won'].mean():.3f})"
    )
    score("production", test["model_prob"], test["won"])
    score("corrected", shift(test["model_prob"], delta, test["over"]), test["won"])
    # The production path with the flag on: the correction moves the *raw*
    # probability and the map the engine already ships is applied to that. This is
    # the line the decision to flip ``MLBE_RUN_ENV`` actually rests on.
    score(
        "flag on",
        pd.Series([prod.apply(market, p) for p in corr_test], index=test.index),
        test["won"],
    )
    score(
        "refit",
        pd.Series([refit.apply(p) for p in test["raw_prob"]], index=test.index),
        test["won"],
    )
    score("both", pd.Series([both.apply(p) for p in corr_test], index=test.index), test["won"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path.home() / ".mlb_engine/audit/ledger.csv")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--split", default="2026-08-12", help="fit on or before, grade after")
    ap.add_argument("--target", type=float, default=8.954, help="league runs per game")
    args = ap.parse_args()

    scale = run_env.scale_for_total(args.target)
    _, base = elasticity(args.sims, 0.96, 1.04)
    print(f"\ntarget {args.target:.3f} runs -> non-out scale {scale:.4f}")
    deltas = logit_deltas(scale, args.sims, base)
    heterogeneity(scale, args.sims)

    prod = load_calibrator(load_config().calibration_file)
    markets = tuple(GRADED) + ("game_total",)
    d = rows(args.ledger, markets)
    print(f"\ngraded rows with a raw probability: {len(d)} ({d['date'].min()}..{d['date'].max()})")
    for key, delta in sorted(deltas.items()):
        market, line = key.split("|")
        sub = d[(d["market"] == market) & (np.isclose(d["line"], float(line)))]
        grade(sub, delta, args.split, market, line, prod)


if __name__ == "__main__":
    main()
