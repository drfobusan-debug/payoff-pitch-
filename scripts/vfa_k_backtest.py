"""Does charging the four-seam velocity term make the strikeout props better?

``PitcherRegression.velocity_k_multiplier`` is fitted -- a level against the
league four-seamer plus how his last start sat against his own window -- and
shipped unpriced (``MLBE_VFA_K_WEIGHT``, default 0.0), because a term fitted per
plate appearance has never been graded as a *prop*. This grades it.

The harness in :mod:`mlb_engine.backtest` replays completed slates with every
feature windowed as-of the day before, emits the model's probability at each
market's own standard line and grades it against the box score. No odds, so the
comparison is calibration and accuracy, not ROI -- which is the right question
for a probability term.

Both arms see the same dates, the same Statcast frame and the same lineups; only
the weight differs, so any move in the strikeout groups is the term. The
non-strikeout groups are printed as a control: the term multiplies K per PA, so
hits and totals should barely move, and a large move there means the arms are
not comparable rather than that velocity helped.

    python -m scripts.vfa_k_backtest --start 2026-06-02 --end 2026-07-20 --every 3
    python -m scripts.vfa_k_backtest --weights 0,0.5,1 --sims 2000

Findings are in the pull request that turns the weight on, or does not.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from mlb_engine.audit.grade import LOSS, PUSH, WIN
from mlb_engine.backtest import load_season_frame, run_backtest, summarize
from mlb_engine.config import load_config
from mlb_engine.data.mlb_statsapi import MLBStatsClient
from mlb_engine.recommendations import Recommendation

K_MARKETS = ("pitcher_k",)
OUT = Path("/home/ubuntu/.mlb_engine/output")


def dates_between(start: Date, end: Date, every: int) -> list[Date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=every)
    return out


def _brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - w) ** 2 for p, w in pairs) / len(pairs) if pairs else float("nan")


def _log_loss(pairs: list[tuple[float, int]]) -> float:
    """Mean negative log likelihood, the scale the term was fitted on."""
    import math

    if not pairs:
        return float("nan")
    tot = 0.0
    for p, w in pairs:
        q = min(max(p, 1e-6), 1 - 1e-6)
        tot -= math.log(q) if w else math.log(1 - q)
    return tot / len(pairs)


def _pairs(
    graded: list[tuple[Recommendation, str]], markets: tuple[str, ...] | None = None
) -> dict[str, list[tuple[float, int]]]:
    """(model probability, won) by market, dropping pushes."""
    out: dict[str, list[tuple[float, int]]] = {}
    for rec, result in graded:
        if result == PUSH or result not in (WIN, LOSS):
            continue
        if markets is not None and rec.market not in markets:
            continue
        out.setdefault(rec.market, []).append((float(rec.model_prob), 1 if result == WIN else 0))
    return out


def _key(rec: Recommendation) -> tuple:
    """Identify the same pick across arms, so only shared picks are compared."""
    return (rec.game_pk, rec.market, rec.selection, rec.side, rec.line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-02")
    ap.add_argument("--end", default="2026-07-20")
    ap.add_argument("--every", type=int, default=3, help="stride in days")
    ap.add_argument("--weights", default="0,1")
    ap.add_argument("--sims", type=int, default=2000, help="game sims; props are analytic")
    ap.add_argument("--cache", default=None, help="statcast pickle range to slice")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    start, end = Date.fromisoformat(args.start), Date.fromisoformat(args.end)
    dates = dates_between(start, end, args.every)
    weights = [float(w) for w in args.weights.split(",")]

    os.environ["MLBE_MC_SIMS"] = str(args.sims)
    cfg = load_config()
    frame = load_season_frame(cfg.cache_dir, start - timedelta(days=50), end)
    print(f"{len(dates)} dates {dates[0]}..{dates[-1]}, {len(frame):,} statcast rows")

    client = MLBStatsClient()
    runs: dict[float, list[tuple[Recommendation, str]]] = {}
    for w in weights:
        os.environ["MLBE_VFA_K_WEIGHT"] = str(w)
        cfg_w = load_config()
        assert cfg_w.windows.vfa_k_weight == w
        t = time.time()
        runs[w] = run_backtest(cfg_w, frame, dates, stats=client)
        print(f"weight {w}: {len(runs[w])} graded picks in {(time.time() - t) / 60:.1f} min")

    with (OUT / "vfa_k_backtest.pkl").open("wb") as f:
        pickle.dump(runs, f)

    # Only picks every arm produced: a term that moves a probability across the
    # engine's own selection boundary changes which props exist, and grading a
    # different population would read as an accuracy change on its own.
    shared = set.intersection(*({_key(r) for r, _ in g} for g in runs.values()))
    print(f"\nshared graded picks: {len(shared)}")

    print("\nstrikeout props, by weight")
    print(f"{'weight':>7} {'n':>5} {'win%':>6} {'avgP':>6} {'brier':>8} {'logloss':>8}")
    for w in weights:
        g = [(r, res) for r, res in runs[w] if _key(r) in shared]
        pairs = _pairs(g, K_MARKETS).get("pitcher_k", [])
        if not pairs:
            print(f"{w:>7} {'--':>5}")
            continue
        win = sum(x for _, x in pairs) / len(pairs)
        avg = sum(p for p, _ in pairs) / len(pairs)
        print(
            f"{w:>7} {len(pairs):>5} {win * 100:>5.1f}% {avg * 100:>5.1f}% "
            f"{_brier(pairs):>8.5f} {_log_loss(pairs):>8.5f}"
        )

    print("\ncontrol: every other market, brier by weight")
    markets = sorted({r.market for r, _ in runs[weights[0]] if r.market not in K_MARKETS})
    head = "".join(f"{w:>10}" for w in weights)
    print(f"{'market':<14}{'n':>6}{head}")
    for m in markets:
        row, n = "", 0
        for w in weights:
            g = [(r, res) for r, res in runs[w] if _key(r) in shared]
            pairs = _pairs(g).get(m, [])
            n = max(n, len(pairs))
            row += f"{_brier(pairs):>10.5f}" if pairs else f"{'--':>10}"
        print(f"{m:<14}{n:>6}{row}")

    print("\nfull group summary at each weight")
    for w in weights:
        print(f"-- weight {w}")
        for r in summarize(runs[w]):
            print(
                f"   {r.group:<16} n={r.n:<6} win%={r.win_pct * 100:5.1f} "
                f"avgP={r.avg_model_prob * 100:5.1f} brier={r.brier:.5f} pushes={r.pushes}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
