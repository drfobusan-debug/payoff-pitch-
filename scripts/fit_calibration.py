"""Fit calibration maps from backtest picks; validate out-of-sample (temporal)."""

from __future__ import annotations

import pickle
from collections import defaultdict
from datetime import date
from pathlib import Path

from mlb_engine.audit.grade import PUSH, WIN
from mlb_engine.calibration import Calibrator

SPLIT = date(2024, 8, 1)  # train < SPLIT, test >= SPLIT


def _brier(pairs: list[tuple[float, int]]) -> float:
    return sum((p - w) ** 2 for p, w in pairs) / len(pairs) if pairs else 0.0


def _gap(pairs: list[tuple[float, int]]) -> float:
    fav = [(p, w) for p, w in pairs if p >= 0.5]
    if not fav:
        return 0.0
    return sum(p for p, _ in fav) / len(fav) - sum(w for _, w in fav) / len(fav)


def _ppv_npv(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    tp = fp = fn = tn = 0
    for p, w in pairs:
        if p >= 0.5 and w:
            tp += 1
        elif p >= 0.5:
            fp += 1
        elif w:
            fn += 1
        else:
            tn += 1
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return ppv, npv


def main() -> int:
    graded = pickle.load(open("/tmp/backtest_graded.pkl", "rb"))
    train_rows: list[tuple[str, float, int]] = []
    test_by_market: dict[str, list[tuple[float, int]]] = defaultdict(list)
    all_rows: list[tuple[str, float, int]] = []
    for rec, result in graded:
        if result == PUSH:
            continue
        won = 1 if result == WIN else 0
        all_rows.append((rec.market, rec.model_prob, won))
        if rec.game_date < SPLIT:
            train_rows.append((rec.market, rec.model_prob, won))
        else:
            test_by_market[rec.market].append((rec.model_prob, won))

    cal = Calibrator.fit(train_rows)

    print(f"Out-of-sample validation (train <{SPLIT} | test >= {SPLIT})")
    print(f"{'market':<14}{'n':>7} {'Brier raw>cal':>16} {'|gap| raw>cal':>16} {'PPV raw>cal':>16}")
    agg_raw: list[tuple[float, int]] = []
    agg_cal: list[tuple[float, int]] = []
    for market in sorted(test_by_market):
        pairs = test_by_market[market]
        cpairs = [(cal.apply(market, p), w) for p, w in pairs]
        agg_raw.extend(pairs)
        agg_cal.extend(cpairs)
        br, bc = _brier(pairs), _brier(cpairs)
        gr, gc = abs(_gap(pairs)), abs(_gap(cpairs))
        pr, _ = _ppv_npv(pairs)
        pc, _ = _ppv_npv(cpairs)
        print(
            f"{market:<14}{len(pairs):>7} {br:>7.4f}>{bc:<7.4f} "
            f"{gr * 100:>6.1f}>{gc * 100:<6.1f}pts {pr:>6.3f}>{pc:<6.3f}"
        )
    print(
        f"{'ALL':<14}{len(agg_raw):>7} {_brier(agg_raw):>7.4f}>{_brier(agg_cal):<7.4f} "
        f"{abs(_gap(agg_raw)) * 100:>6.1f}>{abs(_gap(agg_cal)) * 100:<6.1f}pts "
        f"{_ppv_npv(agg_raw)[0]:>6.3f}>{_ppv_npv(agg_cal)[0]:<6.3f}"
    )

    # Ship the production map fit on ALL 2024 data.
    prod = Calibrator.fit(all_rows)
    out = Path("mlb_engine/data/calibration_2024.json")
    prod.to_json(out)
    print(f"\nProduction calibrator (fit on all {len(all_rows)} rows) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
