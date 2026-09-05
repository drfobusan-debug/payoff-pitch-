"""Fit the power screen's matchup rating on one season, score it on another.

The production rating hands each of five indicators a hand-set threshold and a
hand-set point, then collapses the sum at 3 and 1.  ``power_rating_study.py``
graded that rating out of time and found two of the five carrying no sign and
the collapse landing on a dip.  This script asks whether the indicators support
a rating at all once the weights are fitted rather than guessed: it fits total
bases per plate appearance on the standardised indicators over the fit panel,
scores the held-out panel with those weights, and reports the buckets with
hitter-clustered intervals beside the production label on the same rows.

    .venv/bin/python scripts/power_rating_fit.py \
        --fit  ~/.mlb_engine/audit/rating_panel_2025.csv \
        --test ~/.mlb_engine/audit/rating_panel_2026-04-12_2026-08-24.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = ("share", "fit_delta", "opp_xwoba", "xwoba_con", "k_pct")


@dataclass(frozen=True)
class PanelRow:
    player_id: str
    label: str
    score: int
    features: tuple[float, ...]
    pa: float
    tb: float
    h: float
    hr: float


def _num(raw: str) -> float:
    if raw in ("", "nan", "None"):
        return math.nan
    return float(raw)


def _load(path: Path) -> list[PanelRow]:
    rows: list[PanelRow] = []
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            pa = _num(record["PA"])
            if not pa:
                continue
            rows.append(
                PanelRow(
                    player_id=record["player_id"],
                    label=record["rating"],
                    score=int(record["score"]),
                    features=tuple(_num(record[name]) for name in FEATURES),
                    pa=pa,
                    tb=_num(record["TB"]),
                    h=_num(record["H"]),
                    hr=_num(record["HR"]),
                )
            )
    return rows


def _standardise(rows: list[PanelRow]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    sds: list[float] = []
    for index in range(len(FEATURES)):
        values = [r.features[index] for r in rows if not math.isnan(r.features[index])]
        means.append(statistics.fmean(values))
        sds.append(statistics.pstdev(values) or 1.0)
    return tuple(means), tuple(sds)


def _design(
    rows: list[PanelRow], means: tuple[float, ...], sds: tuple[float, ...]
) -> np.ndarray:
    out = np.zeros((len(rows), len(FEATURES) + 1))
    out[:, 0] = 1.0
    for row_index, row in enumerate(rows):
        for index, raw in enumerate(row.features):
            value = 0.0 if math.isnan(raw) else (raw - means[index]) / sds[index]
            out[row_index, index + 1] = value
    return out


def _fit(rows: list[PanelRow]) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    means, sds = _standardise(rows)
    x = _design(rows, means, sds)
    y = np.array([r.tb / r.pa for r in rows])
    w = np.array([r.pa for r in rows])
    xw = x * w[:, None]
    beta, *_ = np.linalg.lstsq(xw.T @ x, xw.T @ y, rcond=None)
    return beta, means, sds


def _rate(rows: list[PanelRow], values: np.ndarray) -> float:
    total_pa = float(sum(r.pa for r in rows))
    if not total_pa:
        return math.nan
    return float(sum(values)) / total_pa


def _tb_rate(rows: list[PanelRow]) -> float:
    return _rate(rows, np.array([r.tb for r in rows]))


def _boot(left: list[PanelRow], right: list[PanelRow], draws: int = 2000) -> tuple[float, float, float]:
    def cluster(rows: list[PanelRow]) -> list[list[PanelRow]]:
        groups: dict[str, list[PanelRow]] = {}
        for row in rows:
            groups.setdefault(row.player_id, []).append(row)
        return list(groups.values())

    left_groups, right_groups = cluster(left), cluster(right)
    observed = _tb_rate(left) - _tb_rate(right)
    draws_out: list[float] = []
    for _ in range(draws):
        sample_left = [r for g in random.choices(left_groups, k=len(left_groups)) for r in g]
        sample_right = [r for g in random.choices(right_groups, k=len(right_groups)) for r in g]
        draws_out.append(_tb_rate(sample_left) - _tb_rate(sample_right))
    draws_out.sort()
    return observed, draws_out[int(0.025 * draws)], draws_out[int(0.975 * draws)]


def _buckets(rows: list[PanelRow], keys: list[float], label: str) -> None:
    order = sorted(range(len(rows)), key=lambda i: keys[i])
    third = len(order) // 3
    bottom = [rows[i] for i in order[:third]]
    middle = [rows[i] for i in order[third : 2 * third]]
    top = [rows[i] for i in order[2 * third :]]
    print(f"--- {label}, thirds of the held-out season ---")
    for name, bucket in (("top", top), ("middle", middle), ("bottom", bottom)):
        print(
            f"  {name:<7} n {len(bucket):5d} PA {sum(b.pa for b in bucket):6.0f}"
            f"  TB/PA {_tb_rate(bucket):.4f}"
        )
    observed, low, high = _boot(top, bottom)
    print(f"  top minus bottom  {observed:+.4f} [{low:+.4f},{high:+.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    fit_rows = _load(args.fit)
    test_rows = _load(args.test)
    print(f"fit {len(fit_rows)} rows, test {len(test_rows)} rows")

    beta, means, sds = _fit(fit_rows)
    print("--- fitted weights, TB/PA per standard deviation ---")
    print(f"  {'intercept':<20} {beta[0]:+.4f}")
    for index, name in enumerate(FEATURES):
        print(f"  {name:<20} {beta[index + 1]:+.4f}")

    design = _design(test_rows, means, sds)
    predicted = list(design @ beta)
    _buckets(test_rows, predicted, "fitted rating")
    _buckets(test_rows, [float(r.score) for r in test_rows], "production score")

    print("--- production label on the same rows ---")
    for label in ("BUY", "HOLD", "AVOID"):
        bucket = [r for r in test_rows if r.label == label]
        print(
            f"  {label:<7} n {len(bucket):5d} PA {sum(b.pa for b in bucket):6.0f}"
            f"  TB/PA {_tb_rate(bucket):.4f}"
        )
    buy = [r for r in test_rows if r.label == "BUY"]
    hold = [r for r in test_rows if r.label == "HOLD"]
    observed, low, high = _boot(buy, hold)
    print(f"  BUY minus HOLD    {observed:+.4f} [{low:+.4f},{high:+.4f}]")


if __name__ == "__main__":
    main()
