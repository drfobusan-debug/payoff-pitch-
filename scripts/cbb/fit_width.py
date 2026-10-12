"""Fit v0 width (§3.6) from ``final − close`` residuals: SD per pre-close bucket, key-number
excess mass, parity shift. Input CSV columns: close_total, final_total, paired_books, spread_abs.

    .venv/bin/python -m scripts.cbb.fit_width closes_2024_25.csv --out cbb_engine/params/width.json

The input is the §4.e historical buy (or the engine's own close shards after a season);
this script is the only thing allowed to write ``width.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import pstdev

from cbb_engine.models.distribution import PARAMS, _normal_pmf, quality_bucket

KEY_Z = 3.0


def fit(rows: list[dict[str, str]], *, min_n: int = 200) -> dict[str, object]:
    resid: dict[str, list[float]] = defaultdict(list)
    all_resid: list[float] = []
    finals: Counter[int] = Counter()
    for r in rows:
        try:
            c, f = float(r["close_total"]), float(r["final_total"])
        except (KeyError, ValueError):
            continue
        books = int(float(r.get("paired_books", "0") or 0))
        spread = float(r["spread_abs"]) if r.get("spread_abs") else None
        resid[quality_bucket(books, spread)].append(f - c)
        all_resid.append(f - c)
        finals[int(f)] += 1
    if len(all_resid) < min_n:
        raise SystemExit(f"only {len(all_resid)} rows; need {min_n} before a width is written")
    sd_all = pstdev(all_resid)
    sd_by = {"all": round(sd_all, 3)}
    for b, xs in resid.items():
        if len(xs) >= min_n:
            sd_by[b] = round(pstdev(xs), 3)
    # key-number excess: empirical mass at each total minus the normal around each game's close
    n = len(all_resid)
    expected: dict[int, float] = defaultdict(float)
    for r in rows:
        try:
            c = float(r["close_total"])
        except (KeyError, ValueError):
            continue
        for k, v in _normal_pmf(c, sd_all, int(c - 4 * sd_all), int(c + 4 * sd_all)).items():
            expected[k] += v
    # z-test per total against the binomial SE of its expected share; z > 3 because ~150 totals are tested
    key_mass = {}
    for k, cnt in finals.items():
        p_exp = expected[k] / n
        excess = cnt / n - p_exp
        se = math.sqrt(max(p_exp * (1 - p_exp), 1e-12) / n)
        if excess > 0 and excess / se > KEY_Z:
            key_mass[str(k)] = round(excess, 5)
    odd = sum(cnt for k, cnt in finals.items() if k % 2) / n
    return {"sd_by_bucket": sd_by, "key_mass": key_mass, "parity_shift": round(odd - 0.5, 5), "n": n, "fitted_on": date.today().isoformat()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default=str(PARAMS))
    ap.add_argument("--min-n", type=int, default=200)
    args = ap.parse_args(argv)
    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = fit(rows, min_n=args.min_n)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
