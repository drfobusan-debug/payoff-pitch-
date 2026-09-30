"""How long a look-back does a starter -- and a bullpen -- actually need?

The starter's rate profile shared the six-week form window, and the pen was read
over three weeks, both on the reasonable-sounding premise that an arm's recent
work is the informative work. This grades that forward, the same way
``batter_window_study`` does: read over the W days *before* a cutoff, score the
read against the next 21 days.

Two subjects, three questions:

* ``starters`` -- out-of-time correlation and holdout RMSE per window, over the
  pitchers who worked a first inning (so relievers do not dilute the starter
  population), plus the next 21 days regressed on the six-week and the candidate
  read together. A coefficient near zero on the shorter read means recent form
  carries nothing the longer one does not already have.
* ``bullpens`` -- the same, on each club's late-relief frame built by the
  production helper (``bullpen_relief_frame``), so the read is the one the
  engine actually prices: relief only, from ``--min-inning`` on, starters
  excluded game by game.

The prior in the RMSE columns is the league mean, which is knowable at the
cutoff; anything fitted after the holdout would be reading the future.

Usage::

    python -m scripts.pitcher_window_study starters \\
        --frame ~/.mlb_engine/cache/statcast_2026-03-24_2026-08-20.pkl
    python -m scripts.pitcher_window_study bullpens --frame ... --cutoffs 4
"""

from __future__ import annotations

import argparse
from datetime import date as Date
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_engine.features.rolling import (
    LEAGUE_RATES,
    _bucket_counts,
    _pa_rows,
    bullpen_relief_frame,
)

WINDOWS = (14, 21, 28, 42, 60, 90, 180)
PEN_WINDOWS = (21, 28, 42, 60, 90, 180)
OUTCOMES = ("K", "BB", "1B", "HR", "OUT")

Reads = dict[str, dict[str, float]]


def _load(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _cutoffs(dates: pd.Series, holdout: int, n: int) -> list[Date]:
    last, first = dates.max(), dates.min()
    out = [last - timedelta(days=holdout * k) for k in range(1, n + 1)]
    return [c for c in out if (c - first).days >= holdout][::-1]


def _rates(events: pd.Series, floor: int) -> dict[str, float] | None:
    counts = _bucket_counts(events)
    n = sum(counts.values())
    if n < floor:
        return None
    return {k: counts[k] / n for k in counts} | {"pa": float(n)}


def _starter_reads(pa: pd.DataFrame, starters: set[int], lo: Date, hi: Date,
                   floor: int) -> Reads:
    sl = pa[(pa["game_date"] >= lo) & (pa["game_date"] < hi)]
    out: Reads = {}
    for pid, g in sl.groupby("pitcher"):
        if int(pid) not in starters:
            continue
        r = _rates(g["events"], floor)
        if r:
            out[str(int(pid))] = r
    return out


def _pen_reads(df: pd.DataFrame, teams: list[str], as_of: Date, days: int,
               min_inning: int, floor: int) -> Reads:
    out: Reads = {}
    for team in teams:
        relief = bullpen_relief_frame(df, team, as_of, days, min_inning)
        if relief.empty:
            continue
        r = _rates(_pa_rows(relief)["events"], floor)
        if r:
            out[team] = r
    return out


def _correlations(reads: dict[tuple[Date, int], Reads], hold: dict[Date, Reads],
                  windows: tuple[int, ...]) -> None:
    print(f"{'window':>7}{'mean BF':>9}" + "".join(f"{k:>8}" for k in OUTCOMES) + f"{'n':>7}")
    for w in windows:
        xs: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k in OUTCOMES}
        bf: list[float] = []
        for cut, held in hold.items():
            r = reads[(cut, w)]
            for key, h in held.items():
                if key not in r:
                    continue
                bf.append(r[key]["pa"])
                for k in OUTCOMES:
                    xs[k][0].append(r[key][k])
                    xs[k][1].append(h[k])
        line = f"{w:5d}d {np.mean(bf):8.0f}"
        for k in OUTCOMES:
            line += f"{np.corrcoef(*xs[k])[0, 1]:8.3f}"
        print(line + f"{len(xs['K'][0]):7d}")


def _rmse(reads: dict[tuple[Date, int], Reads], hold: dict[Date, Reads],
          windows: tuple[int, ...], prior: float) -> None:
    print(f"\nholdout RMSE x1000, league prior at {prior:.0f} batters faced")
    print(f"{'window':>7}" + "".join(f"{k:>8}" for k in OUTCOMES))
    for w in windows:
        line = f"{w:5d}d "
        for k in OUTCOMES:
            errs = []
            for cut, held in hold.items():
                r = reads[(cut, w)]
                for key, h in held.items():
                    if key not in r:
                        continue
                    n = r[key]["pa"]
                    est = (r[key][k] * n + prior * LEAGUE_RATES[k]) / (n + prior)
                    errs.append((est - h[k]) ** 2)
            line += f"{np.sqrt(np.mean(errs)) * 1000:8.1f}"
        print(line)


def _joint(reads: dict[tuple[Date, int], Reads], hold: dict[Date, Reads],
           base: int, others: tuple[int, ...]) -> None:
    print(f"\nnext 21 days ~ {base}d read + other window, coefficients b/c")
    print(f"{'vs':>7}" + "".join(f"{k:>16}" for k in OUTCOMES))
    for w in others:
        line = f"{w:5d}d "
        for k in OUTCOMES:
            x, y = [], []
            for cut, held in hold.items():
                a, b = reads[(cut, base)], reads[(cut, w)]
                for key, h in held.items():
                    if key in a and key in b:
                        x.append([a[key][k], b[key][k]])
                        y.append(h[k])
            mat = np.column_stack([np.ones(len(x)), np.array(x)])
            beta, *_ = np.linalg.lstsq(mat, np.array(y), rcond=None)
            line += f"{beta[1]:+8.2f}/{beta[2]:+.2f}"
        print(line)


def starters(df: pd.DataFrame, args: argparse.Namespace) -> None:
    pa = _pa_rows(df)
    ids = set(pa.loc[pa["inning"] <= 1, "pitcher"].dropna().astype(int))
    cuts = _cutoffs(pa["game_date"], args.holdout, args.cutoffs)
    print(f"{len(ids)} pitchers worked a first inning")
    print(f"cutoffs: {', '.join(str(c) for c in cuts)}")

    hold = {
        c: _starter_reads(pa, ids, c, c + timedelta(days=args.holdout), args.min_bf)
        for c in cuts
    }
    reads = {
        (c, w): _starter_reads(pa, ids, c - timedelta(days=w), c, args.read_bf)
        for c in cuts
        for w in WINDOWS
    }
    print(f"\nstarters: out-of-time correlation with the next {args.holdout} days")
    _correlations(reads, hold, WINDOWS)
    _rmse(reads, hold, WINDOWS, args.prior)
    _joint(reads, hold, args.base, tuple(w for w in WINDOWS if w != args.base))


def bullpens(df: pd.DataFrame, args: argparse.Namespace) -> None:
    teams = sorted(str(t) for t in pd.unique(df["home_team"].dropna()))
    cuts = _cutoffs(df["game_date"], args.holdout, args.cutoffs)
    print(f"{len(teams)} clubs, late relief from inning {args.min_inning}")
    print(f"cutoffs: {', '.join(str(c) for c in cuts)}")

    hold = {
        c: _pen_reads(df, teams, c + timedelta(days=args.holdout), args.holdout,
                      args.min_inning, args.min_bf)
        for c in cuts
    }
    reads = {
        (c, w): _pen_reads(df, teams, c, w, args.min_inning, args.read_bf)
        for c in cuts
        for w in PEN_WINDOWS
    }
    print(f"\nbullpens: out-of-time correlation with the next {args.holdout} days")
    _correlations(reads, hold, PEN_WINDOWS)
    _rmse(reads, hold, PEN_WINDOWS, args.prior * 2)
    _joint(reads, hold, 21, tuple(w for w in PEN_WINDOWS if w != 21))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["starters", "bullpens"])
    p.add_argument("--frame", required=True, help="cached Statcast pickle")
    p.add_argument("--holdout", type=int, default=21, help="days scored after the cutoff")
    p.add_argument("--cutoffs", type=int, default=4, help="how many disjoint holdouts")
    p.add_argument("--min-bf", type=int, default=40, help="holdout batters-faced floor")
    p.add_argument("--read-bf", type=int, default=20, help="window batters-faced floor")
    p.add_argument("--prior", type=float, default=100.0, help="league prior strength")
    p.add_argument("--base", type=int, default=42, help="window the joint fit compares against")
    p.add_argument("--min-inning", type=int, default=6, help="first relief inning counted")
    args = p.parse_args()

    df = _load(args.frame)
    print(f"{len(df):,} rows, {df['game_date'].min()} .. {df['game_date'].max()}")
    if args.mode == "starters":
        starters(df, args)
    else:
        bullpens(df, args)


if __name__ == "__main__":
    main()
