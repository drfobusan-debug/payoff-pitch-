"""Reliability study: how fast each team rate and goalie GSAx repeats (§5.1, §5.2).

For every metric in ``features.strength.METRICS`` and for goalie GSAx/60 and
raw Sv%, over MoneyPuck regular seasons ``--from``..``--to``:

* **split-half** -- each team-season's games are dealt alternately into two
  halves; the two half-season rates are correlated across all team-seasons.
  ``k = n_half * (1 - r) / r`` in the metric's exposure units (seconds), also
  shown in games so it reads as "the prior weighs as much as N games".
* **ramp check** -- the rate over the first N games vs the rest of the season,
  N in {10, 20, 41}, measured *and* predicted from ``k``. If the two disagree
  the n/(n+k) model is wrong for that metric and the docstring should say so.
* **block** -- first half of the season vs second half (chronological). Lower
  than split-half means the quantity drifts within a season (roster turnover,
  systems changes), which bounds how much a season-long prior is worth.
* **year-to-year** -- full-season rate in season t vs t+1 for the same
  franchise; this is the regression weight the preseason prior uses.

Prints a table and writes JSON (``--out``) that ``config.ShrinkParams`` quotes.
Nothing here is applied automatically: the numbers are copied into ``config.py``
by hand with the study date, so a re-run that moves them is a visible diff.

Usage::

    .venv/bin/python scripts/nhl/reliability_study.py --from 2015 --to 2024 \
        --out ~/.nhl_engine/studies/reliability.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date as Date
from pathlib import Path

import pandas as pd

from engine_common.shrink import k_from_split_half, pearson, reliability_at
from nhl_engine.config import cache_dir
from nhl_engine.data.moneypuck import FRANCHISE_PREDECESSOR, MoneyPuckClient
from nhl_engine.data.teamnames import CODES
from nhl_engine.features.goalie import GOALIE_METRICS, goalie_game_table
from nhl_engine.features.strength import METRICS, game_table

RAMP_N = (10, 20, 41)


def _rate(num: float, exp: float, scale: float) -> float:
    return num / exp * scale if exp > 0 else math.nan


def _team_tables(mp: MoneyPuckClient, seasons: range) -> dict[tuple[str, int], pd.DataFrame]:
    out: dict[tuple[str, int], pd.DataFrame] = {}
    codes = sorted(CODES | set(FRANCHISE_PREDECESSOR.values()))
    for code in codes:
        games = mp.team_games(code)
        if games.empty:
            print(f"  no MoneyPuck file for {code}", file=sys.stderr)
            continue
        for season in seasons:
            g = games[games["season"] == season]
            if g.empty:
                continue
            out[(code, season)] = game_table(g).sort_values("gameDate").reset_index(drop=True)
    return out


def _split_half(
    tables: dict[tuple[str, int], pd.DataFrame], key: str, scale: float, min_games: int
) -> dict[str, float]:
    a_rates, b_rates, half_exp, gpg = [], [], [], []
    for t in tables.values():
        if len(t) < min_games:
            continue
        a, b = t.iloc[::2], t.iloc[1::2]
        ea, eb = float(a[f"{key}__exp"].sum()), float(b[f"{key}__exp"].sum())
        if ea <= 0 or eb <= 0:
            continue
        a_rates.append(_rate(float(a[f"{key}__num"].sum()), ea, scale))
        b_rates.append(_rate(float(b[f"{key}__num"].sum()), eb, scale))
        half_exp.append((ea + eb) / 2)
        gpg.append((ea + eb) / len(t))
    r = pearson(a_rates, b_rates)
    n_half = sum(half_exp) / len(half_exp) if half_exp else math.nan
    exp_per_game = sum(gpg) / len(gpg) if gpg else math.nan
    k = k_from_split_half(r, n_half) if not math.isnan(r) else math.nan
    return {
        "r_split": r,
        "n_half": n_half,
        "exp_per_game": exp_per_game,
        "k": k,
        "k_games": k / exp_per_game if exp_per_game and not math.isinf(k) else k,
        "team_seasons": len(a_rates),
    }


def _ramp(
    tables: dict[tuple[str, int], pd.DataFrame], key: str, scale: float, k: float
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for n in RAMP_N:
        first, rest, exp_first = [], [], []
        for t in tables.values():
            if len(t) < n + 20:
                continue
            a, b = t.iloc[:n], t.iloc[n:]
            ea, eb = float(a[f"{key}__exp"].sum()), float(b[f"{key}__exp"].sum())
            if ea <= 0 or eb <= 0:
                continue
            first.append(_rate(float(a[f"{key}__num"].sum()), ea, scale))
            rest.append(_rate(float(b[f"{key}__num"].sum()), eb, scale))
            exp_first.append(ea)
        r = pearson(first, rest)
        n_first = sum(exp_first) / len(exp_first) if exp_first else math.nan
        # Correlation of a noisy first-N rate with the (also noisy) rest-of-season
        # rate is sqrt(rel_first * rel_rest); both from n/(n+k).
        n_rest = n_first * (max(len(t) for t in tables.values()) - n) / n if exp_first else 0.0
        pred = (
            math.sqrt(reliability_at(n_first, k) * reliability_at(n_rest, k))
            if not (math.isnan(k) or math.isinf(k) or math.isnan(n_first))
            else math.nan
        )
        out[str(n)] = {"r_measured": r, "r_predicted": pred, "n": len(first)}
    return out


def _block(
    tables: dict[tuple[str, int], pd.DataFrame], key: str, scale: float, min_games: int
) -> float:
    a_rates, b_rates = [], []
    for t in tables.values():
        if len(t) < min_games:
            continue
        mid = len(t) // 2
        a, b = t.iloc[:mid], t.iloc[mid:]
        ea, eb = float(a[f"{key}__exp"].sum()), float(b[f"{key}__exp"].sum())
        if ea <= 0 or eb <= 0:
            continue
        a_rates.append(_rate(float(a[f"{key}__num"].sum()), ea, scale))
        b_rates.append(_rate(float(b[f"{key}__num"].sum()), eb, scale))
    return pearson(a_rates, b_rates)


def _year_to_year(
    tables: dict[tuple[str, int], pd.DataFrame], key: str, scale: float, min_games: int
) -> dict[str, float]:
    xs, ys = [], []
    for (code, season), t in tables.items():
        nxt = tables.get((code, season + 1))
        if nxt is None and code in FRANCHISE_PREDECESSOR.values():
            heir = next(h for h, p in FRANCHISE_PREDECESSOR.items() if p == code)
            nxt = tables.get((heir, season + 1))
        if nxt is None or len(t) < min_games or len(nxt) < min_games:
            continue
        e0, e1 = float(t[f"{key}__exp"].sum()), float(nxt[f"{key}__exp"].sum())
        if e0 <= 0 or e1 <= 0:
            continue
        xs.append(_rate(float(t[f"{key}__num"].sum()), e0, scale))
        ys.append(_rate(float(nxt[f"{key}__num"].sum()), e1, scale))
    return {"r_yy": pearson(xs, ys), "pairs": len(xs)}


def _goalie_tables(
    mp: MoneyPuckClient, seasons: range, min_minutes: float
) -> dict[tuple[int, int], pd.DataFrame]:
    out: dict[tuple[int, int], pd.DataFrame] = {}
    ids = mp.goalie_ids(seasons)
    print(f"  {len(ids)} goalies with a season summary in {seasons.start}-{seasons.stop - 1}")
    for pid in ids:
        games = mp.goalie_games(pid)
        if games.empty:
            continue
        table = goalie_game_table(games)
        for season in seasons:
            t = table[table["season"] == season]
            if t.empty or float(t["toi"].sum()) < min_minutes * 60:
                continue
            out[(pid, season)] = t.sort_values("gameDate").reset_index(drop=True)
    return out


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   nan"
    if math.isinf(x):
        return "   inf"
    return f"{x:6.2f}" if abs(x) < 1000 else f"{x:6.0f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="season_from", type=int, default=2015)
    ap.add_argument("--to", dest="season_to", type=int, default=2024)
    ap.add_argument(
        "--min-games", type=int, default=40, help="team-seasons shorter than this are skipped"
    )
    ap.add_argument("--goalie-min-minutes", type=float, default=600.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=cache_dir())
    args = ap.parse_args(argv)

    seasons = range(args.season_from, args.season_to + 1)
    mp = MoneyPuckClient(cache_dir=args.cache_dir, cache_ttl=30 * 86400)

    print(f"team files: seasons {seasons.start}-{seasons.stop - 1}")
    tables = _team_tables(mp, seasons)
    print(f"  {len(tables)} team-seasons")

    results: dict[str, dict] = {}
    print(
        f"\n{'metric':14s} {'r_split':>7s} {'k_games':>8s} {'k_sec':>9s} {'r_block':>7s} {'r_yy':>6s} "
        + " ".join(f"ramp{n}m/p" for n in RAMP_N)
    )
    for m in METRICS:
        sh = _split_half(tables, m.key, m.scale, args.min_games)
        ramp = _ramp(tables, m.key, m.scale, sh["k"])
        blk = _block(tables, m.key, m.scale, args.min_games)
        yy = _year_to_year(tables, m.key, m.scale, args.min_games)
        results[m.key] = {**sh, "r_block": blk, **yy, "ramp": ramp, "label": m.label}
        ramp_s = " ".join(
            f"{ramp[str(n)]['r_measured']:.2f}/{ramp[str(n)]['r_predicted']:.2f}" for n in RAMP_N
        )
        print(
            f"{m.key:14s} {_fmt(sh['r_split'])} {_fmt(sh['k_games'])}  {_fmt(sh['k'])} "
            f"{_fmt(blk)} {_fmt(yy['r_yy'])} {ramp_s}"
        )

    print(f"\ngoalie files (>= {args.goalie_min_minutes:.0f} min in a season)")
    gtables = _goalie_tables(mp, seasons, args.goalie_min_minutes)
    print(f"  {len(gtables)} goalie-seasons")
    for gm in GOALIE_METRICS:
        sh = _split_half(gtables, gm.key, gm.scale, 10)
        blk = _block(gtables, gm.key, gm.scale, 10)
        yy = _year_to_year(gtables, gm.key, gm.scale, 10)
        results[f"goalie_{gm.key}"] = {**sh, "r_block": blk, **yy, "label": gm.label}
        print(
            f"{gm.key:14s} {_fmt(sh['r_split'])} {_fmt(sh['k_games'])}  {_fmt(sh['k'])} "
            f"{_fmt(blk)} {_fmt(yy['r_yy'])}  (n_half={sh['n_half']:.0f} {'sec' if gm.exposure_col == 'toi' else gm.exposure_col})"
        )

    payload = {
        "study": "reliability",
        "run_on": Date.today().isoformat(),
        "seasons": [seasons.start, seasons.stop - 1],
        "min_games": args.min_games,
        "goalie_min_minutes": args.goalie_min_minutes,
        "metrics": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
