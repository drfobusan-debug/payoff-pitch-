"""Goalie studies (§5.2): call-up anchor, time since last start, age curve, team change.

All four run off the same MoneyPuck goalie game logs (all situations) the
engine uses, so the numbers quoted in ``config.GoalieParams`` describe exactly
the input they will be applied to. GSAx/60 = (xG against - GA) per 60.

1. **Call-up anchor** -- pooled GSAx/60 of appearances by *career NHL minutes
   before the game*. The lowest bucket is the empirical prior for a goalie
   under the floor (§5.2 says: not league average, not a guessed percentile).
2. **TSLS** -- pooled GSAx/60 by days since the goalie's previous appearance,
   raw and within-goalie (each game minus that goalie-season's mean, which
   removes "backups get the long gaps" selection). A rust effect is a negative
   within-goalie value at long rests; a fatigue effect is negative at 1 day.
3. **Age curve** -- the delta method: for consecutive goalie-seasons (>= 600 min
   each) the change in GSAx/60 is pooled by age, so it is within-goalie and does
   not confuse "old goalies who are still playing are good" with ageing.
   Birth dates come from the NHL API player landing (cached).
4. **Team change** -- for goalies who switched teams between seasons: the
   residual (new-season GSAx/60 minus prior-season GSAx/60) against the gap in
   the two teams' prior-season 5v5 xGA/60 (new minus old; positive = worse
   defence). A negative slope means xG does *not* fully remove the defence in
   front of him and the carry-over should be shrunk extra.

Every bucket prints its n and a standard error; a bucket with fewer than
``--min-n`` games is shown but marked thin. Nothing here is applied by the
engine until pasted into ``config.py`` with the study line.

Usage::

    .venv/bin/python scripts/nhl/goalie_study.py --from 2015 --to 2024 \
        --out ~/.nhl_engine/studies/goalie.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date as Date
from pathlib import Path

import pandas as pd

from engine_common.shrink import pearson
from nhl_engine.config import cache_dir
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.nhlapi import NHLAPIClient
from nhl_engine.data.teamnames import CODES
from nhl_engine.features.goalie import goalie_game_table
from nhl_engine.features.strength import PER_60, game_table, rate

CAREER_BUCKETS = ((0, 180), (180, 600), (600, 2000), (2000, 5000), (5000, 15000), (15000, 10**9))
REST_BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 7), (8, 14), (15, 30), (31, 10**6))
AGE_BUCKETS = ((18, 23), (24, 26), (27, 29), (30, 32), (33, 35), (36, 45))


def _pooled(t: pd.DataFrame) -> tuple[float, float, int]:
    """Pooled GSAx/60 over appearances, its SE (game-level), and n."""
    exp = float(t["toi"].sum())
    if exp <= 0 or len(t) < 2:
        return math.nan, math.nan, int(len(t))
    r = float(t["gsax60__num"].sum()) / exp * PER_60
    # residual goals per game around the pooled rate, scaled back to per-60
    resid = t["gsax60__num"] - r * t["toi"] / PER_60
    se_goals = float(resid.std(ddof=1)) / math.sqrt(len(t))
    se = se_goals * PER_60 / (exp / len(t))
    return r, se, int(len(t))


def _label(lo: int, hi: int) -> str:
    return f"{lo}+" if hi >= 10**5 else (f"{lo}" if lo == hi else f"{lo}-{hi}")


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:+.3f}"


def _load(mp: MoneyPuckClient, seasons: range) -> pd.DataFrame:
    frames = []
    for pid in mp.goalie_ids(seasons):
        g = mp.goalie_games(pid)
        if g.empty:
            continue
        t = goalie_game_table(g)  # career_toi/days_rest need the full career
        frames.append(t)
    all_games = pd.concat(frames, ignore_index=True)
    return all_games[all_games["season"].isin(list(seasons))].reset_index(drop=True)


def callup(games: pd.DataFrame, min_n: int) -> dict[str, dict[str, float]]:
    print("\n1. GSAx/60 by career NHL minutes before the game")
    out: dict[str, dict[str, float]] = {}
    mins = games["career_toi"] / 60.0
    for lo, hi in CAREER_BUCKETS:
        t = games[(mins >= lo) & (mins < hi)]
        r, se, n = _pooled(t)
        out[_label(lo, hi)] = {"gsax60": r, "se": se, "n": n}
        print(
            f"   {_label(lo, hi):>12s} min  {_fmt(r)} ± {se:.3f}  n={n}{'  thin' if n < min_n else ''}"
        )
    return out


def tsls(games: pd.DataFrame, min_n: int) -> dict[str, dict[str, float]]:
    print("\n2. GSAx/60 by days since previous appearance (raw / within goalie-season)")
    g = games.copy()
    key = ["playerId", "season"]
    season_rate = (
        g.groupby(key)["gsax60__num"].transform("sum")
        / g.groupby(key)["toi"].transform("sum")
        * PER_60
    )
    g["within"] = g["gsax60__num"] - season_rate * g["toi"] / PER_60
    out: dict[str, dict[str, float]] = {}
    for lo, hi in REST_BUCKETS:
        t = g[(g["days_rest"] >= lo) & (g["days_rest"] <= hi)]
        r, se, n = _pooled(t)
        exp = float(t["toi"].sum())
        within = float(t["within"].sum()) / exp * PER_60 if exp > 0 else math.nan
        out[_label(lo, hi)] = {"gsax60": r, "within": within, "se": se, "n": n}
        print(
            f"   {_label(lo, hi):>6s} days  {_fmt(r)} / {_fmt(within)} ± {se:.3f}  n={n}{'  thin' if n < min_n else ''}"
        )
    return out


def _birthdates(ids: list[int], cache: Path) -> dict[int, Date]:
    api = NHLAPIClient(cache_dir=cache, cache_ttl=365 * 86400)
    out: dict[int, Date] = {}
    for pid in ids:
        data = api._get_json(f"https://api-web.nhle.com/v1/player/{pid}/landing")
        if isinstance(data, dict) and isinstance(data.get("birthDate"), str):
            out[pid] = Date.fromisoformat(data["birthDate"])
    return out


def age_curve(
    games: pd.DataFrame, births: dict[int, Date], min_minutes: float, min_n: int
) -> dict[str, dict[str, float]]:
    print(
        "\n3. Age curve: year-over-year change in GSAx/60 by age in the later season (delta method)"
    )
    agg = (
        games.groupby(["playerId", "season"])
        .agg(num=("gsax60__num", "sum"), toi=("toi", "sum"))
        .reset_index()
    )
    agg = agg[agg["toi"] >= min_minutes * 60]
    agg["rate"] = agg["num"] / agg["toi"] * PER_60
    rows = []
    for (pid, season), r in agg.set_index(["playerId", "season"])["rate"].items():
        prev = agg[(agg["playerId"] == pid) & (agg["season"] == season - 1)]
        if prev.empty or pid not in births:
            continue
        age = season - births[pid].year  # age at the season's autumn start
        rows.append({"age": age, "delta": r - float(prev["rate"].iloc[0])})
    d = pd.DataFrame(rows)
    out: dict[str, dict[str, float]] = {}
    for lo, hi in AGE_BUCKETS:
        t = d[(d["age"] >= lo) & (d["age"] <= hi)] if not d.empty else d
        n = int(len(t))
        mean = float(t["delta"].mean()) if n else math.nan
        se = float(t["delta"].std(ddof=1)) / math.sqrt(n) if n > 1 else math.nan
        out[_label(lo, hi)] = {"delta_gsax60": mean, "se": se, "n": n}
        print(
            f"   age {_label(lo, hi):>6s}  {_fmt(mean)} ± {se:.3f}  n={n}{'  thin' if n < min_n else ''}"
        )
    return out


def team_change(
    games: pd.DataFrame, mp: MoneyPuckClient, seasons: range, min_minutes: float
) -> dict[str, float]:
    print(
        "\n4. Team change: GSAx/60 residual vs defensive gap (new team prior-season 5v5 xGA/60 minus old)"
    )
    xga: dict[tuple[str, int], float] = {}
    for code in sorted(CODES | {"ARI"}):
        tg = mp.team_games(code)
        for season in seasons:
            g = tg[tg["season"] == season] if not tg.empty else tg
            if not g.empty:
                xga[(code, season)] = rate(game_table(g), "xga60_5v5")[0]
    agg = (
        games.groupby(["playerId", "season"])
        .agg(num=("gsax60__num", "sum"), toi=("toi", "sum"), team=("playerTeam", "last"))
        .reset_index()
    )
    agg = agg[agg["toi"] >= min_minutes * 60]
    agg["rate"] = agg["num"] / agg["toi"] * PER_60
    gaps, resids = [], []
    for row in agg.itertuples():
        prev = agg[(agg["playerId"] == row.playerId) & (agg["season"] == row.season - 1)]
        if prev.empty or prev["team"].iloc[0] == row.team:
            continue
        old, new = prev["team"].iloc[0], row.team
        k_old, k_new = (old, row.season - 1), (new, row.season - 1)
        if k_old not in xga or k_new not in xga:
            continue
        gaps.append(xga[k_new] - xga[k_old])
        resids.append(row.rate - float(prev["rate"].iloc[0]))
    r = pearson(gaps, resids)
    n = len(gaps)
    slope = math.nan
    if n >= 3:
        mg, mr = sum(gaps) / n, sum(resids) / n
        sxx = sum((g - mg) ** 2 for g in gaps)
        slope = (
            sum((g - mg) * (x - mr) for g, x in zip(gaps, resids, strict=True)) / sxx
            if sxx > 0
            else math.nan
        )
    mean_resid = sum(resids) / n if n else math.nan
    print(
        f"   movers n={n}  mean residual {_fmt(mean_resid)}  corr(gap, residual) {r:+.2f}  slope {slope:+.3f} GSAx/60 per xGA/60 of gap"
    )
    return {"n": n, "mean_residual": mean_resid, "corr": r, "slope": slope}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="season_from", type=int, default=2015)
    ap.add_argument("--to", dest="season_to", type=int, default=2024)
    ap.add_argument(
        "--min-minutes",
        type=float,
        default=600.0,
        help="goalie-season floor for the age and team-change cuts",
    )
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=cache_dir())
    args = ap.parse_args(argv)

    seasons = range(args.season_from, args.season_to + 1)
    mp = MoneyPuckClient(cache_dir=args.cache_dir, cache_ttl=30 * 86400)
    games = _load(mp, seasons)
    print(
        f"{len(games)} goalie appearances, {games['playerId'].nunique()} goalies, seasons {seasons.start}-{seasons.stop - 1}"
    )

    results: dict[str, object] = {
        "callup": callup(games, args.min_n),
        "tsls": tsls(games, args.min_n),
    }
    births = _birthdates(sorted(games["playerId"].unique()), args.cache_dir)
    print(f"   ({len(births)} birth dates from the NHL API)")
    results["age"] = age_curve(games, births, args.min_minutes, 30)
    results["team_change"] = team_change(games, mp, seasons, args.min_minutes)

    payload = {
        "study": "goalie",
        "run_on": Date.today().isoformat(),
        "seasons": [seasons.start, seasons.stop - 1],
        **results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
