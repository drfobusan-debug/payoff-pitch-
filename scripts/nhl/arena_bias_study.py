"""Arena tracking-bias check (master plan §3): how much rink bias survives in the inputs.

    python scripts/nhl/arena_bias_study.py --from 2022 --to 2024 --out ~/.nhl_engine/studies/arena_bias.json

Per arena (home team) and season, over regular-season unblocked attempts:

* ``dist_resid``  -- mean recorded shot distance in that arena minus the mean
  distance the *same visiting teams* recorded everywhere else. A scorer who
  books everything three feet longer shows up here; away-team shots are used
  so a home team's own style cannot masquerade as rink bias.
* ``adj_resid``   -- the same on MoneyPuck's ``arenaAdjustedShotDistance``:
  what is left after their correction, i.e. the leak into our xG inputs.
* ``xg_resid``    -- away teams' xG per unblocked attempt in this arena minus
  their xG per attempt elsewhere: distance bias translated into the currency
  the engine uses. Caveat: this also carries the home team's shot-quality
  suppression (a good defensive team lowers visitors' xG/attempt for real), so
  it is an upper bound on tracking bias, not a measure of it; ``adj_resid`` is
  the clean tracking read.
* ``count_resid`` -- away teams' unblocked attempts per game here vs elsewhere
  (recording bias in *volume*, not just location).

Each residual comes with an approximate SE and the number of shots. The
decision rule the plan sets: if ``|adj_resid|`` or ``|xg_resid|`` clears two SE
for an arena in consecutive seasons, a rolling arena correction goes on the
shot coordinates before any RAPM fit; otherwise MoneyPuck's adjustment is used
as-is and the study is re-run each season. Hits/giveaways/takeaways are not
studied because they are barred as model inputs regardless.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from nhl_engine.config import cache_dir
from nhl_engine.data.moneypuck import MoneyPuckClient


def arena_table(shots: pd.DataFrame) -> pd.DataFrame:
    df = shots[(shots["period"] <= 3) & (shots["event"].isin(["SHOT", "MISS", "GOAL"]))].copy()
    df["arena"] = df["homeTeamCode"]
    df["shooter"] = df["awayTeamCode"].where(df["isHomeTeam"] == 0, df["homeTeamCode"])
    away = df[df["isHomeTeam"] == 0]
    rows = []
    games_per_pair = away.groupby(["arena", "shooter"])["game_id"].nunique()
    for arena, here in away.groupby("arena"):
        visitors = here["shooter"].unique()
        elsewhere = away[(away["arena"] != arena) & (away["shooter"].isin(visitors))]
        row: dict[str, float | str] = {"arena": str(arena), "shots": float(len(here))}
        for label, col in (
            ("dist", "shotDistance"),
            ("adj", "arenaAdjustedShotDistance"),
            ("xg", "xGoal"),
        ):
            # visitor-weighted: compare each visitor's here-vs-elsewhere mean, weight by shots here
            diffs, weights = [], []
            for team, grp in here.groupby("shooter"):
                other = elsewhere[elsewhere["shooter"] == team][col].dropna()
                mine = grp[col].dropna()
                if len(other) < 50 or len(mine) < 20:
                    continue
                diffs.append(mine.mean() - other.mean())
                weights.append(len(mine))
            if weights:
                w = pd.Series(weights, dtype=float)
                d = pd.Series(diffs, dtype=float)
                mean = float((d * w).sum() / w.sum())
                pooled_sd = float(here[col].std(ddof=1)) if len(here) > 1 else float("nan")
                row[f"{label}_resid"] = mean
                row[f"{label}_se"] = pooled_sd / math.sqrt(w.sum()) if w.sum() else float("nan")
            else:
                row[f"{label}_resid"] = float("nan")
                row[f"{label}_se"] = float("nan")
        # volume: visitors' attempts per game here vs elsewhere
        here_rate = len(here) / max(here["game_id"].nunique(), 1)
        rates = []
        for team in visitors:
            other = elsewhere[elsewhere["shooter"] == team]
            g = other["game_id"].nunique()
            if g >= 10:
                rates.append((len(other) / g, games_per_pair.get((arena, team), 0)))
        if rates:
            tot = sum(g for _, g in rates)
            base = sum(r * g for r, g in rates) / tot if tot else float("nan")
            row["count_resid"] = here_rate - base
        else:
            row["count_resid"] = float("nan")
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values("adj_resid", key=lambda s: -s.abs()).reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=2022)
    ap.add_argument("--to", dest="end", type=int, default=2024)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    mp = MoneyPuckClient(cache_dir=cache_dir())
    report: dict[str, object] = {}
    flagged: dict[str, list[int]] = {}
    signs: dict[str, set[int]] = {}
    for season in range(args.start, args.end + 1):
        shots = mp.shots(season)
        if shots.empty:
            print(f"{season}: no shot log")
            continue
        table = arena_table(shots)
        report[str(season)] = table.to_dict(orient="records")
        print(f"\n{season}  (away-team shots; resid = here - elsewhere)")
        print(
            f"{'arena':6}{'shots':>7}{'dist':>8}{'se':>6}{'adj':>8}{'se':>6}"
            f"{'xg/att':>9}{'se':>7}{'att/g':>7}"
        )
        for _, r in table.iterrows():
            print(
                f"{r['arena']:6}{r['shots']:7.0f}{r['dist_resid']:8.2f}{r['dist_se']:6.2f}"
                f"{r['adj_resid']:8.2f}{r['adj_se']:6.2f}{r['xg_resid']:9.4f}{r['xg_se']:7.4f}"
                f"{r['count_resid']:7.2f}"
            )
            sig_adj = abs(r["adj_resid"]) > 2 * r["adj_se"] if r["adj_se"] == r["adj_se"] else False
            sig_xg = abs(r["xg_resid"]) > 2 * r["xg_se"] if r["xg_se"] == r["xg_se"] else False
            if sig_adj or sig_xg:
                flagged.setdefault(str(r["arena"]), []).append(season)
                signs.setdefault(str(r["arena"]), set()).add(1 if r["adj_resid"] > 0 else -1)
    # persistent = clears 2 SE in 2+ seasons with the same sign; 32 arenas x 2 stats
    # x 3 seasons will flag a handful at 2 SE by chance, so single seasons mean nothing
    persistent = {a: s for a, s in flagged.items() if len(s) >= 2 and len(signs[a]) == 1}
    report["flagged_2se"] = flagged
    report["persistent"] = persistent
    print("\nflagged (>2 SE on adjusted distance or xG/attempt):", flagged)
    print("persistent across seasons:", persistent or "none")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=float))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
