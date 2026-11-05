"""Split ``nhl-engine closetest`` totals into October vs the rest of the season.

    python scripts/nhl/october_totals_split.py ct_2324.csv ct_2425.csv

Answers the preregistered question in ``docs/nhl/october_totals_study.md``: is
the sim's totals signal weaker in October? Per split (pooled and per season):
the closetest slope ``b`` (result - p_close on p_sim - p_close, no intercept) +-
SE, mean(p_sim - p_close) +- SE, Over hit rate, Brier sim vs close, and the goal
level. Ends with z = (b_rest - b_oct) / sqrt(se_oct^2 + se_rest^2) and the
decision the preregistered rule gives.
"""

from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

from nhl_engine import closetest
from nhl_engine.closetest import Fit, Obs


def season(o: Obs) -> str:
    y, m = int(o.game_date[:4]), int(o.game_date[5:7])
    return f"{y}-{(y + 1) % 100:02d}" if m >= 9 else f"{y - 1}-{y % 100:02d}"


def is_october(o: Obs) -> bool:
    return o.game_date[5:7] == "10"


def show(tag: str, rows: list[Obs]) -> tuple[Fit, float, float]:
    f = closetest.fit(rows, "game_total")
    t = [o for o in rows if o.market == "game_total" and o.result != 0.5]
    d = [o.p_sim - o.p_close for o in t]
    gap = st.mean(d) if d else math.nan
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else math.nan
    hit = st.mean(o.result for o in t) if t else math.nan
    g = closetest.goal_level_check(rows)
    print(
        f"{tag:14} n={f.n:4} slope {f.slope:+.2f} se {f.se:.2f} | sim-close {100 * gap:+.2f} pts (se {100 * se:.2f})"
        f" | over hit {100 * hit:.1f} | Brier sim {f.brier_sim:.4f} close {f.brier_close:.4f}"
        f" | goals {g.get('goals', 0):.2f} sim {g.get('sim_total', 0):.2f} line {g.get('line', 0):.2f}"
    )
    return f, gap, se


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", type=Path, nargs="+")
    args = ap.parse_args()
    obs = [o for p in args.csv for o in closetest.read_csv(p)]
    oct_ = [o for o in obs if is_october(o)]
    rest = [o for o in obs if not is_october(o)]
    fo, _, _ = show("October", oct_)
    fr, _, _ = show("Nov-Apr", rest)
    show("all", obs)
    level_gap = []
    for s in sorted({season(o) for o in obs}):
        _, gap, se = show(f"{s} Oct", [o for o in oct_ if season(o) == s])
        show(f"{s} rest", [o for o in rest if season(o) == s])
        level_gap.append(gap < 0 and -gap > 2 * se)
    z = (fr.slope - fo.slope) / math.sqrt(fo.se**2 + fr.se**2)
    print(f"z (rest - oct) = {z:+.2f}")
    if fo.slope <= fo.se and z > 1:
        print(
            "decision: October is different -> totals anchor 0.90 in October, 0.60 the rest of the season"
        )
    elif fo.slope > fo.se:
        print("decision: October slope > 1 SE -> keep 0.60 year-round")
    else:
        print("decision: not answerable at this n -> keep 0.60")
    print(
        f"October sim-close < 0 by > 2 SE in every season: {all(level_gap)} (compare with Nov-Apr before reading it)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
