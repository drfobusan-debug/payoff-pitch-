"""What are starters' real batters-faced and pitch-count hooks, by team?

The sim's caps came from a hand-entered manager table, never fitted. Statcast has
the actual pitch and PA counts, so measure them.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from mlb_engine.data.managers import DEFAULT_BF_CAP, DEFAULT_PITCH_CAP
from mlb_engine.data.statcast import dedupe_pitches

CACHE = "/home/ubuntu/.mlb_engine/cache/statcast_*.pkl"
# The hand-entered per-manager caps this measurement retired, kept here so the
# comparison stays runnable and the retired numbers stay auditable.
# team: (manager, entered bf_cap, entered pitch_cap)
RETIRED = {
    "TB": ("Kevin Cash", 19, 85),
    "TOR": ("John Schneider", 20, 88),
    "LAD": ("Dave Roberts", 20, 88),
    "MIN": ("Rocco Baldelli", DEFAULT_BF_CAP, DEFAULT_PITCH_CAP),
    "STL": ("Oliver Marmol", DEFAULT_BF_CAP, DEFAULT_PITCH_CAP),
    "CIN": ("Terry Francona", 29, 105),
    "SF": ("Bob Melvin", 29, 105),
    "CLE": ("Stephen Vogt", DEFAULT_BF_CAP, DEFAULT_PITCH_CAP),
}


def load() -> pd.DataFrame:
    frames = [pd.read_pickle(f) for f in sorted(glob.glob(CACHE))]
    shared = sorted(set.intersection(*(set(f.columns) for f in frames)))
    d = dedupe_pitches(pd.concat([f[shared] for f in frames], ignore_index=True))
    d["game"] = (
        d.game_date.astype(str) + "|" + d.away_team.astype(str) + "@" + d.home_team.astype(str)
    )
    first = d[d.inning == 1]
    starters = (
        first.groupby(["game", "inning_topbot"])
        .pitcher.agg(lambda s: s.value_counts().index[0])
        .rename("starter")
        .reset_index()
    )
    d = d.merge(starters, on=["game", "inning_topbot"], how="left")
    # ``inning_topbot`` is the batting half, so the pitching team is the other one.
    d["pitch_team"] = np.where(d.inning_topbot == "Top", d.home_team, d.away_team)
    return d[d.starter.notna() & (d.pitcher == d.starter)]


def per_start(sp: pd.DataFrame) -> pd.DataFrame:
    g = sp.groupby(["game", "pitcher", "pitch_team"], as_index=False).agg(
        pitches=("pitcher", "size"),
        bf=("events", lambda s: int(s.notna().sum())),
    )
    return g[g.bf >= 6]  # drop openers/injury exits from the hook estimate


def main() -> None:
    starts = per_start(load())
    print(f"{len(starts)} starts, {starts.pitch_team.nunique()} teams\n")
    q = [0.50, 0.75, 0.90]
    print("LEAGUE                 median    p75    p90    mean")
    for col in ("bf", "pitches"):
        v = starts[col]
        print(f"  {col:<20}{v.median():7.0f}{v.quantile(0.75):7.0f}"
              f"{v.quantile(0.90):7.0f}{v.mean():8.1f}")
    print(f"\n  sim defaults: bf_cap {DEFAULT_BF_CAP}, pitch_cap {DEFAULT_PITCH_CAP}")

    print("\nPROFILED TEAMS -- retired hand-entered cap vs measured")
    print(f"{'team':<6}{'mgr':<18}{'BF cap':>7}{'BF p75':>8}{'BF p90':>8}"
          f"{'P cap':>7}{'P p75':>8}{'P p90':>8}{'starts':>8}")
    league_bf = starts.bf.quantile(q).tolist()
    league_p = starts.pitches.quantile(q).tolist()
    for abbr, (name, bf_cap, pitch_cap) in sorted(RETIRED.items()):
        t = starts[starts.pitch_team == abbr]
        if t.empty:
            continue
        print(f"{abbr:<6}{name:<18}{bf_cap:>7}"
              f"{t.bf.quantile(0.75):>8.0f}{t.bf.quantile(0.90):>8.0f}"
              f"{pitch_cap:>7}{t.pitches.quantile(0.75):>8.0f}"
              f"{t.pitches.quantile(0.90):>8.0f}{len(t):>8}")
    print(f"{'LGE':<6}{'(all teams)':<18}{DEFAULT_BF_CAP:>7}{league_bf[1]:>8.0f}"
          f"{league_bf[2]:>8.0f}{DEFAULT_PITCH_CAP:>7}{league_p[1]:>8.0f}{league_p[2]:>8.0f}"
          f"{len(starts):>8}")

    # Does the ordering the table asserts (quick hook vs long leash) hold at all?
    print("\nEVERY TEAM, ranked by p75 batters faced (the sim's hook quantile)")
    tt = starts.groupby("pitch_team").agg(
        starts=("bf", "size"), bf75=("bf", lambda s: s.quantile(0.75)),
        p75=("pitches", lambda s: s.quantile(0.75)),
    ).sort_values("bf75")
    tt["retired_bf"] = [
        RETIRED[a][1] if a in RETIRED else DEFAULT_BF_CAP for a in tt.index
    ]
    print(tt.to_string(float_format=lambda v: f"{v:6.1f}"))

    prof = [a for a in RETIRED if a in tt.index]
    entered = np.array([RETIRED[a][1] for a in prof], dtype=float)
    actual = np.array([tt.loc[a, "bf75"] for a in prof], dtype=float)
    print(f"\ncorrelation, retired BF cap vs measured p75 BF, {len(prof)} profiled teams:"
          f" r = {float(np.corrcoef(entered, actual)[0, 1]):+.2f}")
    p_entered = np.array([RETIRED[a][2] for a in prof], dtype=float)
    p_actual = np.array([tt.loc[a, "p75"] for a in prof], dtype=float)
    print(f"correlation, retired pitch cap vs measured p75 pitches: "
          f"r = {float(np.corrcoef(p_entered, p_actual)[0, 1]):+.2f}")


if __name__ == "__main__":
    main()
