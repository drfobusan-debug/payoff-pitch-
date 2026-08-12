"""Is the engine's starter projection stale, and does it cost money?

The engine builds a starter from a six-week Statcast window. If his last three
weeks are worse than the six before them, that projection is optimistic by
construction — and a bet that *backs* him inherits the error. This script grades
that hypothesis against the ledger rather than against xwOBA, because it is a
claim about our own inputs, not about the pitcher.

For every graded buy it recovers the relevant starter, computes his trend triple
as of that morning from pitches thrown *before* the game (6-week SIERA and
Stuff levels, 3-week vFA; deltas are the last three weeks against everything
before them), decides whether the bet backs the arm or opposes it, and reports
ROI by cell with a Welch t-test and a chronological split.

    python -m scripts.starter_staleness_study
    python -m scripts.starter_staleness_study --pkl statcast_2026-07-01_2026-08-11.pkl

Findings as of 2026-08-12 (177 bets, 07-19 to 08-10) are thin and only one of
the three arrows repeats across the split — see ``INTERPRETATION`` below. This
exists to be re-run at the calibration refit with roughly double the sample.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import date as Date

import numpy as np
import pandas as pd

from mlb_engine.config import load_config
from mlb_engine.features.regression import build_pitcher_regression
from mlb_engine.features.siera import pitcher_siera

FB = ("FF", "SI", "FC")
BUY_TIERS = ("Strong buy", "Moderate buy")
MARKETS = (
    "pitcher_h",
    "pitcher_k",
    "pitcher_outs",
    "game_ml",
    "f5_ml",
    "game_total",
    "f5_total",
)
SIX_WEEKS = 42
THREE_WEEKS = 21
MIN_PRIOR_PITCHES = 200
MIN_WINDOW_PITCHES = 80
PROP_SUFFIX = r"\s+(?:Hits|Strikeouts|Outs|Walks|ER|Earned Runs)\b"

INTERPRETATION = """
Read the cells, not the pooled number. On the 2026-08-12 cut only Stuff repeats
across the chronological split (-31.2% then -15.2% when we back a declining
arm); SIERA and vFA each reverse sign in one half. What does repeat on all
three metrics is the mirror: fading an *improving* arm returned +16% to +28%.
Directionally consistent, individually thin. Not a gate until the refit.
"""


def _vfa(df: pd.DataFrame) -> float:
    fb = df[df["pitch_type"].isin(FB)]
    return float(fb["release_speed"].mean()) if len(fb) else float("nan")


def _stuff(df: pd.DataFrame) -> float:
    if not len(df):
        return float("nan")
    return float(build_pitcher_regression(df).expected_k_pct())


def _siera(df: pd.DataFrame) -> float:
    return float(pitcher_siera(df).siera) if len(df) else float("nan")


def pitcher_name(selection: str) -> str:
    """'Zack Wheeler Strikeouts o6.5' -> 'Zack Wheeler'."""
    return re.split(PROP_SUFFIX, selection)[0].strip()


def backs_the_arm(market: str, selection: str) -> bool | None:
    """Does this bet win when the starter pitches well?

    ``None`` for moneylines, where the side backed decides it and the caller
    already knows which starter it attached.
    """
    sel = selection.lower()
    last = sel.split()[-1] if sel.split() else ""
    if market == "pitcher_h":
        return last.startswith("u") or " under" in sel
    if market in ("pitcher_k", "pitcher_outs"):
        return last.startswith("o") or " over" in sel
    if market.endswith("_total"):
        return sel.startswith("under") or last.startswith("u")
    return None


def name_to_id(audit_dir) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in glob.glob(f"{audit_dir}/predictions_*.json"):
        for r in json.load(open(path)):
            if r["market"].startswith("pitcher_") and r.get("player_id"):
                out[pitcher_name(r["selection"])] = int(r["player_id"])
    return out


def starters_by_game(audit_dir) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for path in glob.glob(f"{audit_dir}/previews_*.json"):
        for g in json.load(open(path)):
            out[(g["game_date"], g["matchup"])] = {
                g["home"]: g["home_starter"]["name"],
                g["away"]: g["away_starter"]["name"],
            }
    return out


def trend_table(df: pd.DataFrame, ids: set[int]) -> dict[tuple[int, Date], dict[str, float]]:
    """(pitcher, date) -> trend triple built only from pitches before that date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["game_date"]).dt.date
    dates = sorted(df["date"].unique())
    out: dict[tuple[int, Date], dict[str, float]] = {}
    for pid, sl in df.groupby("pitcher"):
        if int(pid) not in ids:
            continue
        for d in dates:
            prior = sl[sl["date"] < d]
            if len(prior) < MIN_PRIOR_PITCHES:
                continue
            six = prior[prior["date"] > d - pd.Timedelta(days=SIX_WEEKS).to_pytimedelta()]
            three = prior[prior["date"] > d - pd.Timedelta(days=THREE_WEEKS).to_pytimedelta()]
            older = prior[prior["date"] <= d - pd.Timedelta(days=THREE_WEEKS).to_pytimedelta()]
            if len(three) < MIN_WINDOW_PITCHES or len(older) < MIN_WINDOW_PITCHES:
                continue
            out[(int(pid), d)] = {
                "siera": _siera(six),
                "stuff": _stuff(six),
                "vfa": _vfa(three),
                "d_siera": _siera(three) - _siera(older),
                "d_stuff": _stuff(three) - _stuff(older),
                "d_vfa": _vfa(three) - _vfa(older),
            }
    return out


def attach_trends(led: pd.DataFrame, nmap, games, trends) -> pd.DataFrame:
    rows = []
    for _, r in led.iterrows():
        pids: list[int] = []
        if r["market"].startswith("pitcher_"):
            nm = pitcher_name(r["selection"])
            if nm in nmap:
                pids = [nmap[nm]]
        else:
            starters = games.get((str(r["d"]), r["matchup"]), {})
            if r["market"].endswith("_ml"):
                nm = starters.get(r["selection"].split()[0], "")
                if nm in nmap:
                    pids = [nmap[nm]]
            else:  # a total is a bet on both arms
                pids = [nmap[n] for n in starters.values() if n in nmap]
        vals = [trends[(p, r["d"])] for p in pids if (p, r["d"]) in trends]
        if not vals:
            continue
        row = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        row.update(
            market=r["market"],
            date=r["d"],
            won=r["result"] == "win",
            pnl=float(r["pnl"]) if pd.notna(r["pnl"]) else 0.0,
            backs=backs_the_arm(r["market"], r["selection"]),
        )
        rows.append(row)
    cols = ["siera", "stuff", "vfa", "d_siera", "d_stuff", "d_vfa",
            "market", "date", "won", "pnl", "backs"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).dropna(subset=["d_siera", "d_stuff", "d_vfa"])


def _cell(x: pd.DataFrame) -> str:
    if not len(x):
        return "n/a".ljust(30)
    roi = 100 * x["pnl"].sum() / len(x)
    return f"n={len(x):>3}  win {100 * x['won'].mean():>4.1f}%  ROI {roi:>+6.1f}%"


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Difference in mean per-unit return, and its t-statistic."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    if se == 0:
        return diff, float("inf") if diff else 0.0
    return diff, diff / se


WORSE = {"d_siera": 1.0, "d_stuff": -1.0, "d_vfa": -1.0}  # sign meaning "declining"


def report(d: pd.DataFrame) -> None:
    print(f"{len(d)} graded buys matched to a starter trend, "
          f"{d['date'].min()} to {d['date'].max()}\n")
    print(d["market"].value_counts().to_string(), "\n")
    s = d[d["backs"].notna()].copy()
    s["backs"] = s["backs"].astype(bool)
    cut = sorted(d["date"])[len(d) // 2]

    for col, sign in WORSE.items():
        worse = s[s[col] * sign > 0]
        better = s[s[col] * sign <= 0]
        print(f"\n=== {col} (props and totals only; moneylines excluded) ===")
        print(f"  declining arm, backing him   {_cell(worse[worse['backs']])}")
        print(f"  declining arm, against him   {_cell(worse[~worse['backs']])}")
        print(f"  improving arm, backing him   {_cell(better[better['backs']])}")
        print(f"  improving arm, against him   {_cell(better[~better['backs']])}")
        diff, t = welch(worse[worse["backs"]]["pnl"], better[better["backs"]]["pnl"])
        print(f"  backing a declining vs an improving arm: "
              f"{100 * diff:+.1f} units/100 bets, t {t:+.2f}")
        for half, ss in (("train", s[s["date"] <= cut]), ("test", s[s["date"] > cut])):
            w = ss[(ss[col] * sign > 0) & ss["backs"]]
            b = ss[(ss[col] * sign <= 0) & ss["backs"]]
            print(f"    {half:<6} declining+backing {_cell(w)}   improving+backing {_cell(b)}")

    print("\n=== all three arrows agreeing (every market) ===")
    good = d[(d["d_siera"] < 0) & (d["d_stuff"] > 0) & (d["d_vfa"] > 0)]
    bad = d[(d["d_siera"] > 0) & (d["d_stuff"] < 0) & (d["d_vfa"] < 0)]
    print(f"  improving arm  {_cell(good)}")
    print(f"  declining arm  {_cell(bad)}")
    print(f"  everything     {_cell(d)}")
    diff, t = welch(good["pnl"], bad["pnl"])
    print(f"  improving minus declining: {100 * diff:+.1f} units/100 bets, t {t:+.2f}")
    print(INTERPRETATION)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", help="Statcast pickle in the cache dir (default: newest)")
    args = ap.parse_args(argv)

    cfg = load_config()
    pkl = args.pkl
    if pkl is None:
        found = sorted(glob.glob(str(cfg.cache_dir / "statcast_*.pkl")))
        if not found:
            print("no Statcast pickle in the cache; run the slate first")
            return 2
        pkl = found[-1]
    df = pd.read_pickle(cfg.cache_dir / pkl if not str(pkl).startswith("/") else pkl)

    led = pd.read_csv(cfg.audit_dir / "ledger.csv")
    led = led[led["tier"].isin(BUY_TIERS) & led["market"].isin(MARKETS)]
    led = led[led["result"].isin(["win", "loss"])].copy()
    led["d"] = pd.to_datetime(led["date"]).dt.date

    nmap = name_to_id(cfg.audit_dir)
    d = attach_trends(led, nmap, starters_by_game(cfg.audit_dir),
                      trend_table(df, set(nmap.values())))
    if d.empty:
        print("no graded buys could be matched to a starter trend")
        return 1
    report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
