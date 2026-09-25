"""Extract engine-state closing ML/RL prices (``mlb/closing/closing_<date>.json``) into a flat CSV.

Reads the snapshots straight from a git ref (default ``origin/engine-state``) so nothing from
that branch is checked out or committed. One row per (date, matchup); matchups that appear
with two games on one day (doubleheaders) are dropped because the snapshot does not key them
by first pitch.

    .venv/bin/python scripts/oos_closing_extract.py --out ~/.mlb_engine/audit/oos_closing_2026_late.csv
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

_MATCHUP = re.compile(r"^([A-Z]+) @ ([A-Z]+)$")
_RL = re.compile(r"^([A-Z]+) ([+-]\d+(?:\.\d)?)$")


def _git(args: list[str], repo: Path) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def list_snapshots(repo: Path, ref: str) -> list[str]:
    names = _git(["ls-tree", "-r", "--name-only", ref], repo).split()
    return sorted(n for n in names if n.startswith("mlb/closing/closing_") and n.endswith(".json"))


def parse_snapshot(rows: list[dict], date: str) -> list[dict]:
    """``game_ml`` gives both sides' no-vig probabilities; ``game_rl`` the -1.5/+1.5 prices."""
    by_game: dict[str, dict] = {}
    for r in rows:
        m = _MATCHUP.match(str(r.get("matchup", "")))
        if not m:
            continue
        away, home = m.groups()
        g = by_game.setdefault(r["matchup"], {"date": date, "home": home, "away": away, "n_ml": 0})
        sel = str(r.get("selection", ""))
        if r.get("market") == "game_ml":
            side = "home" if sel.startswith(home + " ") else "away" if sel.startswith(away + " ") else None
            if side:
                g[f"p_{side}_close"] = float(r["no_vig_prob"])
                g[f"{side}_ml_close"] = float(r["american"])
                g["n_ml"] += 1
        elif r.get("market") == "game_rl":
            mm = _RL.match(sel)
            if mm and abs(float(mm.group(2))) == 1.5:
                side = "home" if mm.group(1) == home else "away"
                g.setdefault("rl", {})[(side, float(mm.group(2)))] = float(r["american"])
    out = []
    for g in by_game.values():
        if g["n_ml"] != 2:
            continue
        rl = g.pop("rl", {})
        fav, dog = ("home", "away") if g["p_home_close"] >= 0.5 else ("away", "home")
        for side, line in ((fav, -1.5), (dog, 1.5)):
            if (side, line) in rl:
                g[f"{side}_rl_line_close"] = line
                g[f"{side}_rl_price_close"] = rl[(side, line)]
        out.append(g)
    return out


def extract(repo: Path, ref: str) -> pd.DataFrame:
    out: list[dict] = []
    for name in list_snapshots(repo, ref):
        date = name.rsplit("_", 1)[1][:-5]
        rows = json.loads(_git(["show", f"{ref}:{name}"], repo))
        out.extend(parse_snapshot(rows, date))
    df = pd.DataFrame(out).drop(columns=["n_ml"])
    dup = df.duplicated(subset=["date", "home", "away"], keep=False)
    return df[~dup].sort_values(["date", "home"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--ref", default="origin/engine-state")
    ap.add_argument("--out", type=Path, default=Path.home() / ".mlb_engine" / "audit" / "oos_closing_2026_late.csv")
    args = ap.parse_args()
    df = extract(args.repo, args.ref)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} closing games {df['date'].min()}..{df['date'].max()} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
