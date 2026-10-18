"""Replay one cached slate on two checkouts and diff the card: proof a change is card-neutral.

Runs ``mlb-engine run --date D`` from ``--main-wt`` and from ``--branch-wt``,
each in its own scratch ``MLBE_DATA_DIR`` built from symlinks to the real data
(the audit dir is copied), with ``MLBE_STATE_SYNC=0``, the odds disk cache made
authoritative, a dummy Odds API key (a cache miss fails instead of spending
credits), no ``--email`` and a frozen clock. Then compares the predictions JSON
field by field and the workbook cell by cell, and lists the files only the
branch wrote. Report only; nothing in the real data dir is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
LINK = ("projections", "batx", "fangraphs", "evanalytics", "ros_hitters.csv",
        "calibration_live.json", "market_anchor_live.json")
RUNNER = """
import sys
from datetime import datetime, timezone
import mlb_engine.pipeline as p
orig = p.hours_to_first_pitch
now = datetime.fromisoformat(sys.argv[1] + "T16:00:00+00:00")
p.hours_to_first_pitch = lambda g, now_=None: orig(g, now=now)
from mlb_engine import cli
sys.exit(cli.main(["run", "--date", sys.argv[1]]))
"""


def pick_date(audit: Path) -> str:
    best, best_n = "", -1
    for f in sorted(audit.glob("predictions_????-??-??.json"))[-21:]:
        rows = json.loads(f.read_text())
        n = sum(1 for r in rows if str(r.get("market", "")).startswith(("batter_", "pitcher_"))
                and r.get("market_american") is not None)
        if n > best_n:
            best, best_n = f.stem.split("_")[1], n
    return best


def build_dir(real: Path, dest: Path, day: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "cache" / "oddsapi").mkdir(parents=True)
    (dest / "output").mkdir()
    for name in LINK:
        if (real / name).exists():
            (dest / name).symlink_to(real / name)
    for entry in (real / "cache").iterdir():
        if entry.name != "oddsapi":
            (dest / "cache" / entry.name).symlink_to(entry)
    events = []
    for f in (real / "cache" / "oddsapi").glob("*.json"):
        (dest / "cache" / "oddsapi" / f.name).symlink_to(f)
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        body = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(body, dict) and str(body.get("commence_time", "")).startswith(day):
            events.append({k: body.get(k) for k in
                           ("id", "sport_key", "commence_time", "home_team", "away_team")}
                          | {"bookmakers": []})
    stamp = json.dumps({"url": f"{BASE}/odds", "regions": "us", "oddsFormat": "american",
                        "markets": "h2h,spreads,totals"}, sort_keys=True)
    board = dest / "cache" / "oddsapi" / f"{hashlib.sha256(stamp.encode()).hexdigest()[:20]}.json"
    board.unlink(missing_ok=True)
    board.write_text(json.dumps(events))
    shutil.copytree(real / "audit", dest / "audit")
    for f in (real / "output").glob("vsin_template_*"):
        shutil.copy(f, dest / "output" / f.name)
    print(f"{dest.name}: rebuilt bulk board with {len(events)} events for {day}")


def run(wt: Path, data: Path, day: str) -> int:
    env = {**os.environ, "PYTHONPATH": str(wt), "MLBE_DATA_DIR": str(data),
           "MLBE_STATE_SYNC": "0", "MLBE_ODDS_CACHE_TTL": "99999999",
           "THE_ODDS_API_KEY": "offline-replay-no-key", "ODDS_API_KEY": "offline-replay-no-key"}
    out = subprocess.run([sys.executable, "-c", RUNNER, day], cwd=wt, env=env,
                         capture_output=True, text=True)
    (data / "run.log").write_text(out.stdout + out.stderr)
    print(f"{data.name}: exit {out.returncode} (log {data / 'run.log'})")
    return out.returncode


def _rows(data: Path, day: str) -> dict[tuple, dict]:
    rows = json.loads((data / "audit" / f"predictions_{day}.json").read_text())
    return {(r.get("game_pk"), r.get("market"), r.get("selection"), r.get("line"), r.get("side")): r
            for r in rows}


def _cells(path: Path) -> dict[tuple, object]:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    return {(ws.title, i, j): v for ws in wb.worksheets
            for i, row in enumerate(ws.iter_rows(values_only=True)) for j, v in enumerate(row)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-wt", type=Path, required=True)
    ap.add_argument("--branch-wt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path.home() / ".mlb_engine")
    ap.add_argument("--date", help="cached slate (default: most priced props, last 21 days)")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/mlbe_neutrality"))
    a = ap.parse_args(argv)
    day = a.date or pick_date(a.data / "audit")
    print(f"Replay date: {day}")
    dirs = {name: a.scratch / name for name in ("main", "branch")}
    for name, wt in (("main", a.main_wt), ("branch", a.branch_wt)):
        build_dir(a.data, dirs[name], day)
        if run(wt, dirs[name], day) != 0:
            print(f"{name} run failed; tail of its log:")
            print("\n".join((dirs[name] / "run.log").read_text().splitlines()[-25:]))
            return 1
    m, b = _rows(dirs["main"], day), _rows(dirs["branch"], day)
    fields: dict[str, int] = {}
    for k in m.keys() & b.keys():
        for f in m[k].keys() | b[k].keys():
            if m[k].get(f) != b[k].get(f):
                fields[f] = fields.get(f, 0) + 1
    print(f"predictions: main {len(m)} rows, branch {len(b)} rows, "
          f"only-main {len(m.keys() - b.keys())}, only-branch {len(b.keys() - m.keys())}")
    print(f"  fields that differ on shared rows: {fields or 'none'}")
    xlsx = f"mlb_recommendations_{day}.xlsx"
    cm, cb = _cells(dirs["main"] / "output" / xlsx), _cells(dirs["branch"] / "output" / xlsx)
    diff = [k for k in cm.keys() | cb.keys() if cm.get(k) != cb.get(k)]
    print(f"workbook: {len(diff)} differing cells" + (f", e.g. {sorted(diff)[:5]}" if diff else ""))
    new = sorted(p.name for p in (dirs["branch"] / "audit").iterdir()
                 if not (dirs["main"] / "audit" / p.name).exists())
    print(f"files only the branch wrote: {new}")
    env = {**os.environ, "PYTHONPATH": str(a.branch_wt), "MLBE_DATA_DIR": str(dirs["branch"]),
           "MLBE_STATE_SYNC": "0"}
    sh = subprocess.run([sys.executable, "-m", "mlb_engine.cli", "shadow", "--date", day],
                        cwd=a.branch_wt, env=env, capture_output=True, text=True)
    print((sh.stdout + sh.stderr).strip().splitlines()[-1:] or ["shadow: no output"])
    ok = not fields and not diff and m.keys() == b.keys()
    print("NEUTRAL: card unchanged" if ok else "NOT NEUTRAL: see the diffs above")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
