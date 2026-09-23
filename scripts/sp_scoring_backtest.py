"""Backtest of the daily worksheet's starter and bullpen scoring (hypotheses H1-H6).

Analysis only: nothing here touches production scoring, pricing, tiers or gates.
Re-runnable in four stages, each cached under ``~/.mlb_engine/audit/sp_study``::

    python scripts/sp_scoring_backtest.py games   --start 2026-04-15 --end 2026-09-22
    python scripts/sp_scoring_backtest.py fg                       # FanGraphs as-of tables
    python scripts/sp_scoring_backtest.py espn                     # ESPN/DraftKings closing lines
    python scripts/sp_scoring_backtest.py build                    # game-level frame
    python scripts/sp_scoring_backtest.py analyze                  # report + CSVs

``games`` pulls every final regular-season game from statsapi with the actual
starters, innings 1-5 runs, the starters' and relievers' runs allowed.  ``fg``
pulls FanGraphs' starter and team-reliever leaderboards **as of the day before
each game date** (season-to-date and the worksheet's skill windows), so every
metric is what the sheet could have known that morning.  ``build`` re-scores the
starters and pens the way ``daily_worksheet.score`` does (+2/+1/-2) and under two
alternatives (decile and z-score points), joins the closing prices from the
``engine-state`` branch (``prices/closing/closing_*.json``, keyed by matchup with
doubleheaders excluded) and writes ``games_scored.csv``.  ``espn`` fills dates
the engine never priced (pre 2026-07-19) with the DraftKings closing moneyline,
run line and full-game total from ESPN's public odds API, written in the same
closing-file format under ``prices/espn``; ESPN has no F5 lines, so F5 market
tests stay on the engine-state window.  ``analyze`` runs the
pre-registered tests and writes ``sp_scoring_backtest.md`` plus one CSV per table.

Every statistical routine lives in :mod:`scripts.sp_backtest_lib`, which is pure
pandas/numpy so it can be unit-tested on synthetic frames without a network.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlb_engine.data import http  # noqa: E402
from mlb_engine.output.daily_worksheet import (  # noqa: E402
    BP_COLS,
    SP_COLS,
    SP_MIN_IP,
    SP_MIN_PITCHES,
    SP_MIN_PITCHES_SEASON,
    SP_SKILL_DAYS,
)
from mlb_engine.output.totals_sheet import _FG_ABBR, _FG_HEADERS, _FG_URL  # noqa: E402
from scripts.sp_backtest_lib import (  # noqa: E402
    BP_KEY,
    ESPN_ABBR,
    build_frame,
    espn_entries,
    load_prices,
    score_bullpens,
    score_starters,
    write_report,
)

log = logging.getLogger("sp_scoring_backtest")
STATS = "https://statsapi.mlb.com/api/v1"
STUDY_DIR = Path.home() / ".mlb_engine" / "audit" / "sp_study"
_FG_TO_MLB = {v: k for k, v in _FG_ABBR.items()}
BP_WINDOWS = sorted({c[5] for c in BP_COLS if c[5] is not None})


# --- statsapi ---------------------------------------------------------------------------


def _get(url: str, tries: int = 4, user_agent: str | None = None) -> dict:
    for i in range(tries):
        try:
            r = http.get(url, timeout=60, **({"user_agent": user_agent} if user_agent else {}))
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if i == tries - 1:
                raise
            log.warning("retry %s: %s", url, exc)
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def _pitching_line(bs_team: dict) -> tuple[list[int], dict[int, dict]]:
    order = [int(p) for p in bs_team.get("pitchers", [])]
    lines = {}
    for pid in order:
        stats = (bs_team["players"].get(f"ID{pid}") or {}).get("stats", {}).get("pitching", {})
        ip = str(stats.get("inningsPitched", "0"))
        whole, _, frac = ip.partition(".")
        lines[pid] = {
            "outs": int(whole or 0) * 3 + int(frac or 0),
            "runs": int(stats.get("runs", 0) or 0),
            "er": int(stats.get("earnedRuns", 0) or 0),
            "pitches": int(stats.get("numberOfPitches", 0) or 0),
        }
    return order, lines


def _game_row(g: dict, day: Date) -> dict | None:
    status = g.get("status") or {}
    if status.get("abstractGameState") != "Final" or status.get("codedGameState") not in ("F", "O"):
        return None  # postponed / suspended games are also abstractGameState=Final
    if g.get("gameType") != "R":
        return None
    if g["teams"]["away"].get("score") is None or g["teams"]["home"].get("score") is None:
        return None
    pk = int(g["gamePk"])
    bs = _get(f"{STATS}/game/{pk}/boxscore")
    ls = _get(f"{STATS}/game/{pk}/linescore")
    inn = ls.get("innings", [])
    if len(inn) < 5:
        return None
    row: dict = {
        "date": day.isoformat(), "game_pk": pk, "game_number": int(g.get("gameNumber", 1)),
        "doubleheader": g.get("doubleHeader", "N") != "N",
        "away": g["teams"]["away"]["team"]["abbreviation"],
        "home": g["teams"]["home"]["team"]["abbreviation"],
        "away_runs": int(g["teams"]["away"].get("score", 0)),
        "home_runs": int(g["teams"]["home"].get("score", 0)),
        "away_f5": sum(int((i.get("away") or {}).get("runs", 0) or 0) for i in inn[:5]),
        "home_f5": sum(int((i.get("home") or {}).get("runs", 0) or 0) for i in inn[:5]),
        "innings": len(inn),
    }
    for side in ("away", "home"):
        order, lines = _pitching_line(bs["teams"][side])
        if not order:
            return None
        sp = order[0]
        pl = (bs["teams"][side]["players"].get(f"ID{sp}") or {}).get("person", {})
        row[f"{side}_sp_id"] = sp
        row[f"{side}_sp_name"] = pl.get("fullName", "")
        row[f"{side}_sp_outs"] = lines[sp]["outs"]
        row[f"{side}_sp_runs"] = lines[sp]["runs"]
        row[f"{side}_sp_pitches"] = lines[sp]["pitches"]
        row[f"{side}_bp_outs"] = sum(v["outs"] for p, v in lines.items() if p != sp)
        row[f"{side}_bp_runs"] = sum(v["runs"] for p, v in lines.items() if p != sp)
        # runs the starter's team allowed in innings 1-5 (charged to the other side)
    row["away_ra_f5"] = row["home_f5"]
    row["home_ra_f5"] = row["away_f5"]
    return row


def fetch_games(start: Date, end: Date, out: Path) -> pd.DataFrame:
    """Every final regular-season game in [start, end]; cached per day."""
    day_dir = out / "games"
    day_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cur = start
    while cur <= end:
        cache = day_dir / f"{cur.isoformat()}.json"
        if cache.exists():
            rows.extend(json.loads(cache.read_text()))
        else:
            sched = _get(f"{STATS}/schedule?sportId=1&date={cur.isoformat()}&gameType=R&hydrate=team")
            day_rows = []
            for d in sched.get("dates", []):
                for g in d.get("games", []):
                    r = _game_row(g, cur)
                    if r is not None:
                        day_rows.append(r)
            cache.write_text(json.dumps(day_rows))
            log.info("%s: %d games", cur, len(day_rows))
            rows.extend(day_rows)
        cur += timedelta(days=1)
    df = pd.DataFrame(rows)
    df.to_csv(out / "games.csv", index=False)
    return df


# --- FanGraphs as-of leaderboards -------------------------------------------------------


def _fg_range(stats: str, season: int, start: Date, end: Date, team: str = "0") -> list[dict]:
    url = _FG_URL.format(stats=stats, season=season, month=1000, team=team)
    url += f"&startdate={start.isoformat()}&enddate={end.isoformat()}"
    for i in range(4):
        try:
            resp = http.get(url, headers=_FG_HEADERS, timeout=90)
            resp.raise_for_status()
            data = resp.json().get("data")
            if isinstance(data, list):
                return data
            raise ValueError("no data")
        except Exception as exc:  # noqa: BLE001
            if i == 3:
                raise
            log.warning("fg retry %s %s..%s: %s", stats, start, end, exc)
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def fetch_fg(dates: list[Date], out: Path) -> None:
    """Starter and team-reliever tables as of the day before each date."""
    fg_dir = out / "fg"
    fg_dir.mkdir(parents=True, exist_ok=True)
    for day in dates:
        asof = day - timedelta(days=1)
        season_start = Date(day.year, 3, 1)
        jobs = {
            "sta_season": ("sta", season_start, "0"),
            f"sta_{SP_SKILL_DAYS}": ("sta", asof - timedelta(days=SP_SKILL_DAYS), "0"),
            "rel_season": ("rel", season_start, "0%2Cts"),
        }
        for w in BP_WINDOWS:
            jobs[f"rel_{w}"] = ("rel", asof - timedelta(days=w), "0%2Cts")
        for key, (stats, start, team) in jobs.items():
            path = fg_dir / f"{day.isoformat()}_{key}.json"
            if path.exists():
                continue
            data = _fg_range(stats, day.year, start, asof, team)
            path.write_text(json.dumps(data))
            time.sleep(0.5)
        log.info("fg %s done", day)


def _sp_value_tables(day: Date, fg_dir: Path) -> pd.DataFrame | None:
    """One row per starter with the 11 SP_COLS values the sheet would have used."""
    p_season = fg_dir / f"{day.isoformat()}_sta_season.json"
    p_skill = fg_dir / f"{day.isoformat()}_sta_{SP_SKILL_DAYS}.json"
    if not (p_season.exists() and p_skill.exists()):
        return None
    season = {int(r["xMLBAMID"]): r for r in json.loads(p_season.read_text()) if r.get("xMLBAMID")}
    skill = {int(r["xMLBAMID"]): r for r in json.loads(p_skill.read_text()) if r.get("xMLBAMID")}
    key = {"xERA": "xERA", "xFIP": "xFIP", "SIERA": "SIERA", "K%": "K%", "K-BB%": "K-BB%",
           "CSW%": "C+SwStr%", "HardHit%": "HardHit%", "Stuff+": "sp_stuff", "FB%": "FB%",
           "O-Swing%": "O-Swing%", "Barrel%": "Barrel%"}
    rows = []
    for pid, r in season.items():
        ip = float(r.get("IP") or 0)
        pitches = float(r.get("Pitches") or 0)
        if ip < SP_MIN_IP or pitches < SP_MIN_PITCHES_SEASON:
            continue
        sk = skill.get(pid)
        sk_ok = sk is not None and float(sk.get("Pitches") or 0) >= SP_MIN_PITCHES
        row: dict = {"date": day.isoformat(), "pitcher": pid, "name": r.get("PlayerName", ""),
                     "team": _FG_TO_MLB.get(str(r.get("TeamName", "")), str(r.get("TeamName", ""))),
                     "IP": ip}
        for label, _nd, _pct, _lower, win in SP_COLS:
            src = sk if (win is not None and sk_ok) else r
            v = src.get(key[label]) if src else None
            if v is None and sk is not None and win is not None:
                v = r.get(key[label])
            row[label] = float(v) if v is not None else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _bp_value_tables(day: Date, fg_dir: Path) -> pd.DataFrame | None:
    paths: dict[int | None, Path] = {None: fg_dir / f"{day.isoformat()}_rel_season.json"}
    for w in BP_WINDOWS:
        paths[w] = fg_dir / f"{day.isoformat()}_rel_{w}.json"
    if not all(p.exists() for p in paths.values()):
        return None
    tables = {w: {_FG_TO_MLB.get(r["TeamName"], r["TeamName"]): r for r in json.loads(p.read_text())}
              for w, p in paths.items()}
    rows = []
    for team in sorted(tables[None]):
        row: dict = {"date": day.isoformat(), "team": team}
        for label, fgkey, _nd, _pct, _lower, win in BP_COLS:
            v = tables[win].get(team, {}).get(fgkey)
            row[label] = float(v) if v is not None else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- espn odds ------------------------------------------------------------------------

ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
ESPN_ODDS = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{eid}/competitions/{eid}/odds"
ESPN_UA = "Mozilla/5.0 (X11; Linux x86_64) sp_scoring_backtest"  # ESPN 403s non-browser agents


def fetch_espn(dates: list[Date], out: Path) -> None:
    """Cache ESPN's DraftKings closing ML / RL / total per date as ``prices/espn/espn_YYYY-MM-DD.json``."""
    d = out / "prices" / "espn"
    d.mkdir(parents=True, exist_ok=True)
    for day in dates:
        p = d / f"espn_{day.isoformat()}.json"
        if p.exists():
            continue
        sb = _get(f"{ESPN_SB}?dates={day.strftime('%Y%m%d')}", user_agent=ESPN_UA)
        entries: list[dict] = []
        for ev in sb.get("events", []):
            if (ev.get("season") or {}).get("type") != 2:
                continue
            comp = ev["competitions"][0]
            sides = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
            away, home = (ESPN_ABBR.get(sides[s], sides[s]) for s in ("away", "home"))
            odds = _get(ESPN_ODDS.format(eid=ev["id"]), user_agent=ESPN_UA)
            items = odds.get("items") or []
            dk = [i for i in items if (i.get("provider") or {}).get("name") == "DraftKings"] or items[:1]
            if dk:
                entries += espn_entries(away, home, dk[0])
            time.sleep(0.2)
        p.write_text(json.dumps(entries))
        log.info("espn %s: %d games, %d entries", day, len(sb.get("events", [])), len(entries))


# --- stages ---------------------------------------------------------------------------


def _study_dates(games: pd.DataFrame) -> list[Date]:
    return sorted(Date.fromisoformat(d) for d in games["date"].unique())


def stage_build(out: Path) -> pd.DataFrame:
    games = pd.read_csv(out / "games.csv")
    fg_dir = out / "fg"
    sp_frames, bp_frames = [], []
    for day in _study_dates(games):
        sp = _sp_value_tables(day, fg_dir)
        bp = _bp_value_tables(day, fg_dir)
        if sp is not None and len(sp):
            sp_frames.append(score_starters(sp))
        if bp is not None and len(bp):
            bp_frames.append(score_bullpens(bp))
    sp_all = pd.concat(sp_frames, ignore_index=True)
    bp_all = pd.concat(bp_frames, ignore_index=True)
    sp_all.to_csv(out / "starter_scores.csv", index=False)
    bp_all.to_csv(out / "bullpen_scores.csv", index=False)
    prices = load_prices(out / "prices" / "closing", out / "prices" / "board", out / "prices" / "espn")
    frame = build_frame(games, sp_all, bp_all, prices)
    frame.to_csv(out / "games_scored.csv", index=False)
    log.info("built %d games (%d priced)", len(frame), int(frame["ml_prob_away"].notna().sum()))
    return frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["games", "fg", "espn", "build", "analyze", "all"])
    ap.add_argument("--start", type=Date.fromisoformat, default=Date(2026, 4, 15))
    ap.add_argument("--end", type=Date.fromisoformat, default=Date(2026, 9, 22))
    ap.add_argument("--out", type=Path, default=STUDY_DIR)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--report", default="sp_scoring_backtest.md", help="report filename under --out")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("games", "all"):
        fetch_games(a.start, a.end, a.out)
    if a.stage in ("fg", "all"):
        games = pd.read_csv(a.out / "games.csv")
        fetch_fg(_study_dates(games), a.out)
    if a.stage in ("espn", "all"):
        games = pd.read_csv(a.out / "games.csv")
        fetch_espn(_study_dates(games), a.out)
    if a.stage in ("build", "all"):
        stage_build(a.out)
    if a.stage in ("analyze", "all"):
        frame = pd.read_csv(a.out / "games_scored.csv")
        sp_all = pd.read_csv(a.out / "starter_scores.csv")
        write_report(frame, sp_all, a.out, boot=a.boot, bp_key=BP_KEY, report_name=a.report)
        log.info("report: %s", a.out / a.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
