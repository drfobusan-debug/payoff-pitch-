"""Retrospective, as-of-date worksheet batting study (no engine writes).

Run: .venv/bin/python -m scripts.worksheet_batting_backtest --start 2026-08-04
     --end 2026-09-22 --out ~/.mlb_engine/audit

Inputs: daily Baseball Savant CSVs in OUT/statcast_csv/YYYY-MM-DD.csv, the
MLB Stats API schedule in OUT/schedule_2026.json and the locally fetched
origin/engine-state closing snapshots. --fetch-inputs downloads missing
schedule/Statcast days; by default this script performs no network requests
unless --fangraphs is given. Dated FanGraphs responses are cached in OUT.
Never checks out or pushes state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.special import expit, logit

from mlb_engine.data.statcast import USE_COLS, StatcastRepository, dedupe_pitches
from mlb_engine.output.daily_worksheet import (
    _FG_TO_MLB,
    BAT_COLS,
    BP_COLS,
    K_TEAMS,
    SP_COLS,
    SP_MIN_IP,
    SP_MIN_PITCHES_SEASON,
    SP_SKILL_DAYS,
    Ranking,
    _bat_metrics,
    _fg,
    _prepare,
    _score_offense,
    _sp_raw,
    score,
)

POWER = ("ISO", "Barrel%", "EV90", "HardHit%", "Blast Con%", "SqUp Con%")
DISCIPLINE = ("Z-Contact%", "Whiff%", "O-Swing%")
PRODUCTION = ("wRC", "xwOBA")
SPLITS = ("Overall", "vs hand", "Innings 6+")
SEED = 20260923


@dataclass
class BatExtras:
    pa: dict[str, int]
    windows: dict[int, dict[str, dict[str, float]]]


def _selection_line(selection: str) -> float:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)$", selection)
    return float(m[1]) if m else math.nan


def quote(rows: list[dict[str, object]], market: str, prefix: str) -> dict[str, float] | None:
    """Choose a single same-line pair; missing or ambiguous pairs are not prices."""
    options = [r for r in rows if r["market"] == market and
               str(r["selection"]).startswith(prefix) and
               opposite_quote(rows, market, _selection_line(str(r["selection"]))) is not None]
    if not options:
        return None
    options.sort(key=lambda r: _selection_line(str(r["selection"])))
    r = options[len(options) // 2]
    return {"american": float(str(r["american"])), "prob": float(str(r["no_vig_prob"])),
            "line": _selection_line(str(r["selection"]))}


def opposite_quote(rows: list[dict[str, object]], market: str, line: float) -> dict[str, float] | None:
    prefix = "F5 Under " if market == "f5_total" else "Under "
    for row in rows:
        if row["market"] != market or not str(row["selection"]).startswith(prefix):
            continue
        if _selection_line(str(row["selection"])) == line:
            return {"american": float(str(row["american"])),
                    "prob": float(str(row["no_vig_prob"])),
                    "line": line}
    return None


def payout(american: float, result: float) -> float:
    if math.isnan(american) or math.isnan(result):
        return math.nan
    if result == 0.5:
        return 0.0
    return (american / 100 if american > 0 else 100 / -american) if result else -1.0


def read_prices(repo: Path, day: date) -> dict[str, list[dict[str, object]]]:
    path = f"mlb/closing/closing_{day.isoformat()}.json"
    run = subprocess.run(
        ["git", "-C", str(repo), "show", f"origin/engine-state:{path}"],
        capture_output=True, text=True, check=False,
    )
    if run.returncode:
        return {}
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in json.loads(run.stdout):
        out[str(row["matchup"])].append(row)
    return out


def fetch_inputs(end: date, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    schedule = out / f"schedule_{end.year}.json"
    first = date(end.year, 3, 25)
    if not schedule.exists():
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": "1", "startDate": first.isoformat(),
                    "endDate": end.isoformat(), "hydrate": "team,linescore"},
            timeout=90,
        )
        response.raise_for_status()
        schedule.write_text(response.text)
    csv_dir = out / "statcast_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    repository = StatcastRepository(out / "statcast_cache")
    for offset in range((end - first).days + 1):
        day = first + timedelta(days=offset)
        target = csv_dir / f"{day}.csv"
        if not target.exists():
            repository.load_range(day, day).to_csv(target, index=False)


def schedule_games(schedule: dict, start: date, end: date) -> list[dict]:
    """Discard non-regular, unfinished and ambiguous doubleheader matchups."""
    days = schedule["dates"]
    games = []
    for day in days:
        current = date.fromisoformat(day["date"])
        if not start <= current <= end:
            continue
        qualified = [
            g for g in day["games"]
            if g["gameType"] == "R" and g["status"]["abstractGameState"] == "Final"
            and g["teams"]["away"].get("score") is not None
            and g["teams"]["home"].get("score") is not None
        ]
        keys = Counter(
            (g["teams"]["away"]["team"]["abbreviation"],
             g["teams"]["home"]["team"]["abbreviation"]) for g in qualified
        )
        for g in qualified:
            away = g["teams"]["away"]["team"]["abbreviation"]
            home = g["teams"]["home"]["team"]["abbreviation"]
            if keys[away, home] > 1:
                continue
            innings = g.get("linescore", {}).get("innings", [])
            if len(innings) < 9:
                continue
            games.append({
                "date": current, "game_pk": int(g["gamePk"]),
                "away": away, "home": home, "matchup": f"{away} @ {home}",
                "away_runs": int(g["teams"]["away"]["score"]),
                "home_runs": int(g["teams"]["home"]["score"]),
                "away_f5": sum(i.get("away", {}).get("runs") or 0 for i in innings[:5]),
                "home_f5": sum(i.get("home", {}).get("runs") or 0 for i in innings[:5]),
                "away_late": sum(i.get("away", {}).get("runs") or 0 for i in innings[5:]),
                "home_late": sum(i.get("home", {}).get("runs") or 0 for i in innings[5:]),
            })
    return games


def read_pitches(out: Path, end: date) -> pd.DataFrame:
    frames = []
    day = date(end.year, 3, 25)
    while day <= end:
        path = out / "statcast_csv" / f"{day}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Statcast day: {path}")
        if path.stat().st_size > 1000:
            frame = pd.read_csv(path, usecols=lambda c: c in USE_COLS, low_memory=False)
            frames.append(frame)
        day += timedelta(days=1)
    if not frames:
        raise ValueError("No Statcast pitch data")
    df = dedupe_pitches(pd.concat(frames, ignore_index=True))
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in ("pitcher", "inning", "bat_speed", "release_speed", "zone", "launch_speed",
                "launch_speed_angle", "woba_value", "woba_denom",
                "estimated_woba_using_speedangle"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team"] = np.where(df["inning_topbot"].eq("Bot"), df["home_team"], df["away_team"])
    df["pitching_team"] = np.where(df["inning_topbot"].eq("Top"),
                                    df["home_team"], df["away_team"])
    return _prepare(df)


def _rates(df: pd.DataFrame) -> tuple[float, float]:
    pa = df[df["pa_end"]]
    lg_woba = float(pa["woba_value"].sum() / pa["woba_denom"].sum())
    fb = df[df["bip"] & df["bb_type"].isin(("fly_ball", "popup"))]
    hr_fb = float(pa["events"].eq("home_run").sum() / max(len(fb), 1))
    return lg_woba, hr_fb


def _bp_metrics(g: pd.DataFrame, hr_fb: float) -> dict[str, float]:
    raw = _sp_raw(g, hr_fb)
    tracked = g[g["tracked"]]
    return raw | {
        "xERA": raw.get("xwOBA", math.nan),
        "xFIP": raw.get("xFIP_raw", math.nan),
        "SIERA": raw.get("SIERA_raw", math.nan),
        "SqUp Con%": float(tracked["squp"].mean()) if len(tracked) else math.nan,
    }


def pitching_scores(history: pd.DataFrame, day: date) -> tuple[dict[str, int], dict[int, int],
                                                                dict[int, dict[str, int]]]:
    """Statcast-only worksheet proxies (no FG centering or Stuff+)."""
    _, hr_fb = _rates(history)
    season = history[history["pitcher"].notna()]
    season_stats = {int(pid): _bp_metrics(g, hr_fb) for pid, g in season.groupby("pitcher")}
    qualified = [p for p, m in season_stats.items()
                 if m["IP"] >= SP_MIN_IP and m["pitches"] >= SP_MIN_PITCHES_SEASON]
    short = season[season["game_date"] >= day - timedelta(days=SP_SKILL_DAYS)]
    short_stats = {int(pid): _bp_metrics(g, hr_fb) for pid, g in short.groupby("pitcher")}
    skill = {"K%", "K-BB%", "CSW%", "O-Swing%"}
    sp_cols = [(c[0], c[1], c[2], c[3]) for c in SP_COLS if c[0] != "Stuff+"]
    sp_rank = score(
        [str(p) for p in qualified], sp_cols,
        lambda p, label: (short_stats.get(int(p), season_stats[int(p)])[label]
                          if label in skill and short_stats.get(int(p), {}).get("pitches", 0) >= 150
                          else season_stats[int(p)][label]),
        max(3, round(len(qualified) * 0.1)),
        lambda p: season_stats[int(p)]["SIERA"],
    ) if qualified else None
    pen = season.copy()
    pen = pen[pen["inning"] >= 6]
    pen_stats: dict[int | None, dict[str, dict[str, float]]] = {}
    for w in (None, 30, 60):
        cut = pen if w is None else pen[pen["game_date"] >= day - timedelta(days=w)]
        pen_stats[w] = {str(t): _bp_metrics(g, hr_fb) for t, g in cut.groupby("pitching_team")}
    cols = [c for c in BP_COLS if c[0] != "Stuff+"]
    bp_rank = score(
        sorted(pen_stats[None]), [(c[0], c[2], c[3], c[4]) for c in cols],
        lambda t, label: pen_stats[next(c[5] for c in cols if c[0] == label)]
        .get(t, {}).get(label, math.nan),
        K_TEAMS, lambda t: pen_stats[None][t]["SIERA"],
    )
    return (
        {t: bp_rank.total(t) or 0 for t in bp_rank.ranked},
        {p: sp_rank.total(str(p)) or 0 for p in qualified} if sp_rank else {},
        {p: sp_rank.pts[str(p)] for p in qualified} if sp_rank else {},
    )


def fangraphs_pitching_scores(day: date, out: Path) -> tuple[dict[str, int],
                                                               dict[int, int],
                                                               dict[int, dict[str, int]]]:
    """Use strictly dated FanGraphs leaderboard snapshots, including Stuff+."""
    cache = out / f"fg_pitching_{day}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    last = day - timedelta(days=1)
    sta = _fg("sta", day.year, 365, last)
    sta_skill = _fg("sta", day.year, SP_SKILL_DAYS, last)
    rel = {w: _fg("rel", day.year, w, last, team="0%2Cts")
           for w in (365, 60, 30)}
    pitchers = {int(r["xMLBAMID"]): r for r in sta
                if r.get("xMLBAMID") and r.get("IP") is not None
                and float(r["IP"]) >= SP_MIN_IP}
    skill = {int(r["xMLBAMID"]): r for r in sta_skill if r.get("xMLBAMID")}
    names = {
        w: {_FG_TO_MLB.get(str(row["TeamName"]), str(row["TeamName"])): row
            for row in rows}
        for w, rows in rel.items()
    }
    teams = sorted(names[365])
    bp_windows = {c[0]: 365 if c[5] is None else c[5] for c in BP_COLS}
    bp_rank = score(
        teams, [(c[0], c[2], c[3], c[4]) for c in BP_COLS],
        lambda t, label: float(names[bp_windows[label]].get(t, {}).get(
            next(c[1] for c in BP_COLS if c[0] == label)) or math.nan),
        K_TEAMS, lambda t: float(names[365][t].get("SIERA") or math.nan),
    )
    sp_windows = {c[0]: c[4] for c in SP_COLS}

    def sp_value(pid: str, label: str) -> float:
        key = "sp_stuff" if label == "Stuff+" else label
        row = (skill.get(int(pid), {}) if sp_windows[label] is not None
               else pitchers[int(pid)])
        value = row.get(key)
        if value is None:
            value = pitchers[int(pid)].get(key)
        return float(value) if value is not None else math.nan

    sp_rank = score(
        [str(p) for p in sorted(pitchers)],
        [(c[0], c[1], c[2], c[3]) for c in SP_COLS], sp_value,
        max(3, round(len(pitchers) * .1)),
        lambda p: float(pitchers[int(p)].get("SIERA") or math.nan),
    )
    result = (
        {t: bp_rank.total(t) or 0 for t in teams},
        {p: sp_rank.total(str(p)) or 0 for p in pitchers},
        {p: sp_rank.pts[str(p)] for p in pitchers},
    )
    pd.to_pickle(result, cache)
    return result


def batting_scores(history: pd.DataFrame, day: date) -> tuple[dict[str, Ranking],
                                                              dict[str, BatExtras]]:
    lg_woba, _ = _rates(history)
    masks = {
        "Overall": history["pitcher"].notna(),
        "vs LHP": history["p_throws"].eq("L"),
        "vs RHP": history["p_throws"].eq("R"),
        "Innings 6+": history["inning"] >= 6,
    }
    ranks = {}
    extras = {}
    for label, mask in masks.items():
        sub = history[mask]
        tables = {}
        for w in (30, 60, 90, 365):
            cut = sub if w == 365 else sub[sub["game_date"] >= day - timedelta(days=w)]
            tables[w] = {str(t): _bat_metrics(g, lg_woba) for t, g in cut.groupby("team")}
        ranks[label] = _score_offense(tables)
        extras[label] = BatExtras(
            pa={str(t): int(g["pa_end"].sum())
                for t, g in sub[sub["game_date"] >= day - timedelta(days=60)]
                .groupby("team")},
            windows=tables,
        )
    return ranks, extras


def alternative_scores(extras: BatExtras) -> dict[str, dict[str, float]]:
    """Aggregate metric-specific as-of percentile deciles and standardized values."""
    teams = sorted(extras.windows[90])
    decile = dict.fromkeys(teams, 0.0)
    zscore = dict.fromkeys(teams, 0.0)
    for metric, _, _, lower, window in BAT_COLS:
        values = pd.Series({team: extras.windows[window].get(team, {}).get(metric, math.nan)
                            for team in teams}, dtype=float)
        oriented = -values if lower else values
        ranked = oriented.rank(pct=True, method="average")
        standard = (oriented - oriented.mean()) / oriented.std(ddof=0)
        for team in teams:
            decile[team] += float(np.ceil(10 * ranked[team]))
            zscore[team] += float(standard[team])
    return {"decile": decile, "zscore": zscore}


def _actual_starters(today: pd.DataFrame, away: str, home: str) -> tuple[int | None, int | None]:
    starters = []
    for team in (away, home):
        first = today[(today["pitching_team"] == team) & (today["inning"] == 1)]
        ids = first["pitcher"].dropna()
        starters.append(int(ids.iloc[0]) if len(ids) else None)
    return starters[0], starters[1]


def _side_prices(rows: list[dict[str, object]], market: str, team: str) -> dict[str, float] | None:
    matches = [r for r in rows if r["market"] == market and
               str(r["selection"]).startswith(team + " ")]
    if len(matches) != 1:
        return None
    r = matches[0]
    m = re.search(r"([-+]\d+(?:\.\d+)?)$", str(r["selection"]))
    return {"american": float(str(r["american"])), "prob": float(str(r["no_vig_prob"])),
            "line": float(m[1]) if m else 0.0}


def assemble(games: list[dict], pitches: pd.DataFrame, prices: dict[date, dict],
             out: Path, fangraphs: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_date: dict[date, list[dict]] = defaultdict(list)
    for g in games:
        by_date[g["date"]].append(g)
    games_out: list[dict[str, object]] = []
    teams_out: list[dict] = []
    for day, slate in sorted(by_date.items()):
        cache = out / f"features_{day}.pkl"
        if cache.exists():
            bats, extras, pens, starters, sp_parts = pd.read_pickle(cache)
        else:
            history = pitches[pitches["game_date"] < day]
            bats, extras = batting_scores(history, day)
            pens, starters, sp_parts = pitching_scores(history, day)
            pd.to_pickle((bats, extras, pens, starters, sp_parts), cache)
        if fangraphs:
            pens, starters, sp_parts = fangraphs_pitching_scores(day, out)
        alternatives = {key: alternative_scores(value) for key, value in extras.items()}
        today = pitches[pitches["game_date"] == day]
        for g in slate:
            away, home = g["away"], g["home"]
            pair = prices.get(day, {}).get(g["matchup"], [])
            if not pair:
                continue
            # Exclude doubled labels even if only one game finished.
            day_matches = [h for h in slate if h["matchup"] == g["matchup"]]
            if len(day_matches) > 1:
                continue
            ap, hp = _actual_starters(today[
                today["home_team"].eq(home) & today["away_team"].eq(away)], away, home
            )
            if ap not in starters or hp not in starters or away not in pens or home not in pens:
                continue
            hand = {}
            for team, opposing in ((away, hp), (home, ap)):
                theirs = today[today["pitcher"].eq(opposing)]
                if theirs.empty:
                    break
                hand[team] = str(theirs["p_throws"].iloc[0])
            if len(hand) != 2:
                continue
            ml = _side_prices(pair, "game_ml", away)
            rl = _side_prices(pair, "game_rl", away)
            home_ml = _side_prices(pair, "game_ml", home)
            home_rl = _side_prices(pair, "game_rl", home)
            f5ml = _side_prices(pair, "f5_ml", away)
            f5rl = _side_prices(pair, "f5_rl", away)
            home_f5ml = _side_prices(pair, "f5_ml", home)
            home_f5rl = _side_prices(pair, "f5_rl", home)
            total = quote(pair, "game_total", "Over ")
            f5total = quote(pair, "f5_total", "F5 Over ")
            total_under = opposite_quote(pair, "game_total", total["line"]) if total else None
            if ml and home_ml and not math.isclose(
                ml["prob"] + home_ml["prob"], 1.0, abs_tol=1e-3
            ):
                continue
            if total and total_under and not math.isclose(
                total["prob"] + total_under["prob"], 1.0, abs_tol=1e-3
            ):
                continue
            result = g.copy()
            result["pitching_source"] = "FanGraphs as-of" if fangraphs else "Statcast proxy"
            result["sp_gap"] = starters[ap] - starters[hp]
            result["bp_gap"] = pens[away] - pens[home]
            result["sp_contact_away"] = sum(sp_parts[ap].get(k, 0) for k in (
                "xERA", "xFIP", "SIERA", "FB%", "Barrel%", "HardHit%"))
            result["sp_contact_home"] = sum(sp_parts[hp].get(k, 0) for k in (
                "xERA", "xFIP", "SIERA", "FB%", "Barrel%", "HardHit%"))
            result["run_diff"] = g["away_runs"] - g["home_runs"]
            result["total_runs"] = g["away_runs"] + g["home_runs"]
            result["win"] = float(result["run_diff"] > 0)
            for market, q in (("ml", ml), ("rl", rl), ("total", total),
                              ("f5ml", f5ml), ("f5rl", f5rl), ("f5total", f5total)):
                result[f"{market}_american"] = q["american"] if q else math.nan
                result[f"{market}_prob"] = q["prob"] if q else math.nan
                result[f"{market}_line"] = q["line"] if q else math.nan
            result["home_ml_american"] = home_ml["american"] if home_ml else math.nan
            result["home_rl_american"] = home_rl["american"] if home_rl else math.nan
            result["home_f5ml_american"] = (
                home_f5ml["american"] if home_f5ml else math.nan)
            result["home_f5ml_prob"] = home_f5ml["prob"] if home_f5ml else math.nan
            result["home_f5rl_american"] = (
                home_f5rl["american"] if home_f5rl else math.nan)
            result["home_f5rl_prob"] = (
                home_f5rl["prob"] if home_f5rl else math.nan)
            result["total_under_american"] = (
                total_under["american"] if total_under else math.nan)
            if ml:
                result["ml_residual"] = result["win"] - ml["prob"]
                result["ml_units"] = payout(ml["american"], result["win"])
                result["ml_logit"] = float(logit(ml["prob"]))
            if rl:
                d = result["run_diff"] + rl["line"]
                result["rl_cover"] = math.nan if d == 0 else float(d > 0)
                result["rl_residual"] = (
                    result["rl_cover"] - rl["prob"] if d != 0 else math.nan)
                result["rl_units"] = 0 if d == 0 else payout(rl["american"], result["rl_cover"])
            for market, runs in (("total", result["total_runs"]),
                                 ("f5total", g["away_f5"] + g["home_f5"])):
                q = total if market == "total" else f5total
                if q:
                    over = 0.5 if runs == q["line"] else float(runs > q["line"])
                    result[f"{market}_over"] = over
                    result[f"{market}_residual"] = (
                        over - q["prob"] if over != 0.5 else math.nan)
                    result[f"{market}_units"] = payout(q["american"], over)
            if f5ml:
                d = g["away_f5"] - g["home_f5"]
                result["f5ml_win"] = 0.5 if d == 0 else float(d > 0)
                result["f5ml_units"] = payout(f5ml["american"], result["f5ml_win"])
            if f5rl:
                d = g["away_f5"] - g["home_f5"] + f5rl["line"]
                result["f5rl_cover"] = 0.5 if d == 0 else float(d > 0)
                result["f5rl_units"] = payout(f5rl["american"], result["f5rl_cover"])
            for split in SPLITS:
                a = ("vs " + hand[away] + "HP") if split == "vs hand" else split
                h = ("vs " + hand[home] + "HP") if split == "vs hand" else split
                if away not in bats[a].pts or home not in bats[h].pts:
                    continue
                result[f"bat_{split}_gap"] = bats[a].total(away) - bats[h].total(home)
                result[f"bat_{split}_sum"] = bats[a].total(away) + bats[h].total(home)
                result[f"bat_{split}_away"] = bats[a].total(away)
                result[f"bat_{split}_home"] = bats[h].total(home)
                for variant in ("decile", "zscore"):
                    result[f"bat_{split}_{variant}_gap"] = (
                        alternatives[a][variant][away] - alternatives[h][variant][home])
                for metric in (c[0] for c in BAT_COLS):
                    result[f"{split}_{metric}_gap"] = (
                        bats[a].pts[away][metric] - bats[h].pts[home][metric])
                    result[f"{split}_{metric}_sum"] = (
                        bats[a].pts[away][metric] + bats[h].pts[home][metric])
                for family, cols in (("power", POWER), ("discipline", DISCIPLINE),
                                     ("production", PRODUCTION)):
                    result[f"{split}_{family}_gap"] = sum(
                        bats[a].pts[away][k] - bats[h].pts[home][k] for k in cols)
            games_out.append(result)
            for team, split, opp, runs, f5, late in (
                (away, "vs " + hand[away] + "HP", hp,
                 g["away_runs"], g["away_f5"], g["away_late"]),
                (home, "vs " + hand[home] + "HP", ap,
                 g["home_runs"], g["home_f5"], g["home_late"]),
            ):
                row = {"date": day, "game_pk": g["game_pk"], "team": team,
                       "hand": hand[team], "runs": runs, "f5_runs": f5,
                       "late_runs": late, "sp_contact_opp": sum(
                           sp_parts[opp].get(k, 0) for k in
                           ("xERA", "xFIP", "SIERA", "FB%", "Barrel%", "HardHit%")),
                       "total_residual": result.get("total_residual", math.nan),
                       "f5total_residual": result.get("f5total_residual", math.nan)}
                row["ml_prob"] = ((ml if team == away else home_ml) or {}).get(
                    "prob", math.nan)
                row["ml_residual"] = float(runs > (g["home_runs"] if team == away
                                                 else g["away_runs"])) - row["ml_prob"]
                for name, key in (("Overall", "Overall"), ("vs hand", split),
                                  ("Innings 6+", "Innings 6+")):
                    rank = bats[key]
                    row[f"{name}_score"] = rank.total(team)
                    for variant in ("decile", "zscore"):
                        row[f"{name}_{variant}"] = alternatives[key][variant][team]
                    row[f"{name}_pa"] = extras[key].pa.get(team, 0)
                    for metric in (c[0] for c in BAT_COLS):
                        row[f"{name}_{metric}"] = rank.pts.get(team, {}).get(metric, math.nan)
                for family, cols in (("power", POWER), ("discipline", DISCIPLINE),
                                     ("production", PRODUCTION)):
                    row[family] = sum(bats[split].pts.get(team, {}).get(k, 0) for k in cols)
                for w in (30, 60, 90, 365):
                    vals = extras[split].windows[w].get(team, {})
                    row[f"power_{w}"] = float(np.nanmean([
                        vals.get(k, math.nan) for k in POWER if k != "EV90"
                    ])) if vals else math.nan
                    row[f"discipline_{w}"] = float(np.nanmean([
                        vals.get(k, math.nan) * (-1 if k != "Z-Contact%" else 1)
                        for k in DISCIPLINE
                    ])) if vals else math.nan
                teams_out.append(row)
        print(day, len(games_out), "priced games", flush=True)
    return pd.DataFrame(games_out), pd.DataFrame(teams_out)


def season_late_games(games: list[dict], pitches: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Unpriced season-wide late scoring: include games without starter/price data."""
    grouped: dict[date, list[dict]] = defaultdict(list)
    for game in games:
        grouped[game["date"]].append(game)
    records: list[dict] = []
    for day, slate in sorted(grouped.items()):
        if day < date(day.year, 4, 25):
            continue
        cache = out / f"features_{day}.pkl"
        season_cache = out / f"season_bats_{day}.pkl"
        if cache.exists():
            ranks = pd.read_pickle(cache)[0]
        elif season_cache.exists():
            ranks = pd.read_pickle(season_cache)
        else:
            ranks, _ = batting_scores(pitches[pitches["game_date"] < day], day)
            pd.to_pickle(ranks, season_cache)
        for game in slate:
            for team, runs in ((game["away"], game["away_late"]),
                               (game["home"], game["home_late"])):
                if team in ranks["Innings 6+"].pts:
                    records.append({"date": day, "game_pk": game["game_pk"],
                                    "team": team, "late_runs": runs,
                                    "Innings 6+_score": ranks["Innings 6+"].total(team)})
        print("season", day, len(records), "team-games", flush=True)
    return pd.DataFrame(records)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def partial(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    c = np.column_stack([np.ones(len(x)), control])
    return corr(x - c @ np.linalg.lstsq(c, x, rcond=None)[0],
                y - c @ np.linalg.lstsq(c, y, rcond=None)[0])


def sign_p(values: np.ndarray) -> float:
    return min(1.0, 2 * (1 + min(
        int(np.count_nonzero(values <= 0)), int(np.count_nonzero(values >= 0))
    )) / (len(values) + 1))


def bootstrap(x: np.ndarray, y: np.ndarray, control: np.ndarray | None = None,
              draws: int = 2000, seed: int = SEED) -> dict[str, float]:
    """Two-sided, percentile game bootstrap; zero excluded if p < .05."""
    mask = np.isfinite(x) & np.isfinite(y)
    if control is not None:
        mask &= np.isfinite(control).all(axis=1)
    x, y = x[mask], y[mask]
    c = control[mask] if control is not None else None
    if len(x) < 8:
        return {"n": float(len(x)), "effect": math.nan, "lo": math.nan,
                "hi": math.nan, "p": math.nan}
    fn = (lambda a, b, z: partial(a, b, z)) if c is not None else (
        lambda a, b, z: corr(a, b))
    effect = fn(x, y, c)
    rng = np.random.default_rng(seed)
    values = np.array([fn(x[idx], y[idx], c[idx] if c is not None else None)
                       for idx in rng.integers(0, len(x), size=(draws, len(x)))])
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": float(len(x)), "effect": effect, "lo": math.nan,
                "hi": math.nan, "p": math.nan}
    return {"n": float(len(x)), "effect": effect,
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
            "p": sign_p(values)}


def bootstrap_mean(values: np.ndarray, draws: int = 2000,
                   null: float = 0.0) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0.0, "effect": math.nan, "lo": math.nan,
                "hi": math.nan, "p": math.nan}
    rng = np.random.default_rng(SEED)
    samples = np.array([values[idx].mean() for idx in
                        rng.integers(0, len(values), size=(draws, len(values)))])
    return {"n": float(len(values)), "effect": float(values.mean()),
            "lo": float(np.percentile(samples, 2.5)),
            "hi": float(np.percentile(samples, 97.5)),
            "p": sign_p(samples - null)}


def correlation_difference(x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                           draws: int = 2000) -> dict[str, float]:
    valid = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1, x2, y = x1[valid], x2[valid], y[valid]
    if len(y) < 8:
        return {"n": float(len(y)), "effect": math.nan,
                "lo": math.nan, "hi": math.nan, "p": math.nan}
    effect = corr(x1, y) - corr(x2, y)
    rng = np.random.default_rng(SEED)
    vals = np.array([
        corr(x1[idx], y[idx]) - corr(x2[idx], y[idx])
        for idx in rng.integers(0, len(y), size=(draws, len(y)))
    ])
    vals = vals[np.isfinite(vals)]
    return {"n": float(len(y)), "effect": effect,
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)),
            "p": sign_p(vals)}


def model(x: np.ndarray, y: np.ndarray, controls: np.ndarray, logistic: bool,
          draws: int = 2000) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(controls).all(axis=1)
    x, y, controls = x[mask], y[mask], controls[mask]
    if len(x) < 25 or np.std(x) == 0:
        return {"n": float(len(x)), "effect": math.nan, "lo": math.nan,
                "hi": math.nan, "p": math.nan}
    scale = np.std(x)
    design = np.column_stack([np.ones(len(x)), x / scale, controls])

    def fit(a: np.ndarray, b: np.ndarray) -> float:
        if logistic:
            def loss(beta: np.ndarray) -> float:
                t = a @ beta
                return float(np.logaddexp(0, t).sum() - np.dot(b, t))

            def gradient(beta: np.ndarray) -> np.ndarray:
                return a.T @ (expit(a @ beta) - b)

            res = minimize(loss, np.zeros(a.shape[1]), jac=gradient, method="BFGS")
            return float(res.x[1]) if np.isfinite(res.x[1]) else math.nan
        return float(np.linalg.lstsq(a, b, rcond=None)[0][1])

    effect = fit(design, y)
    rng = np.random.default_rng(SEED)
    values = np.array([fit(design[idx], y[idx]) for idx in
                       rng.integers(0, len(y), size=(draws, len(y)))])
    values = values[np.isfinite(values)]
    return {"n": float(len(x)), "effect": effect,
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
            "p": sign_p(values)}


def _record(rows: list[dict], label: str, result: dict[str, float]) -> None:
    rows.append({"test": label, **result})


def analyze(games: pd.DataFrame, teams: pd.DataFrame, out: Path,
            draws: int = 2000, season: pd.DataFrame | None = None) -> None:
    results: list[dict] = []
    metrics: list[dict] = []
    pitching = games[["sp_gap", "bp_gap"]].to_numpy(dtype=float)
    ml_logit = games.get("ml_logit", pd.Series(math.nan, index=games.index)).to_numpy(float)
    total_line = games["total_line"].to_numpy(float)
    outcomes = ("win", "run_diff", "rl_cover", "total_residual")
    for split in SPLITS:
        x = games[f"bat_{split}_gap"].to_numpy(float)
        for outcome in outcomes:
            y = games[outcome].to_numpy(float)
            _record(results, f"{split} {outcome} raw", bootstrap(x, y, draws=draws))
            _record(results, f"{split} {outcome} partial SP BP",
                    bootstrap(x, y, pitching, draws=draws))
            log = outcome in ("win", "rl_cover")
            _record(results, f"{split} {outcome} model SP BP",
                    model(x, y, pitching, log, draws))
            if outcome == "win":
                extra = ml_logit
            elif outcome == "total_residual":
                extra = total_line
            elif outcome == "rl_cover":
                extra = games["rl_prob"].to_numpy(float)
            else:
                extra = ml_logit
            _record(results, f"{split} {outcome} model SP BP market",
                    model(x, y, np.column_stack([pitching, extra]), log, draws))
        for metric in (c[0] for c in BAT_COLS):
            xx = games[f"{split}_{metric}_gap"].to_numpy(float)
            row: dict[str, object] = {"split": split, "metric": metric}
            for outcome in ("win", "run_diff", "rl_cover", "total_residual",
                            "ml_prob", "ml_residual"):
                row[f"r_{outcome}"] = bootstrap(
                    xx, games[outcome].to_numpy(float), draws=draws)["effect"]
            side = np.sign(xx)
            good = (side != 0) & np.isfinite(xx)
            row["n_sides"] = int(good.sum())
            row["side_accuracy"] = float(np.mean(
                (games["run_diff"].to_numpy(float)[good] * side[good]) > 0
            )) if good.any() else math.nan
            for market, target in (("ml", "win"), ("rl", "rl_cover"),
                                   ("total", "total_over")):
                outcome_values = games[target].to_numpy(float)
                american = games[f"{market}_american"].to_numpy(float)
                if market == "total":
                    summed = games[f"{split}_{metric}_sum"].to_numpy(float)
                    side = np.sign(summed - np.nanmedian(summed))
                    opposite = games.get("total_under_american", pd.Series(
                        math.nan, index=games.index)).to_numpy(float)
                else:
                    side = np.sign(xx)
                    opposite = games[f"home_{market}_american"].to_numpy(float)
                row[f"{market}_units"] = float(np.nansum([
                    payout(american[i] if side[i] > 0 else opposite[i],
                           outcome_values[i] if side[i] > 0 else
                           1 - outcome_values[i]) if side[i] != 0 else math.nan
                    for i in range(len(xx))
                ]))
            metrics.append(row)
        matched = games[(games["sp_gap"].abs() <= 1) & (games["bp_gap"].abs() <= 1)]
        _record(results, f"{split} pitching matched run_diff",
                bootstrap(matched[f"bat_{split}_gap"].to_numpy(float),
                          matched["run_diff"].to_numpy(float), draws=draws))
        strata = pd.qcut(games["sp_gap"] + games["bp_gap"], 3, duplicates="drop")
        for label, group in games.groupby(strata, observed=True):
            _record(results, f"{split} pitch tercile {label} run_diff",
                    bootstrap(group[f"bat_{split}_gap"].to_numpy(float),
                              group["run_diff"].to_numpy(float), draws=draws))
    _record(results, "B1 batting sum vs total residual", bootstrap(
        games["bat_vs hand_sum"].to_numpy(float),
        games["total_residual"].to_numpy(float), draws=draws))
    _record(results, "B1 bat gap vs ML market residual", bootstrap(
        games["bat_vs hand_gap"].to_numpy(float),
        games["ml_residual"].to_numpy(float), draws=draws))
    top = teams["vs hand_score"].quantile(2 / 3)
    both_top = games[(games["bat_vs hand_away"] >= top) &
                     (games["bat_vs hand_home"] >= top)]
    _record(results, "B1 both top tercile Over hit rate", bootstrap_mean(
        both_top["total_over"].where(both_top["total_over"] != .5).to_numpy(float),
        draws, null=.54))
    _record(results, "B1 both top tercile Over market residual", bootstrap_mean(
        both_top["total_residual"].to_numpy(float), draws))
    _record(results, "B1 both top tercile Over units/wager", bootstrap_mean(
        both_top["total_units"].to_numpy(float), draws))
    if season is not None:
        _record(results, "B5 season own late runs", bootstrap(
            season["Innings 6+_score"].to_numpy(float),
            season["late_runs"].to_numpy(float), draws=draws))
    for split in ("Overall", "vs hand", "Innings 6+"):
        _record(results, f"B5 {split} own late runs", bootstrap(
            teams[f"{split}_score"].to_numpy(float),
            teams["late_runs"].to_numpy(float), draws=draws))
    for family in ("power", "discipline", "production"):
        for outcome in ("runs", "total_residual", "f5total_residual",
                        "ml_prob", "ml_residual"):
            _record(results, f"B3 {family} {outcome}", bootstrap(
                teams[family].to_numpy(float),
                teams[outcome].to_numpy(float), draws=draws))
    _record(results, "B6 vs-hand F5 runs", bootstrap(
        teams["vs hand_score"].to_numpy(float), teams["f5_runs"].to_numpy(float),
        draws=draws))
    top_teams = teams[teams["vs hand_score"] >= top]
    for outcome in ("f5total_residual", "total_residual"):
        _record(results, f"B6 top offense {outcome}", bootstrap_mean(
            top_teams[outcome].to_numpy(float), draws))
    _record(results, "B6 top offense F5 minus full total residual", bootstrap_mean(
        (top_teams["f5total_residual"] - top_teams["total_residual"]).to_numpy(float),
        draws))
    for market in ("f5ml", "f5rl"):
        offered = games[(games["bat_vs hand_away"] >= top) & (games["f5ml_prob"] < 0.5)]
        home_dogs = games[(games["bat_vs hand_home"] >= top) &
                          (games["home_f5ml_prob"] < 0.5)]
        outcome = "f5ml_win" if market == "f5ml" else "f5rl_cover"
        wins = pd.concat([
            pd.DataFrame({"won": offered[outcome],
                          "american": offered[f"{market}_american"],
                          "prob": offered[f"{market}_prob"]}),
            pd.DataFrame({"won": 1 - home_dogs[outcome],
                          "american": home_dogs[f"home_{market}_american"],
                          "prob": home_dogs[f"home_{market}_prob"]}),
        ], ignore_index=True)
        settled = wins[wins["won"].notna() & wins["won"].ne(.5) &
                       wins["prob"].notna()]
        rate = bootstrap_mean(settled["won"].to_numpy(float), draws)
        edge = bootstrap_mean((settled["won"] - settled["prob"]).to_numpy(float),
                              draws)
        rate["p"] = edge["p"]
        _record(results, f"B6 top offense underdog {market} win rate", rate)
        _record(results, f"B6 top offense underdog {market} market residual", edge)
        units = [payout(a, w) for a, w in zip(wins["american"], wins["won"], strict=True)]
        _record(results, f"B6 top offense underdog {market} units/wager",
                bootstrap_mean(np.array(units, dtype=float), draws))
    for hand in ("L", "R"):
        for name, group in teams[teams["hand"] == hand].groupby(
            pd.cut(teams.loc[teams["hand"] == hand, "vs hand_pa"],
                   [-1, 399, 799, np.inf], labels=["<400", "400-799", "800+"]),
            observed=True,
        ):
            for split in ("Overall", "vs hand"):
                _record(results, f"B2 {hand} {name} {split} runs", bootstrap(
                    group[f"{split}_score"].to_numpy(float),
                    group["runs"].to_numpy(float), draws=draws))
            _record(results, f"B2 {hand} {name} split minus Overall r", correlation_difference(
                group["vs hand_score"].to_numpy(float),
                group["Overall_score"].to_numpy(float),
                group["runs"].to_numpy(float), draws))
    for w in (30, 60, 90, 365):
        for family in ("power", "discipline"):
            _record(results, f"B7 {family} {w} next runs", bootstrap(
                teams[f"{family}_{w}"].to_numpy(float),
                teams["runs"].to_numpy(float), draws=draws))
            if w != 365:
                _record(results, f"B7 {family} season minus {w} r",
                        correlation_difference(
                            teams[f"{family}_365"].to_numpy(float),
                            teams[f"{family}_{w}"].to_numpy(float),
                            teams["runs"].to_numpy(float), draws))
    for variant in ("decile", "zscore"):
        _record(results, f"B8 {variant} gap vs run diff", bootstrap(
            games[f"bat_vs hand_{variant}_gap"].to_numpy(float),
            games["run_diff"].to_numpy(float), draws=draws))
        _record(results, f"B8 {variant} score vs own runs", bootstrap(
            teams[f"vs hand_{variant}"].to_numpy(float),
            teams["runs"].to_numpy(float), draws=draws))
    _record(results, "B8 compressed score vs own runs", bootstrap(
        teams["vs hand_score"].to_numpy(float), teams["runs"].to_numpy(float),
        draws=draws))
    _record(results, "B8 decile minus compressed r", correlation_difference(
        games["bat_vs hand_decile_gap"].to_numpy(float),
        games["bat_vs hand_gap"].to_numpy(float),
        games["run_diff"].to_numpy(float), draws))
    rng = np.random.default_rng(SEED)
    power_terc = pd.qcut(
        teams["power"] + rng.uniform(0, 1e-6, len(teams)), 3, labels=False)
    contact_terc = pd.qcut(
        teams["sp_contact_opp"] + rng.uniform(0, 1e-6, len(teams)), 3, labels=False)
    cells = teams.assign(power_terc=power_terc, contact_terc=contact_terc)
    grid = []
    for (power, contact), group in cells.groupby(
        ["power_terc", "contact_terc"], observed=True
    ):
        for outcome in ("runs", "total_residual", "f5total_residual"):
            grid.append({"power_terc": power, "contact_terc": contact,
                         "outcome": outcome,
                         **bootstrap_mean(group[outcome].to_numpy(float), draws)})
    pd.DataFrame(grid).to_csv(out / "batting_interaction_grid.csv", index=False)
    interaction = teams[["power", "sp_contact_opp"]].to_numpy(float)
    product = interaction[:, 0] * interaction[:, 1]
    for outcome in ("runs", "total_residual", "f5total_residual"):
        _record(results, f"B4 additive interaction {outcome}", model(
            product, teams[outcome].to_numpy(float), interaction, False, draws))
    pd.DataFrame(results).to_csv(out / "batting_effects.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "batting_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 8, 4))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 22))
    parser.add_argument("--out", type=Path, default=Path.home() / ".mlb_engine/audit")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--season-late", action="store_true",
                        help="Compute season-wide unpriced B5 late-run sample")
    parser.add_argument("--fangraphs", action="store_true",
                        help="Use as-of-date FanGraphs pitching; cache under audit/")
    parser.add_argument("--fetch-inputs", action="store_true",
                        help="Download missing MLB schedule and daily Statcast inputs")
    args = parser.parse_args()
    if args.draws < 2000:
        parser.error("--draws must be >= 2000")
    if args.fetch_inputs:
        fetch_inputs(args.end, args.out)
    schedule = json.loads((args.out / f"schedule_{args.end.year}.json").read_text())
    games = schedule_games(schedule, args.start, args.end)
    prices = {day: read_prices(args.repo, day) for day in
              (args.start + timedelta(days=i)
               for i in range((args.end - args.start).days + 1))}
    pitches = read_pitches(args.out, args.end)
    game_frame, team_frame = assemble(games, pitches, prices, args.out, args.fangraphs)
    game_frame.to_csv(args.out / "batting_games.csv", index=False)
    team_frame.to_csv(args.out / "batting_teams.csv", index=False)
    season = None
    if args.season_late:
        season_path = args.out / "season_batting_teams.csv"
        if season_path.exists() and pd.read_csv(season_path, usecols=["date"])["date"].max() == (
            args.end.isoformat()
        ):
            season = pd.read_csv(season_path)
        else:
            season = season_late_games(schedule_games(
                schedule, date(args.start.year, 3, 25), args.end), pitches, args.out)
            season.to_csv(season_path, index=False)
    analyze(game_frame, team_frame, args.out, args.draws, season)
    print(f"{len(game_frame)} games, {len(team_frame)} team-games -> {args.out}")


if __name__ == "__main__":
    main()
