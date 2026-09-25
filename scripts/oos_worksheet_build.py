"""Rebuild the daily worksheet's starter, bullpen and batting scores as of every
game day of a season, with no look-ahead, and write one row per game.

The scoring is production's own: ``daily_worksheet.bullpen_ranking``,
``starter_table`` and ``offense_rankings`` are called unchanged, with the
FanGraphs reader swapped for the date-range tables cached by
``oos_fangraphs_pull.py`` (season-to-date and windowed leaderboards that end
the day before the game) and the Statcast frame cut to pitches thrown strictly
before the game day.

Per game the row carries the final, the linescore split into innings 1-5 and
6+, the actual starters (first-inning pitcher per side) and their hands, and for
each of the 11 SP and 11 BP metrics the compressed points (+2/+1/-2), the raw
value, the percentile rank and the z-score of both sides; for batting the
vs-hand, Overall and innings-6+ totals, rank positions, platoon spread and the
split's 60-day PA count.

    .venv/bin/python scripts/oos_worksheet_build.py --season 2025 --out ~/.mlb_engine/audit/oos_games_2025.csv
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from mlb_engine.output import daily_worksheet as dw
from mlb_engine.output.daily_worksheet import (
    BAT_COLS,
    BP_COLS,
    SP_COLS,
    Ranking,
    _prepare,
    bullpen_ranking,
    offense_rankings,
    starter_table,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oos_fangraphs_pull import DEFAULT_CACHE as FG_CACHE  # noqa: E402
from oos_fangraphs_pull import LAST_DAY, OPENING_DAY, load_day  # noqa: E402
from oos_statcast_pull import season_frame  # noqa: E402

log = logging.getLogger("oos_worksheet_build")

SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule"
TEAM_FIX = {"OAK": "ATH"}
SP_LABELS = [c[0] for c in SP_COLS]
BP_LABELS = [c[0] for c in BP_COLS]
BAT_LABELS = [c[0] for c in BAT_COLS]
SP_LOWER = {c[0]: c[3] for c in SP_COLS}
BP_LOWER = {c[0]: c[4] for c in BP_COLS}
BAT_LOWER = {c[0]: c[3] for c in BAT_COLS}


def fix_team(t: str) -> str:
    return TEAM_FIX.get(t, t)


# --- schedule / results ---------------------------------------------------------------


def season_games(season: int) -> pd.DataFrame:
    """Final regular-season games with scores and innings 1-5 / 6+ splits."""
    rows = []
    cur = OPENING_DAY[season]
    while cur <= LAST_DAY[season]:
        stop = min((cur.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
                   LAST_DAY[season])
        params: dict[str, str | int] = {
            "sportId": 1, "gameType": "R", "startDate": cur.isoformat(), "endDate": stop.isoformat(),
            "hydrate": "team,linescore",
        }
        r = requests.get(SCHED_URL, params=params, timeout=90)
        r.raise_for_status()
        for day in r.json().get("dates", []):
            for g in day.get("games", []):
                st = str(g.get("status", {}).get("detailedState", ""))
                if not st.startswith(("Final", "Completed")):
                    continue
                th, ta = g["teams"]["home"], g["teams"]["away"]
                if th.get("score") is None or ta.get("score") is None:
                    continue
                inn = g.get("linescore", {}).get("innings", [])
                if len(inn) < 5:
                    continue
                h15 = sum(int(i.get("home", {}).get("runs", 0) or 0) for i in inn[:5])
                a15 = sum(int(i.get("away", {}).get("runs", 0) or 0) for i in inn[:5])
                rows.append({
                    "game_pk": int(g["gamePk"]), "date": Date.fromisoformat(g["officialDate"]),
                    "commence": g["gameDate"], "dh": str(g.get("doubleHeader", "N")),
                    "home": fix_team(th["team"]["abbreviation"]), "away": fix_team(ta["team"]["abbreviation"]),
                    "home_runs": int(th["score"]), "away_runs": int(ta["score"]),
                    "home_r15": h15, "away_r15": a15,
                    "home_r6": int(th["score"]) - h15, "away_r6": int(ta["score"]) - a15,
                    "innings": len(inn),
                })
        cur = stop + timedelta(days=1)
    return pd.DataFrame(rows)


# --- per-day scoring -------------------------------------------------------------------


def _patch_fg(tables: dict[str, dict[int | None, list[dict]]]) -> None:
    def fake_fg(stats: str, season: int, window: int | None, as_of: Date, team: str = "0") -> list[dict]:
        rows = tables.get(stats, {}).get(window, [])
        if not rows:
            raise ValueError(f"no FanGraphs {stats} table for window {window}")
        return [dict(r, TeamName=fix_team(str(r.get("TeamName", "")))) for r in rows]
    dw._fg = fake_fg


def _rank_stats(r: Ranking, lower: dict[str, bool]) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    """Per metric: (entity -> percentile rank, entity -> z), both oriented higher = better."""
    out = {}
    for label in next(iter(r.vals.values()), {}):
        s = pd.Series({e: r.vals[e].get(label, math.nan) for e in r.vals}, dtype=float)
        if lower[label]:
            s = -s
        pct = s.rank(pct=True)
        sd = s.std()
        z = (s - s.mean()) / sd if sd and not math.isnan(sd) else s * 0
        out[label] = (pct.to_dict(), z.to_dict())
    return out


def _entity_cols(prefix: str, key: str, r: Ranking, labels: list[str],
                 stats: dict[str, tuple[dict[str, float], dict[str, float]]]) -> dict[str, float]:
    row: dict[str, float] = {}
    pts = r.pts.get(key)
    row[f"{prefix}_total"] = float(sum(pts.values())) if pts else math.nan
    for label in labels:
        row[f"{prefix}_{label}_pts"] = float(pts[label]) if pts and label in pts else math.nan
        row[f"{prefix}_{label}_val"] = r.vals.get(key, {}).get(label, math.nan)
        pct, z = stats.get(label, ({}, {}))
        row[f"{prefix}_{label}_pct"] = pct.get(key, math.nan)
        row[f"{prefix}_{label}_z"] = z.get(key, math.nan)
    return row


def _starters(day_pitches: pd.DataFrame) -> dict[tuple[str, str, str], tuple[int, str]]:
    """(home, away, side) -> (starter mlbam id, hand) from first-inning pitches."""
    out: dict[tuple[str, str, str], tuple[int, str]] = {}
    first = day_pitches[day_pitches["inning"].eq(1)]
    for (h, a, tb), g in first.groupby(["home_team", "away_team", "inning_topbot"]):
        side = "home" if tb == "Top" else "away"  # top: home team pitching
        pid = int(g["pitcher"].value_counts().idxmax())
        hand = str(g.loc[g["pitcher"].eq(pid), "p_throws"].iloc[0])
        out[(str(h), str(a), side)] = (pid, hand)
    return out


def score_day(day: Date, games: pd.DataFrame, season_df: pd.DataFrame,
              fg_cache: Path = FG_CACHE) -> list[dict]:
    """Every game on ``day`` scored from data strictly before ``day``."""
    prior = season_df[season_df["game_date"] < day]
    if prior["game_date"].nunique() < 7:
        return []
    tables = load_day(day, fg_cache, fetch=False)
    if not tables["rel"].get(None) or not tables["sta"].get(None):
        log.warning("%s: FanGraphs tables missing", day)
        return []
    _patch_fg(tables)
    prepared = _prepare(prior)
    bp = bullpen_ranking(day)
    bp_stats = _rank_stats(bp, BP_LOWER)
    try:
        sp = starter_table(prepared, day)
        sp_stats = _rank_stats(sp.ranking, SP_LOWER)
    except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        log.warning("%s: starter table failed: %s", day, exc)
        sp = None
        sp_stats = {}
    bats = offense_rankings(prepared, day)
    bat_stats = {k: _rank_stats(v, BAT_LOWER) for k, v in bats.items()}
    starters = _starters(season_df[season_df["game_date"] == day])
    last = prior["game_date"].max()
    pa60 = prepared[prepared["pa_end"] & (prepared["game_date"] > last - timedelta(days=60))]
    pa_split = pa60.groupby(["team", "p_throws"]).size()

    out = []
    for g in games.itertuples(index=False):
        row: dict = {
            "game_pk": g.game_pk, "date": day.isoformat(), "commence": g.commence, "dh": g.dh,
            "home": g.home, "away": g.away, "home_runs": g.home_runs, "away_runs": g.away_runs,
            "home_r15": g.home_r15, "away_r15": g.away_r15, "home_r6": g.home_r6, "away_r6": g.away_r6,
            "innings": g.innings, "sp_pool": len(sp.ranking.ranked) if sp else 0,
            "sp_k": sp.ranking.k if sp else 0,
        }
        for side, team, _opp in (("home", g.home, g.away), ("away", g.away, g.home)):
            st = starters.get((g.home, g.away, side))
            pid, hand = (st if st else (None, ""))
            row[f"{side}_sp_id"] = pid
            row[f"{side}_sp_hand"] = hand
            row.update(_entity_cols(f"{side}_bp", team, bp, BP_LABELS, bp_stats))
            if sp is not None and pid is not None:
                row.update(_entity_cols(f"{side}_sp", str(pid), sp.ranking, SP_LABELS, sp_stats))
                row[f"{side}_sp_ip"] = sp.ip.get(pid, math.nan)
            # batting: this lineup vs the opposing starter's hand
            opp_st = starters.get((g.home, g.away, "away" if side == "home" else "home"))
            opp_hand = opp_st[1] if opp_st else ""
            split = {"L": "vs LHP", "R": "vs RHP"}.get(opp_hand)
            row[f"{side}_opp_hand"] = opp_hand
            for label, key in (("bat_ovr", "Overall"), ("bat_6", "Innings 6+"), ("bat_vsl", "vs LHP"),
                               ("bat_vsr", "vs RHP")):
                r = bats[key]
                row[f"{side}_{label}_total"] = float(sum(r.pts[team].values())) if team in r.pts else math.nan
                row[f"{side}_{label}_rank"] = r.ranked.index(team) + 1 if team in r.ranked else math.nan
            if split:
                r = bats[split]
                row.update(_entity_cols(f"{side}_bat_hand", team, r, BAT_LABELS, bat_stats[split]))
                row[f"{side}_bat_hand_rank"] = row[f"{side}_bat_vsl_rank" if split == "vs LHP" else
                                                   f"{side}_bat_vsr_rank"]
                row[f"{side}_bat_hand_pa60"] = int(pa_split.get((team, opp_hand), 0))
            else:
                row[f"{side}_bat_hand_total"] = math.nan
        out.append(row)
    return out


def _worker(args: tuple[Date, pd.DataFrame, pd.DataFrame, Path]) -> list[dict]:
    day, games, season_df, fg_cache = args
    try:
        return score_day(day, games, season_df, fg_cache)
    except Exception as exc:  # noqa: BLE001 -- one bad day must not sink the season
        log.error("%s failed: %s", day, exc)
        return []


_SEASON_DF: pd.DataFrame | None = None


def _init_worker(season: int, cache_dir: Path) -> None:
    global _SEASON_DF
    _SEASON_DF = season_frame(season, cache_dir)


def _worker_pool(args: tuple[Date, pd.DataFrame, Path]) -> list[dict]:
    day, games, fg_cache = args
    assert _SEASON_DF is not None
    return _worker((day, games, _SEASON_DF, fg_cache))


def build_season(season: int, out: Path, workers: int = 4, statcast_cache: Path | None = None,
                 fg_cache: Path = FG_CACHE, limit: int | None = None,
                 start: Date | None = None) -> pd.DataFrame:
    from oos_statcast_pull import DEFAULT_CACHE as SC_CACHE
    sc = statcast_cache or SC_CACHE
    games = season_games(season)
    days = sorted(d for d in games["date"].unique() if start is None or d >= start)
    if limit:
        days = days[:limit]
    jobs = [(d, games[games["date"] == d], fg_cache) for d in days]
    rows: list[dict] = []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(season, sc)) as ex:
        for i, res in enumerate(ex.map(_worker_pool, jobs)):
            rows.extend(res)
            if i % 10 == 0:
                log.info("%s: %s/%s days, %s games", season, i + 1, len(days), len(rows))
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N game days only (smoke test)")
    ap.add_argument("--start", type=Date.fromisoformat, default=None, help="first game day to score")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out or Path.home() / ".mlb_engine" / "audit" / f"oos_games_{args.season}.csv"
    df = build_season(args.season, out, args.workers, limit=args.limit, start=args.start)
    print(f"{args.season}: {len(df)} games -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
