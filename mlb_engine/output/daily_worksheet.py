"""The daily MLB worksheet: the hand matchup sheet, its prices, and its receipt.

Three ranking tables are rebuilt every morning on the rule the sheet has always
used -- per metric the best three teams score +2, the worst three -2, everyone
else +1 -- each metric read on the window where it is most reliable:

* **Bullpens** (FanGraphs team relievers): xERA, xFIP, SIERA, K%, CSW%, HardHit%,
  K-BB%, Stuff+, squared-up contact%, O-Swing%, FB%.
* **Offenses** (Statcast pitch level, so the split can be cut): ISO, Barrel%,
  EV90, Z-Contact%, Whiff%, O-Swing%, squared-up%, wRC, xwOBA, HardHit%, blast%
  -- overall, vs LHP, vs RHP and innings 6+.
* **Starters** (Statcast pitch level, FanGraphs' starter list and Stuff+): xERA,
  xFIP, SIERA, K%, K-BB%, CSW%, HardHit%, Stuff+, FB%, O-Swing%, Barrel%. With
  ~170 arms the ends are the best and worst 10%, not three.

Each game then gets a row per team -- the offense vs the opposing starter's hand
(the starter's innings), the offense from the 6th on (the bullpen's), the pen
and the starter -- and a weighted total that gives the starter the share of a
game he actually pitches: bat x1 + bat 6+ x1 + pen x1.5 + starter x3. The
away-minus-home row is the disparity; the right-hand block ranks the slate by
it, biggest mismatch first.

Every game is written to ``worksheet_ledger.csv`` with its gap, the best
moneyline and run-line price on the board and VSIN's handle/bets at Circa and
DK, so the sheet can be audited later against what it could have been bet at.
The next morning fills in the final and the workbook's Audit tab tallies the
favoured side's record by gap band against the market's own implied win rate --
the one comparison that says whether the gap knows anything the price does not.

Nothing here feeds a price. It is a record.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from mlb_engine.config import Config
from mlb_engine.data import http
from mlb_engine.data.oddsapi import OddsAPIClient
from mlb_engine.data.statcast import StatcastRepository, dedupe_pitches
from mlb_engine.data.vsin import VSINClient, _norm_name
from mlb_engine.market.ev import MarketQuote
from mlb_engine.output.totals_audit import Final, finals
from mlb_engine.output.totals_sheet import _FG_ABBR, _FG_HEADERS, _FG_URL
from mlb_engine.schemas import Game, Hand, Slate, TeamGameInfo

log = logging.getLogger(__name__)

LEDGER_NAME = "worksheet_ledger.csv"
TOP, BOT, MID, K_TEAMS = 2, -2, 1, 3
SP_SKILL_DAYS = 42
SP_MIN_PITCHES = 150
SP_MIN_IP, SP_MIN_PITCHES_SEASON = 20.0, 300
# Weighted total: the starter pitches ~40% of a game's defensive innings, the pen
# ~20%, and the lineup is the other side of both. Batting is counted once per
# phase, not five ways like the hand sheet did.
WEIGHTS = {"bat vs hand": 1.0, "bat 6+": 1.0, "bp": 1.5, "sp": 3.0}
WEIGHTS_VERSION = "w1-1-1.5-3"
GAP_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-9", 0, 10), ("10-19", 10, 20), ("20-34", 20, 35), ("35+", 35, math.inf)
)
_FG_TO_MLB = {v: k for k, v in _FG_ABBR.items()}

# (label, fangraphs key, decimals, is_pct, lower_is_better, window_days | None=season)
BP_COLS: tuple[tuple[str, str, int, bool, bool, int | None], ...] = (
    ("xERA", "xERA", 2, False, True, None),
    ("xFIP", "xFIP", 2, False, True, None),
    ("SIERA", "SIERA", 2, False, True, None),
    ("K%", "K%", 1, True, False, 60),
    ("CSW%", "C+SwStr%", 1, True, False, 60),
    ("HardHit%", "HardHit%", 1, True, True, None),
    ("K-BB%", "K-BB%", 1, True, False, 60),
    ("Stuff+", "sp_stuff", 0, False, False, 30),
    ("SqUp Con%", "SquaredUpContact%", 1, True, True, None),
    ("O-Swing%", "O-Swing%", 1, True, False, 60),
    ("FB%", "FB%", 1, True, True, None),
)
# (label, decimals, is_pct, lower_is_better, window_days)
BAT_COLS: tuple[tuple[str, int, bool, bool, int], ...] = (
    ("ISO", 3, False, False, 60),
    ("Barrel%", 1, True, False, 60),
    ("EV90", 1, False, False, 60),
    ("Z-Contact%", 1, True, False, 30),
    ("Whiff%", 1, True, True, 30),
    ("O-Swing%", 1, True, True, 30),
    ("SqUp Con%", 1, True, False, 60),
    ("wRC", 0, False, False, 90),
    ("xwOBA", 3, False, False, 90),
    ("HardHit%", 1, True, False, 60),
    ("Blast Con%", 1, True, False, 60),
)
# (label, decimals, is_pct, lower_is_better, window_days | None=season)
SP_COLS: tuple[tuple[str, int, bool, bool, int | None], ...] = (
    ("xERA", 2, False, True, None),
    ("xFIP", 2, False, True, None),
    ("SIERA", 2, False, True, None),
    ("K%", 1, True, False, SP_SKILL_DAYS),
    ("K-BB%", 1, True, False, SP_SKILL_DAYS),
    ("CSW%", 1, True, False, SP_SKILL_DAYS),
    ("HardHit%", 1, True, True, None),
    ("Stuff+", 0, False, False, SP_SKILL_DAYS),
    ("FB%", 1, True, True, None),
    ("O-Swing%", 1, True, False, SP_SKILL_DAYS),
    ("Barrel%", 1, True, True, None),
)
BAT_SPLITS = ("Overall", "vs LHP", "vs RHP", "Innings 6+")

_SWINGS = [
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
    "missed_bunt", "foul_bunt", "bunt_foul_tip",
]
_WHIFFS = ["swinging_strike", "swinging_strike_blocked", "missed_bunt"]
_AB_EVENTS = [
    "single", "double", "triple", "home_run", "strikeout", "field_out", "grounded_into_double_play",
    "force_out", "double_play", "fielders_choice", "fielders_choice_out", "field_error",
    "strikeout_double_play", "triple_play",
]
_OUTS = {
    "field_out": 1, "strikeout": 1, "force_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "fielders_choice_out": 1, "caught_stealing_2b": 1, "other_out": 1,
    "grounded_into_double_play": 2, "double_play": 2, "strikeout_double_play": 2,
    "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "triple_play": 3,
}
LG_RPA, WOBA_SCALE = 0.118, 1.25

GREEN, PINK, ORANGE, YEL, GREY = "C6EFCE", "F8CBAD", "FFC000", "FFFF00", "BFBFBF"
NEON_GREEN, PALE_GREEN = (0.22, 1.00, 0.08), (0.85, 1.00, 0.80)
LIGHT_PINK, NEON_PINK = (1.00, 0.85, 0.92), (1.00, 0.08, 0.58)


# --- scoring ----------------------------------------------------------------------


@dataclass
class Ranking:
    """One scored table: per-entity points by metric, raw values, and the order."""

    pts: dict[str, dict[str, int]]
    vals: dict[str, dict[str, float]]
    stars: dict[str, set[str]]
    ranked: list[str]
    k: int

    def total(self, key: str) -> int | None:
        p = self.pts.get(key)
        return sum(p.values()) if p else None


def score(
    entities: list[str],
    cols: list[tuple[str, int, bool, bool]],
    value: Callable[[str, str], float],  # (entity, label) -> value, nan allowed
    k: int,
    tiebreak: Callable[[str], float],
    stars: dict[str, set[str]] | None = None,
) -> Ranking:
    """Best ``k`` per metric +2, worst ``k`` -2, rest +1; NaN sorts last (worst)."""
    pts: dict[str, dict[str, int]] = {e: {} for e in entities}
    vals: dict[str, dict[str, float]] = {e: {} for e in entities}
    for label, nd, pct, lower in cols:
        scored = {e: float(value(e, label)) for e in entities}
        order = sorted(
            entities,
            key=lambda e: (math.isnan(scored[e]), scored[e] if lower else -scored[e]),
        )
        for i, e in enumerate(order):
            v = scored[e]
            vals[e][label] = round(v * (100 if pct else 1), nd) if not math.isnan(v) else math.nan
            pts[e][label] = TOP if i < k else BOT if i >= len(order) - k else MID
    total = {e: sum(pts[e].values()) for e in entities}
    ranked = sorted(entities, key=lambda e: (-total[e], tiebreak(e)))
    return Ranking(pts, vals, stars or {e: set() for e in entities}, ranked, k)


# --- bullpens (FanGraphs) -----------------------------------------------------------


def _fg(stats: str, season: int, window: int | None, as_of: Date, team: str = "0") -> list[dict]:
    url = _FG_URL.format(stats=stats, season=season, month=0 if window is None else 1000, team=team)
    if window is not None:
        url += f"&startdate={(as_of - timedelta(days=window)).isoformat()}&enddate={as_of.isoformat()}"
    resp = http.get(url, headers=_FG_HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("data")
    if not isinstance(data, list):
        raise ValueError(f"FanGraphs {stats} leaderboard returned no data")
    return data


def bullpen_ranking(as_of: Date) -> Ranking:
    tables: dict[int | None, dict[str, dict]] = {}
    for w in {c[5] for c in BP_COLS}:
        rows = _fg("rel", as_of.year, w, as_of, team="0%2Cts")
        tables[w] = {_FG_TO_MLB.get(r["TeamName"], r["TeamName"]): r for r in rows}
    teams = sorted(tables[None])
    key_of = {c[0]: c[1] for c in BP_COLS}
    win_of = {c[0]: c[5] for c in BP_COLS}

    def value(t: str, label: str) -> float:
        v = tables[win_of[label]].get(t, {}).get(key_of[label])
        return math.nan if v is None else float(v)

    return score(teams, [(c[0], c[2], c[3], c[4]) for c in BP_COLS], value, K_TEAMS,
                 lambda t: tables[None][t]["SIERA"])


# --- Statcast frame ------------------------------------------------------------------


def season_pitches(cache_dir: Path, as_of: Date) -> pd.DataFrame:
    """Every pitch of the season through the day before ``as_of``.

    Completed months are cached once; only the current month is refreshed, so a
    morning run downloads days, not the season.
    """
    repo = StatcastRepository(cache_dir)
    end = as_of - timedelta(days=1)
    start = Date(as_of.year, 3, 1)
    frames = []
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        stop = min(nxt - timedelta(days=1), end)
        frames.append(repo.load_range(cur, stop, refresh=stop == end and stop.month == as_of.month))
        cur = nxt
    df = dedupe_pitches(pd.concat(frames, ignore_index=True))
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team"] = np.where(df["inning_topbot"].eq("Bot"), df["home_team"], df["away_team"])
    df["swing"] = df["description"].isin(_SWINGS)
    df["whiff"] = df["description"].isin(_WHIFFS)
    df["called"] = df["description"].eq("called_strike")
    df["in_zone"] = df["zone"].between(1, 9)
    df["bip"] = df["type"].eq("X") & df["launch_speed"].notna()
    df["pa_end"] = df["events"].notna() & df["events"].ne("")
    if "bat_speed" in df.columns:
        # Savant: squared-up = EV / max possible EV (1.23*bat + 0.2116*pitch) >= .80 on
        # contact; blast = squared-up with bat speed + 100*ratio >= 164.
        ratio = df["launch_speed"] / (1.23 * df["bat_speed"] + 0.2116 * df["release_speed"] * 0.92)
        df["tracked"] = df["swing"] & ~df["whiff"] & df["bat_speed"].notna()
        df["squp"] = df["bip"] & df["bat_speed"].notna() & (ratio >= 0.80)
        df["blast"] = df["squp"] & (df["bat_speed"] + 100 * ratio >= 164)
    else:
        df["tracked"] = False
        df["squp"] = False
        df["blast"] = False
    return df


# --- offenses (Statcast) -------------------------------------------------------------


def _bat_metrics(g: pd.DataFrame, lg_woba: float) -> dict[str, float]:
    pa = g[g["pa_end"]]
    ev = g["events"]
    ab = pa["events"].isin(_AB_EVENTS).sum()
    tb = (ev.eq("double") * 2 + ev.eq("triple") * 3 + ev.eq("home_run") * 4 + ev.eq("single")).sum()
    hits = ev.isin(["single", "double", "triple", "home_run"]).sum()
    bip = g[g["bip"]]
    z = g[g["in_zone"] & g["swing"]]
    o = g[~g["in_zone"] & g["zone"].notna()]
    sw = g[g["swing"]]
    tracked = g[g["tracked"]]
    denom = pa["woba_denom"].sum()
    woba = pa["woba_value"].sum() / denom if denom else math.nan
    return {
        "ISO": (tb - hits) / ab if ab else math.nan,
        "Barrel%": bip["launch_speed_angle"].eq(6).mean(),
        "EV90": bip["launch_speed"].quantile(0.9),
        "Z-Contact%": 1 - z["whiff"].mean(),
        "Whiff%": sw["whiff"].mean(),
        "O-Swing%": o["swing"].mean(),
        "SqUp Con%": tracked["squp"].mean() if len(tracked) else math.nan,
        "wRC": ((woba - lg_woba) / WOBA_SCALE + LG_RPA) * len(pa),
        "xwOBA": pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"]).sum() / denom
        if denom else math.nan,
        "HardHit%": (bip["launch_speed"] >= 95).mean(),
        "Blast Con%": tracked["blast"].mean() if len(tracked) else math.nan,
    }


def offense_rankings(df: pd.DataFrame, as_of: Date) -> dict[str, Ranking]:
    """Overall, vs LHP, vs RHP and innings 6+ offense tables."""
    lg = df[df["pa_end"]]
    lg_woba = lg["woba_value"].sum() / lg["woba_denom"].sum()
    masks = {
        "Overall": df["pitcher"].notna(),
        "vs LHP": df["p_throws"].eq("L"),
        "vs RHP": df["p_throws"].eq("R"),
        "Innings 6+": df["inning"] >= 6,
    }
    last = max(df["game_date"])
    out: dict[str, Ranking] = {}
    for split, mask in masks.items():
        sub = df[mask]
        tables: dict[int, dict[str, dict[str, float]]] = {}
        for w in {c[4] for c in BAT_COLS}:
            cut = sub[sub["game_date"] > last - timedelta(days=w)]
            tables[w] = {str(t): _bat_metrics(g, lg_woba) for t, g in cut.groupby("team")}
        out[split] = _score_offense(tables)
    return out


def _score_offense(tables: dict[int, dict[str, dict[str, float]]]) -> Ranking:
    win_of = {c[0]: c[4] for c in BAT_COLS}
    return score(
        sorted(tables[90]), [(c[0], c[1], c[2], c[3]) for c in BAT_COLS],
        lambda t, label: tables[win_of[label]].get(t, {}).get(label, math.nan),
        K_TEAMS, lambda t: -tables[90][t]["xwOBA"],
    )


# --- starters (Statcast + FanGraphs) --------------------------------------------------


def _sp_raw(g: pd.DataFrame, lg_hr_fb: float) -> dict[str, float]:
    pa = g[g["pa_end"]]
    n = len(pa)
    ev = pa["events"]
    k = ev.isin(["strikeout", "strikeout_double_play"]).sum()
    bb = ev.isin(["walk", "intent_walk"]).sum()
    hbp = ev.eq("hit_by_pitch").sum()
    bip = g[g["bip"]]
    gb = bip["bb_type"].eq("ground_ball").sum()
    fb = bip["bb_type"].eq("fly_ball").sum()
    pu = bip["bb_type"].eq("popup").sum()
    ip = ev.map(_OUTS).fillna(0).sum() / 3
    o = g[~g["in_zone"] & g["zone"].notna()]
    if n == 0:
        return {"pitches": len(g), "IP": ip}
    kr, bbr, nb = k / n, bb / n, (gb - fb - pu) / n
    siera = (6.145 - 16.986 * kr + 11.434 * bbr - 1.858 * nb + 7.653 * kr**2
             + (6.664 if nb < 0 else -6.664) * nb**2 + 10.130 * kr * nb - 5.195 * bbr * nb)
    denom = pa["woba_denom"].sum()
    return {
        "pitches": len(g), "IP": ip,
        "K%": kr, "K-BB%": kr - bbr,
        "CSW%": (g["called"] | g["whiff"]).mean(),
        "O-Swing%": o["swing"].mean(),
        "HardHit%": (bip["launch_speed"] >= 95).mean(),
        "Barrel%": bip["launch_speed_angle"].eq(6).mean(),
        "FB%": (fb + pu) / len(bip) if len(bip) else math.nan,
        "SIERA_raw": siera,
        "xFIP_raw": (13 * (fb + pu) * lg_hr_fb + 3 * (bb + hbp) - 2 * k) / ip if ip else math.nan,
        "xwOBA": pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"]).sum() / denom
        if denom else math.nan,
    }


@dataclass
class StarterTable:
    ranking: Ranking
    name: dict[int, str]
    team: dict[int, str]
    ip: dict[int, float]
    by_name: dict[str, int]  # normalized name -> mlbam id

    def pts_for(self, name: str) -> int | None:
        pid = self.by_name.get(_norm_name(name))
        return self.ranking.total(str(pid)) if pid is not None else None


def starter_table(df: pd.DataFrame, as_of: Date) -> StarterTable:
    season = _fg("sta", as_of.year, None, as_of)
    fg_season = {int(r["xMLBAMID"]): r for r in season if r.get("xMLBAMID")}
    last = max(df["game_date"])
    skill_rows = _fg("sta", as_of.year, SP_SKILL_DAYS, last)
    stuff: dict[int | None, dict[int, float]] = {
        None: {p: float(r["sp_stuff"]) for p, r in fg_season.items() if r.get("sp_stuff") is not None},
        SP_SKILL_DAYS: {int(r["xMLBAMID"]): float(r["sp_stuff"]) for r in skill_rows
                        if r.get("xMLBAMID") and r.get("sp_stuff") is not None},
    }
    sub = df[df["pitcher"].isin(fg_season)].copy()
    lg_pa = sub[sub["pa_end"]]
    lg_fb = sub[sub["bip"] & sub["bb_type"].isin(["fly_ball", "popup"])]
    lg_hr_fb = lg_pa["events"].eq("home_run").sum() / max(len(lg_fb), 1)

    season_m = {int(p): _sp_raw(g, lg_hr_fb) for p, g in sub.groupby("pitcher")}
    # Centre the formula stats on FanGraphs' own season numbers, and fit xERA as
    # a line through xwOBA against, so the scale is FanGraphs' even though the
    # split is ours.
    cal = pd.DataFrame([
        {"xw": m["xwOBA"], "xera": fg_season[p]["xERA"], "sfg": fg_season[p]["SIERA"],
         "sraw": m["SIERA_raw"], "xfg": fg_season[p]["xFIP"], "xraw": m["xFIP_raw"]}
        for p, m in season_m.items()
        if fg_season[p].get("xERA") and m.get("IP", 0) >= 30
    ]).dropna()
    xera_b, xera_a = np.polyfit(cal["xw"], cal["xera"], 1)
    siera_c = float((cal["sfg"] - cal["sraw"]).mean())
    xfip_c = float((cal["xfg"] - cal["xraw"]).mean())

    def finish(m: dict[str, float]) -> dict[str, float]:
        if "xwOBA" in m:
            m["xERA"] = xera_a + xera_b * m["xwOBA"]
            m["SIERA"] = m["SIERA_raw"] + siera_c
            m["xFIP"] = m["xFIP_raw"] + xfip_c
        return m

    for m in season_m.values():
        finish(m)
    cut = sub[sub["game_date"] > last - timedelta(days=SP_SKILL_DAYS)]
    skill_m = {int(p): finish(_sp_raw(g, lg_hr_fb)) for p, g in cut.groupby("pitcher")
               if len(g) >= SP_MIN_PITCHES}
    pitchers = [p for p, m in season_m.items()
                if m.get("IP", 0) >= SP_MIN_IP and m["pitches"] >= SP_MIN_PITCHES_SEASON]
    k = max(3, round(len(pitchers) * 0.10))
    stars: dict[str, set[str]] = {str(p): set() for p in pitchers}
    win_of = {c[0]: c[4] for c in SP_COLS}

    def value(key: str, label: str) -> float:
        p = int(key)
        win = win_of[label]
        if label == "Stuff+":
            v = stuff[win].get(p)
            if v is None:
                stars[key].add(label)
                return stuff[None].get(p, math.nan)
            return v
        if win is None or p not in skill_m:
            if win is not None:
                stars[key].add(label)
            return season_m[p].get(label, math.nan)
        return skill_m[p].get(label, math.nan)

    ranking = score([str(p) for p in pitchers], [(c[0], c[1], c[2], c[3]) for c in SP_COLS],
                    value, k, lambda key: season_m[int(key)].get("SIERA", 9.0), stars)
    name = {p: str(fg_season[p]["PlayerName"]) for p in pitchers}
    team = {p: _FG_TO_MLB.get(str(fg_season[p].get("TeamName", "")), str(fg_season[p].get("TeamName", "")))
            for p in pitchers}
    return StarterTable(
        ranking, name, team, {p: round(season_m[p]["IP"], 1) for p in pitchers},
        {_norm_name(n): p for p, n in name.items()},
    )


# --- prices ---------------------------------------------------------------------------


@dataclass
class Prices:
    """Best board price and VSIN splits for one team in one game."""

    ml: float | None = None
    ml_book: str = ""
    rl_line: float | None = None
    rl: float | None = None
    rl_book: str = ""
    dk_ml_handle: float | None = None
    dk_ml_bets: float | None = None
    dk_rl_handle: float | None = None
    dk_rl_bets: float | None = None
    circa_ml_handle: float | None = None
    circa_ml_bets: float | None = None
    circa_rl_handle: float | None = None
    circa_rl_bets: float | None = None

    def ml_text(self) -> str:
        return f"{self.ml:+.0f} {self.ml_book}" if self.ml is not None else ""

    def rl_text(self) -> str:
        if self.rl_line is None:
            return ""
        px = f" {self.rl:+.0f} {self.rl_book}" if self.rl is not None else ""
        return f"{self.rl_line:+.1f}{px}"

    @staticmethod
    def _hb(h: float | None, b: float | None) -> str:
        return "" if h is None or b is None else f"{h:.0f}/{b:.0f}"

    def split_cells(self) -> list[str]:
        return [
            self._hb(self.circa_ml_handle, self.circa_ml_bets), self._hb(self.circa_rl_handle, self.circa_rl_bets),
            self._hb(self.dk_ml_handle, self.dk_ml_bets), self._hb(self.dk_rl_handle, self.dk_rl_bets),
        ]


def _best(quotes: list[MarketQuote]) -> MarketQuote | None:
    return max(quotes, key=lambda q: q.american) if quotes else None


def fetch_prices(cfg: Config, slate: Slate) -> dict[tuple[str, str], Prices]:
    """(matchup, abbrev) -> Prices. Board from the Odds API (one credit), splits from VSIN."""
    out = {(g.matchup(), tm.abbrev): Prices() for g in slate.games for tm in (g.away, g.home)}
    try:
        board = OddsAPIClient(cfg.creds.odds_api_key, cache_dir=cfg.odds_cache_dir).fetch(
            slate, include_props=False
        )
    except Exception as exc:  # noqa: BLE001 -- the sheet still goes out without a board
        log.warning("worksheet: odds board unavailable: %s", exc)
        board = {}
    for (matchup, market, selection), quotes in board.items():
        abbrev = selection.split()[0] if selection else ""
        p = out.get((matchup, abbrev))
        if p is None:
            continue
        q = _best(quotes)
        if q is None:
            continue
        if market == "game_ml":
            p.ml, p.ml_book = q.american, q.book
        elif market == "game_rl":
            try:
                p.rl_line = float(selection.split()[-1])
            except ValueError:
                continue
            p.rl, p.rl_book = q.american, q.book
    try:
        sides = VSINClient(cfg.creds).fetch_side_splits(slate)
    except Exception as exc:  # noqa: BLE001
        log.warning("worksheet: VSIN splits unavailable: %s", exc)
        sides = {}
    for (matchup, abbrev, book), s in sides.items():
        p = out.get((matchup, abbrev))
        if p is None:
            continue
        if p.ml is None and s.ml_american is not None:
            p.ml, p.ml_book = s.ml_american, f"vsin {book}"
        if p.rl_line is None and s.rl_line is not None:
            p.rl_line = s.rl_line
        if book == "draftkings":
            p.dk_ml_handle, p.dk_ml_bets = s.ml.handle_pct, s.ml.bets_pct
            p.dk_rl_handle, p.dk_rl_bets = s.rl.handle_pct, s.rl.bets_pct
        elif book == "circa":
            p.circa_ml_handle, p.circa_ml_bets = s.ml.handle_pct, s.ml.bets_pct
            p.circa_rl_handle, p.circa_rl_bets = s.rl.handle_pct, s.rl.bets_pct
    return out


def implied(american: float | None, opposite: float | None) -> float | None:
    """No-vig win probability of the side priced ``american`` against ``opposite``."""
    if american is None or opposite is None:
        return None

    def raw(a: float) -> float:
        return 100 / (a + 100) if a > 0 else -a / (-a + 100)

    p, q = raw(american), raw(opposite)
    return p / (p + q) if p + q else None


# --- the sheet ------------------------------------------------------------------------


@dataclass
class TeamLine:
    abbrev: str
    starter: str
    hand: str
    bat_vs_hand: int | None
    bat_6: int | None
    bp: int | None
    sp: int | None

    def cells(self) -> list[int | None]:
        return [self.bat_vs_hand, self.bat_6, self.bp, self.sp]

    def raw_total(self) -> int:
        return sum(v for v in self.cells() if v is not None)

    def weighted(self) -> float:
        w = [WEIGHTS[k] for k in WEIGHTS]
        return round(sum(wt * v for wt, v in zip(w, self.cells(), strict=True) if v is not None), 1)

    def label(self) -> str:
        return f"{self.abbrev.lower()} ({self.starter or 'TBD'}{', ' + self.hand if self.hand else ''})"


def team_line(
    team: TeamGameInfo, opp: TeamGameInfo, bats: dict[str, Ranking], pens: Ranking, sps: StarterTable
) -> TeamLine:
    pp = team.probable_pitcher
    opp_hand = opp.probable_pitcher.throws.value if opp.probable_pitcher and opp.probable_pitcher.throws else ""
    split = {"L": "vs LHP", "R": "vs RHP"}.get(opp_hand, "Overall")
    return TeamLine(
        team.abbrev, pp.name if pp else "", pp.throws.value if pp and pp.throws else "",
        bats[split].total(team.abbrev), bats["Innings 6+"].total(team.abbrev),
        pens.total(team.abbrev), sps.pts_for(pp.name) if pp else None,
    )


@dataclass
class LedgerRow:
    date: str
    game: str
    game_pk: int
    away: str
    home: str
    away_sp: str
    home_sp: str
    away_bat: int | None
    away_bat6: int | None
    away_bp: int | None
    away_sp_pts: int | None
    home_bat: int | None
    home_bat6: int | None
    home_bp: int | None
    home_sp_pts: int | None
    away_w: float
    home_w: float
    gap: float  # away_w - home_w
    fav: str
    complete: bool  # both starters scored
    away_ml: float | None = None
    home_ml: float | None = None
    away_rl_line: float | None = None
    away_rl: float | None = None
    home_rl: float | None = None
    fav_implied: float | None = None  # no-vig market win probability of the favoured side
    dk_ml_handle_away: float | None = None
    dk_ml_bets_away: float | None = None
    dk_rl_handle_away: float | None = None
    dk_rl_bets_away: float | None = None
    circa_ml_handle_away: float | None = None
    circa_ml_bets_away: float | None = None
    circa_rl_handle_away: float | None = None
    circa_rl_bets_away: float | None = None
    dk_ml_handle_home: float | None = None
    dk_ml_bets_home: float | None = None
    dk_rl_handle_home: float | None = None
    dk_rl_bets_home: float | None = None
    circa_ml_handle_home: float | None = None
    circa_ml_bets_home: float | None = None
    circa_rl_handle_home: float | None = None
    circa_rl_bets_home: float | None = None
    weights: str = WEIGHTS_VERSION
    away_runs: int | None = None
    home_runs: int | None = None
    result: str = ""  # fav | dog | tie | "" (ungraded)
    rl_result: str = ""  # away | home | push | ""

    @property
    def graded(self) -> bool:
        return self.result != ""


def _pk(g: Game) -> int:
    return g.game_pk


def build_rows(
    slate: Slate, bats: dict[str, Ranking], pens: Ranking, sps: StarterTable,
    prices: dict[tuple[str, str], Prices],
) -> list[tuple[Game, TeamLine, TeamLine, LedgerRow]]:
    out = []
    for g in sorted(slate.games, key=lambda x: x.game_datetime_utc or ""):
        a = team_line(g.away, g.home, bats, pens, sps)
        h = team_line(g.home, g.away, bats, pens, sps)
        gap = round(a.weighted() - h.weighted(), 1)
        fav = g.away.abbrev if gap >= 0 else g.home.abbrev
        pa, ph = prices[(g.matchup(), g.away.abbrev)], prices[(g.matchup(), g.home.abbrev)]
        fav_ml, dog_ml = (pa.ml, ph.ml) if fav == g.away.abbrev else (ph.ml, pa.ml)
        row = LedgerRow(
            slate.slate_date.isoformat(), g.matchup(), _pk(g), g.away.abbrev, g.home.abbrev,
            a.starter, h.starter, a.bat_vs_hand, a.bat_6, a.bp, a.sp, h.bat_vs_hand, h.bat_6, h.bp, h.sp,
            a.weighted(), h.weighted(), gap, fav, a.sp is not None and h.sp is not None,
            pa.ml, ph.ml, pa.rl_line, pa.rl, ph.rl, implied(fav_ml, dog_ml),
            pa.dk_ml_handle, pa.dk_ml_bets, pa.dk_rl_handle, pa.dk_rl_bets,
            pa.circa_ml_handle, pa.circa_ml_bets, pa.circa_rl_handle, pa.circa_rl_bets,
            ph.dk_ml_handle, ph.dk_ml_bets, ph.dk_rl_handle, ph.dk_rl_bets,
            ph.circa_ml_handle, ph.circa_ml_bets, ph.circa_rl_handle, ph.circa_rl_bets,
        )
        out.append((g, a, h, row))
    return out


# --- ledger ---------------------------------------------------------------------------


def ledger_path(cfg: Config) -> Path:
    return cfg.audit_dir / LEDGER_NAME


_FIELD_TYPES = {f.name: f.type for f in fields(LedgerRow)}


def _parse(name: str, v: str) -> object:
    if v == "":
        return None if name not in ("result", "rl_result", "weights", "away_sp", "home_sp", "fav") else ""
    t = _FIELD_TYPES[name]
    if t in ("int", "int | None"):
        return int(float(v))
    if t in ("float", "float | None"):
        return float(v)
    if t == "bool":
        return v == "True"
    return v


def load_ledger(path: Path) -> list[LedgerRow]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        rows = []
        for rec in csv.DictReader(fh):
            kwargs = {k: _parse(k, v) for k, v in rec.items() if k in _FIELD_TYPES}
            rows.append(LedgerRow(**kwargs))  # type: ignore[arg-type]
        return rows


def save_ledger(path: Path, rows: list[LedgerRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(LedgerRow)])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in asdict(r).items()})
    tmp.replace(path)


def merge(ledger: list[LedgerRow], fresh: list[LedgerRow]) -> list[LedgerRow]:
    """A day's rows are re-written by a re-run until graded; a graded row is never touched."""
    kept = {(r.date, r.game_pk): r for r in ledger}
    for r in fresh:
        old = kept.get((r.date, r.game_pk))
        if old is None or not old.graded:
            kept[(r.date, r.game_pk)] = r
    return sorted(kept.values(), key=lambda r: (r.date, r.game))


def grade(rows: list[LedgerRow], results: dict[int, Final]) -> int:
    n = 0
    for r in rows:
        if r.graded or r.game_pk not in results:
            continue
        _, ar, hr = results[r.game_pk]
        r.away_runs, r.home_runs = ar, hr
        if ar == hr:
            r.result = "tie"
        else:
            winner = r.away if ar > hr else r.home
            r.result = "fav" if winner == r.fav else "dog"
        if r.away_rl_line is not None:
            m = ar + r.away_rl_line - hr
            r.rl_result = "away" if m > 0 else "home" if m < 0 else "push"
        n += 1
    return n


def grade_pending(rows: list[LedgerRow], today: Date) -> int:
    days = sorted({Date.fromisoformat(r.date) for r in rows if not r.graded and Date.fromisoformat(r.date) < today})
    n = 0
    for d in days:
        try:
            n += grade(rows, finals(d))
        except Exception as exc:  # noqa: BLE001
            log.warning("worksheet: finals for %s unavailable: %s", d, exc)
    return n


# --- audit ----------------------------------------------------------------------------


@dataclass
class BandTally:
    band: str
    n: int = 0
    fav_wins: int = 0
    implied_sum: float = 0.0
    implied_n: int = 0
    rl_n: int = 0
    rl_fav_covers: int = 0

    @property
    def win_rate(self) -> float | None:
        return self.fav_wins / self.n if self.n else None

    @property
    def market_rate(self) -> float | None:
        return self.implied_sum / self.implied_n if self.implied_n else None

    @property
    def edge(self) -> float | None:
        w, m = self.win_rate, self.market_rate
        return None if w is None or m is None else w - m

    @property
    def rl_rate(self) -> float | None:
        return self.rl_fav_covers / self.rl_n if self.rl_n else None


def band_of(gap: float) -> str:
    for name, lo, hi in GAP_BANDS:
        if lo <= abs(gap) < hi:
            return name
    return GAP_BANDS[-1][0]


def tally(rows: list[LedgerRow]) -> list[BandTally]:
    out = {name: BandTally(name) for name, _, _ in GAP_BANDS}
    total = BandTally("all")
    for r in rows:
        if r.result not in ("fav", "dog") or not r.complete or r.weights != WEIGHTS_VERSION:
            continue
        for t in (out[band_of(r.gap)], total):
            t.n += 1
            t.fav_wins += r.result == "fav"
            if r.fav_implied is not None:
                t.implied_sum += r.fav_implied
                t.implied_n += 1
            if r.rl_result in ("away", "home"):
                t.rl_n += 1
                t.rl_fav_covers += (r.rl_result == "away") == (r.fav == r.away)
    return list(out.values()) + [total]


def summary_text(rows: list[LedgerRow], graded_day: Date | None, today: Date | None = None) -> str:
    lines = []
    if today is not None:
        slate = sorted(
            (r for r in rows if r.date == today.isoformat()), key=lambda r: -abs(r.gap)
        )
        if slate:
            lines.append(
                f"Worksheet {today} by weighted gap: "
                + "; ".join(
                    f"{r.game} {r.fav} {abs(r.gap):.1f}{'' if r.complete else '*'}" for r in slate
                )
                + ("  (* starter missing)" if any(not r.complete for r in slate) else "")
            )
    if graded_day is not None:
        day = [r for r in rows if r.date == graded_day.isoformat() and r.result in ("fav", "dog")]
        if day:
            w = sum(r.result == "fav" for r in day)
            lines.append(f"Worksheet {graded_day}: favoured side {w}-{len(day) - w}"
                         + "".join(f"; {r.game} {r.fav} by {abs(r.gap):.0f} {'hit' if r.result == 'fav' else 'missed'}"
                                   for r in sorted(day, key=lambda r: -abs(r.gap))[:3]))
    t = tally(rows)
    if t[-1].n:
        parts = []
        for b in t:
            if not b.n:
                continue
            m = f" vs mkt {b.market_rate * 100:.0f}%" if b.market_rate is not None else ""
            parts.append(f"{b.band}: {b.fav_wins}-{b.n - b.fav_wins} ({b.fav_wins / b.n * 100:.0f}%{m})")
        lines.append("Worksheet gap bands (fav record vs market implied): " + "; ".join(parts))
    return "\n".join(lines)


# --- workbook -------------------------------------------------------------------------


def _lerp(a: tuple[float, float, float], b: tuple[float, float, float], f: float) -> str:
    rgb = tuple(a[i] + (b[i] - a[i]) * f for i in range(3))
    return "".join(f"{int(round(c * 255)):02X}" for c in rgb)


def _row_colour(rank: int, n: int) -> str:
    half = max((n - 1) / 2, 1e-9)
    if rank - 1 <= half:
        return _lerp(NEON_GREEN, PALE_GREEN, (rank - 1) / half)
    return _lerp(LIGHT_PINK, NEON_PINK, (rank - 1 - half) / half)


def _hdr(ws: Worksheet, row: int, values: list[str], col0: int = 1) -> None:
    for j, v in enumerate(values):
        c = ws.cell(row=row, column=col0 + j, value=v)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor=GREY)
        c.alignment = Alignment(horizontal="center", wrap_text=True)


def _ranking_sheet(
    ws: Worksheet, rk: Ranking, cols: list[str],
    name_of: Callable[[str], str] | None = None, notes: tuple[str, ...] = (),
) -> None:
    hdr = ["Rk", "Team" if name_of is None else "Pitcher", "PTS"] + cols
    _hdr(ws, 1, hdr)
    for i, e in enumerate(rk.ranked, 1):
        vals = []
        for c in cols:
            v = rk.vals[e].get(c, math.nan)
            s = "n/a" if isinstance(v, float) and math.isnan(v) else (f"{v:.0f}" if c == "Stuff+" else f"{v}")
            p = rk.pts[e].get(c)
            vals.append(s + ("*" if c in rk.stars.get(e, set()) else "") + (f" ({p:+d})" if p is not None else ""))
        ws.append([i, name_of(e) if name_of else e, rk.total(e)] + vals)
        fill = PatternFill("solid", fgColor=_row_colour(i, len(rk.ranked)))
        for j in range(1, len(hdr) + 1):
            ws.cell(row=i + 1, column=j).fill = fill
            ws.cell(row=i + 1, column=j).alignment = Alignment(horizontal="center")
    ws.append([])
    for n in notes:
        ws.append([n])
    ws.freeze_panes = "D2"
    for j in range(1, len(hdr) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 6 if j in (1, 3) else 12
    ws.column_dimensions["B"].width = 20


def write_workbook(
    path: Path, day: Date, as_of: Date,
    rows: list[tuple[Game, TeamLine, TeamLine, LedgerRow]],
    prices: dict[tuple[str, str], Prices],
    pens: Ranking, bats: dict[str, Ranking], sps: StarterTable,
    ledger: list[LedgerRow], graded_day: Date | None,
) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Matchups"
    cols = list(WEIGHTS)
    odds_cols = ["ML best", "RL best", "Circa ML h/b", "Circa RL h/b", "DK ML h/b", "DK RL h/b"]
    _hdr(ws, 1, ["game", "team"] + cols + ["raw TOTAL", "wTOTAL"] + odds_cols)
    nc = 2 + len(cols) + 2 + len(odds_cols)
    r = 2
    for g, a, h, lr in rows:
        pa, ph = prices[(g.matchup(), g.away.abbrev)], prices[(g.matchup(), g.home.abbrev)]
        flag = "" if lr.complete else " *"
        ws.append([g.matchup(), a.label()] + a.cells() + [a.raw_total(), a.weighted(), pa.ml_text(), pa.rl_text()] + pa.split_cells())
        ws.append(["", h.label()] + h.cells() + [h.raw_total(), h.weighted(), ph.ml_text(), ph.rl_text()] + ph.split_cells())
        diff = [None if x is None and y is None else (x or 0) - (y or 0) for x, y in zip(a.cells(), h.cells(), strict=True)]
        ws.append(["", "away - home" + flag] + diff + [a.raw_total() - h.raw_total(), lr.gap])
        for j in range(1, nc + 1):
            ws.cell(row=r, column=j).fill = PatternFill("solid", fgColor=GREEN)
            ws.cell(row=r + 1, column=j).fill = PatternFill("solid", fgColor=PINK)
            ws.cell(row=r + 2, column=j).fill = PatternFill("solid", fgColor=ORANGE)
            ws.cell(row=r + 2, column=j).font = Font(bold=True)
            for k in range(3):
                ws.cell(row=r + k, column=j).alignment = Alignment(horizontal="center")
        ws.cell(row=r + 2, column=2 + len(cols) + 2).fill = PatternFill("solid", fgColor=YEL)
        r += 4
        ws.append([])
    ws.append([
        "bat vs hand = offense split vs the opposing starter's hand (overall table if no probable) -- the lineup's "
        "read for the starter's innings; bat 6+ = innings 6+ offense split -- the read against the bullpen; bp = "
        "bullpen table; sp = starter's PTS in the Starters tab (blank = not a FanGraphs starter or under "
        f"{SP_MIN_IP:.0f} IP; * = a starter is missing so the total is incomplete and the game is not audited). "
        f"raw TOTAL = plain sum. wTOTAL = bat x{WEIGHTS['bat vs hand']} + bat 6+ x{WEIGHTS['bat 6+']} + "
        f"bp x{WEIGHTS['bp']} + sp x{WEIGHTS['sp']}. away - home row = disparity; the wTOTAL gap is the sheet's "
        "call. ML/RL best = best price across books at write time; h/b = VSIN handle% / bets% at Circa and DK."
    ])
    # Ranked block to the right.
    c0 = nc + 2
    _hdr(ws, 1, ["rank", "game", "favored", "underdog", "away wTOTAL", "home wTOTAL", "w gap", "favored ML", "mkt %"], c0)
    ranked = sorted(rows, key=lambda t: -abs(t[3].gap))
    for i, (g, _a, _h, lr) in enumerate(ranked):
        fav_p = prices[(g.matchup(), lr.fav)]
        dog = g.home.abbrev if lr.fav == g.away.abbrev else g.away.abbrev
        vals = [i + 1, g.matchup() + ("" if lr.complete else " *"), lr.fav.lower(), dog.lower(),
                lr.away_w, lr.home_w, abs(lr.gap), fav_p.ml_text(),
                f"{lr.fav_implied * 100:.0f}%" if lr.fav_implied is not None else ""]
        fill = PatternFill("solid", fgColor=GREEN if i / max(len(ranked) - 1, 1) < 0.5 else PINK)
        for j, v in enumerate(vals):
            c = ws.cell(row=2 + i, column=c0 + j, value=v)
            c.fill = fill
            c.alignment = Alignment(horizontal="center")
    ws.cell(row=3 + len(ranked), column=c0,
            value="ranked by |weighted gap|: top = biggest mismatch, bottom = most evenly matched; mkt % = no-vig win probability the board gives the favoured side")
    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 34
    for j in range(3, c0 + 10):
        ws.column_dimensions[get_column_letter(j)].width = 12

    # Audit tab.
    wa = wb.create_sheet("Audit")
    _hdr(wa, 1, ["gap band", "games", "fav W", "fav L", "fav win %", "market implied %", "edge (pts)", "fav RL cover %"])
    for b in tally(ledger):
        wa.append([
            b.band, b.n, b.fav_wins, b.n - b.fav_wins,
            round(b.win_rate * 100, 1) if b.win_rate is not None else None,
            round(b.market_rate * 100, 1) if b.market_rate is not None else None,
            round(b.edge * 100, 1) if b.edge is not None else None,
            round(b.rl_rate * 100, 1) if b.rl_rate is not None else None,
        ])
    wa.append([])
    wa.append([
        "Favoured side = the team with the higher wTOTAL. Market implied = the no-vig win probability its best "
        "moneyline gave it when the sheet was written; edge = fav win % minus that. Only games with both starters "
        f"scored and weights {WEIGHTS_VERSION} count. Under ~50 games a band the numbers are noise."
    ])
    if graded_day is not None:
        wa.append([])
        _hdr(wa, wa.max_row + 1, ["date", "game", "fav", "gap", "fav ML", "mkt %", "final", "result", "RL"])
        for lr in sorted((x for x in ledger if x.date == graded_day.isoformat()), key=lambda x: -abs(x.gap)):
            fav_ml = lr.away_ml if lr.fav == lr.away else lr.home_ml
            wa.append([
                lr.date, lr.game, lr.fav, abs(lr.gap), fav_ml,
                round(lr.fav_implied * 100) if lr.fav_implied is not None else None,
                f"{lr.away_runs}-{lr.home_runs}" if lr.graded else "",
                lr.result, lr.rl_result,
            ])
    for j in range(1, 10):
        wa.column_dimensions[get_column_letter(j)].width = 14

    # Ledger tab.
    wl = wb.create_sheet("Ledger")
    names = [f.name for f in fields(LedgerRow)]
    _hdr(wl, 1, names)
    for lr in sorted(ledger, key=lambda x: (x.date, x.game), reverse=True):
        wl.append([getattr(lr, n) for n in names])
    wl.freeze_panes = "A2"

    # Ranking tabs.
    _ranking_sheet(wb.create_sheet("Bullpens"), pens, [c[0] for c in BP_COLS], notes=(
        f"Per metric best {K_TEAMS} +{TOP}, worst {K_TEAMS} {BOT}, others +{MID}. Windows: K%, CSW%, K-BB%, O-Swing% 60d; "
        "Stuff+ 30d; xERA, xFIP, SIERA, HardHit%, SqUp Con%, FB% season. Lower is better on xERA, xFIP, SIERA, "
        "HardHit%, SqUp Con%, FB%. Source: FanGraphs team relievers.",
    ))
    for split in BAT_SPLITS:
        _ranking_sheet(wb.create_sheet(f"Bat {split}"), bats[split], [c[0] for c in BAT_COLS], notes=(
            f"Offense, {split}, through {as_of}. Per metric best {K_TEAMS} +{TOP}, worst {K_TEAMS} {BOT}, others +{MID}. "
            "Windows: Z-Contact%, Whiff%, O-Swing% 30d; ISO, Barrel%, EV90, HardHit%, SqUp Con%, Blast Con% 60d; "
            "wRC, xwOBA 90d. Lower is better on Whiff%, O-Swing%. Source: Statcast pitch level.",
        ))
    _ranking_sheet(
        wb.create_sheet("Starters"), sps.ranking, [c[0] for c in SP_COLS],
        name_of=lambda key: f"{sps.name[int(key)]} ({sps.team[int(key)]}, {sps.ip[int(key)]} IP)",
        notes=(
            f"{len(sps.ranking.ranked)} FanGraphs starters with {SP_MIN_IP:.0f}+ IP through {as_of}. Per metric best "
            f"{sps.ranking.k} +{TOP}, worst {sps.ranking.k} {BOT}, others +{MID}. Windows: K%, K-BB%, CSW%, O-Swing%, "
            f"Stuff+ {SP_SKILL_DAYS}d (* = under {SP_MIN_PITCHES} pitches, season used); xERA, xFIP, SIERA, HardHit%, "
            "FB%, Barrel% season. Lower is better on xERA, xFIP, SIERA, HardHit%, FB%, Barrel%. Statcast pitch level; "
            "Stuff+ FanGraphs; xERA fitted to FanGraphs' season xERA, SIERA/xFIP centred on FanGraphs.",
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# --- entry point ----------------------------------------------------------------------


# (date, team, MLB's listed probable) -> (starter on the posted lineup card, hand).
# Hand corrections from the user's lineup card for a day MLB's probable lagged.
STARTER_OVERRIDES: dict[tuple[Date, str, str], tuple[str, Hand]] = {
    (Date(2026, 9, 14), "CWS", "Sean Newcomb"): ("Sean Burke", Hand.R),
}


def apply_starter_overrides(slate: Slate) -> Slate:
    for g in slate.games:
        for tm in (g.away, g.home):
            pp = tm.probable_pitcher
            if pp is None:
                continue
            fix = STARTER_OVERRIDES.get((slate.slate_date, tm.abbrev, pp.name))
            if fix is not None:
                tm.probable_pitcher = pp.model_copy(update={"name": fix[0], "throws": fix[1]})
    return slate


def run_worksheet(cfg: Config, day: Date, slate: Slate) -> tuple[Path, str]:
    """Build the day's worksheet, record it, grade what is pending, write the workbook and note."""
    slate = apply_starter_overrides(slate)
    as_of = day - timedelta(days=1)
    log.info("worksheet: loading season Statcast")
    df = _prepare(season_pitches(cfg.cache_dir, day))
    log.info("worksheet: scoring bullpens, offenses, starters")
    pens = bullpen_ranking(as_of)
    bats = offense_rankings(df, as_of)
    sps = starter_table(df, as_of)
    prices = fetch_prices(cfg, slate)
    rows = build_rows(slate, bats, pens, sps, prices)

    path = ledger_path(cfg)
    ledger = merge(load_ledger(path), [r for _, _, _, r in rows])
    graded = grade_pending(ledger, day)
    save_ledger(path, ledger)
    prev = day - timedelta(days=1)
    graded_day = prev if any(r.date == prev.isoformat() and r.graded for r in ledger) else None
    log.info("worksheet: %d games written, %d graded", len(rows), graded)

    out = cfg.output_dir / f"worksheet_{day.isoformat()}.xlsx"
    write_workbook(out, day, as_of, rows, prices, pens, bats, sps, ledger, graded_day)
    text = summary_text(ledger, graded_day, day)
    (cfg.output_dir / f"worksheet_{day.isoformat()}.txt").write_text(text + ("\n" if text else ""))
    return out, text
