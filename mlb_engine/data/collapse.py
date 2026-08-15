"""Capture per-pitcher, per-inning run attribution from the MLB play-by-play feed.

This is an *observe-only* audit capture: it records how many runs each pitcher was
charged in each inning (with correct inherited-runner credit via the feed's
``responsiblePitcher``), so the volatility / big-inning ("collapse") rate of every
starter and reliever can be measured empirically. It changes no pricing, tiering,
or bet selection.

For each game it writes one row per (pitcher, inning-appearance):

    date, game_pk, inning, half, pitch_team, bat_team, pitcher_id, pitcher_name,
    is_start, batters_faced, runs_charged, earned_charged

From these rows a downstream study computes P(>=2 runs / inning), the variance and
skew of runs allowed per inning, and starter-vs-reliever splits.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from mlb_engine.data import http

BASE = "https://statsapi.mlb.com/api/v1"

log = logging.getLogger(__name__)

FIELDNAMES = [
    "date",
    "game_pk",
    "inning",
    "half",
    "pitch_team",
    "bat_team",
    "pitcher_id",
    "pitcher_name",
    "is_start",
    "batters_faced",
    "runs_charged",
    "earned_charged",
]


@dataclass
class _Agg:
    pitch_team: str
    bat_team: str
    pitcher_name: str = ""
    batters_faced: int = 0
    runs_charged: int = 0
    earned_charged: int = 0


@dataclass
class InningLine:
    date: str
    game_pk: int
    inning: int
    half: str  # "top" | "bottom"
    pitch_team: str
    bat_team: str
    pitcher_id: int
    pitcher_name: str
    is_start: bool
    batters_faced: int
    runs_charged: int
    earned_charged: int


def _pbp_cache_path(cache_dir: Path | None, game_pk: int) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / "pbp" / f"{game_pk}.json"


def _fetch_pbp(
    game_pk: int,
    session: requests.Session,
    cache_dir: Path | None,
    timeout: int,
) -> dict | None:
    """Fetch the play-by-play feed, caching it, with a cache fallback offline."""
    cache_path = _pbp_cache_path(cache_dir, game_pk)
    try:
        pbp = session.get(f"{BASE}/game/{game_pk}/playByPlay", timeout=timeout).json()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(pbp))
        return pbp
    except requests.RequestException as exc:
        log.warning("play-by-play fetch failed for %s: %s", game_pk, exc)
        if cache_path is not None and cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError):
                return None
        return None


def _team_abbrevs(boxscore: dict) -> tuple[str, str]:
    teams = boxscore.get("teams", {}) or {}
    home = ((teams.get("home", {}) or {}).get("team", {}) or {}).get("abbreviation", "")
    away = ((teams.get("away", {}) or {}).get("team", {}) or {}).get("abbreviation", "")
    return home, away


def game_inning_lines(
    pbp: dict,
    game_pk: int,
    date: str,
    home_abbr: str,
    away_abbr: str,
) -> list[InningLine]:
    """Build per-pitcher inning rows from one game's play-by-play."""
    plays = pbp.get("allPlays", []) or []
    # (inning, half, pitcher_id) -> mutable aggregate
    agg: dict[tuple[int, str, int], _Agg] = {}
    start_pid: dict[str, int] = {}  # half -> starting pitcher id

    def _slot(inning: int, half: str, pid: int) -> _Agg:
        key = (inning, half, pid)
        if key not in agg:
            pitch_team = home_abbr if half == "top" else away_abbr
            bat_team = away_abbr if half == "top" else home_abbr
            agg[key] = _Agg(pitch_team=pitch_team, bat_team=bat_team)
        return agg[key]

    for p in plays:
        about = p.get("about", {}) or {}
        inning = int(about.get("inning", 0) or 0)
        half = "top" if str(about.get("halfInning", "")).lower() == "top" else "bottom"
        if inning <= 0:
            continue
        matchup = p.get("matchup", {}) or {}
        pitcher = matchup.get("pitcher", {}) or {}
        pid = pitcher.get("id")
        if not pid:
            continue
        pid = int(pid)
        if inning == 1 and half not in start_pid:
            start_pid[half] = pid

        slot = _slot(inning, half, pid)
        slot.pitcher_name = pitcher.get("fullName", "") or slot.pitcher_name
        # Each play in allPlays is one completed plate appearance.
        slot.batters_faced += 1

        for runner in p.get("runners", []) or []:
            movement = runner.get("movement", {}) or {}
            if movement.get("end") != "score":
                continue
            details = runner.get("details", {}) or {}
            resp = details.get("responsiblePitcher") or {}
            rpid = int(resp.get("id", pid) or pid)
            # The run scores in this play's inning; credit the responsible pitcher.
            rslot = _slot(inning, half, rpid)
            rslot.runs_charged += 1
            if details.get("earned", True):
                rslot.earned_charged += 1

    lines: list[InningLine] = []
    for (inning, half, pid), a in agg.items():
        lines.append(
            InningLine(
                date=date,
                game_pk=game_pk,
                inning=inning,
                half=half,
                pitch_team=a.pitch_team,
                bat_team=a.bat_team,
                pitcher_id=pid,
                pitcher_name=a.pitcher_name,
                is_start=start_pid.get(half) == pid,
                batters_faced=a.batters_faced,
                runs_charged=a.runs_charged,
                earned_charged=a.earned_charged,
            )
        )
    lines.sort(key=lambda x: (x.inning, x.half, x.pitcher_id))
    return lines


def _boxscore(cache_dir: Path | None, game_pk: int) -> dict | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / "boxscores" / f"{game_pk}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("boxscore")
    except (OSError, json.JSONDecodeError):
        return None


def capture_game(
    game_pk: int,
    date: str,
    cache_dir: Path | None,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> list[InningLine]:
    """Fetch PBP for one final game and return its per-pitcher inning rows."""
    s = session or http.session()
    pbp = _fetch_pbp(game_pk, s, cache_dir, timeout)
    if not pbp:
        return []
    box = _boxscore(cache_dir, game_pk)
    home_abbr, away_abbr = _team_abbrevs(box) if box else ("", "")
    return game_inning_lines(pbp, game_pk, date, home_abbr, away_abbr)


def write_collapse_ledger(
    lines: list[InningLine],
    date: str,
    audit_dir: Path,
) -> tuple[Path, Path]:
    """Persist rows to a per-date JSON and append to the cumulative CSV ledger.

    The cumulative CSV is de-duplicated on (game_pk, pitcher_id, inning, half) so a
    re-run of the same slate replaces rather than double-counts its rows.
    """
    rows = [asdict(x) for x in lines]
    daily = audit_dir / f"collapse_{date}.json"
    daily.write_text(json.dumps(rows, indent=2))

    ledger = audit_dir / "collapse_ledger.csv"
    existing: list[dict[str, object]] = []
    if ledger.exists():
        with ledger.open(newline="") as fh:
            existing = [r for r in csv.DictReader(fh)]
    keys = {(str(r["game_pk"]), str(r["pitcher_id"]), str(r["inning"]), r["half"]) for r in rows}
    kept = [
        r
        for r in existing
        if (str(r["game_pk"]), str(r["pitcher_id"]), str(r["inning"]), r["half"]) not in keys
    ]
    with ledger.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in kept:
            writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
        for r in rows:
            writer.writerow({k: r[k] for k in FIELDNAMES})
    return daily, ledger


def capture_slate(
    game_pks: list[int],
    date: str,
    cache_dir: Path | None,
    audit_dir: Path,
    timeout: int = 20,
) -> list[InningLine]:
    """Capture every game on a slate and persist the collapse ledger."""
    session = http.session()
    lines: list[InningLine] = []
    for pk in game_pks:
        try:
            lines.extend(capture_game(pk, date, cache_dir, session, timeout))
        except Exception:  # noqa: BLE001 -- capture must never break the audit
            log.warning("collapse capture failed for %s", pk, exc_info=True)
    if lines:
        write_collapse_ledger(lines, date, audit_dir)
    return lines
