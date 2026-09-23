"""Cache FanGraphs date-range leaderboards as of every game day, look-ahead free.

Production's ``bullpen_ranking`` / ``starter_table`` read a *season* table with
``month=0`` and windowed tables with ``month=1000&startdate..enddate``. Replayed
historically, ``month=0`` would return the finished season, so every table here
is a date range ending the day *before* the game day: the season table from
opening day, the windowed ones over ``window`` days.

Per game day: relievers by team (season, 30d, 60d) and starters (season, 42d)
-- five requests. Responses are cached under
``~/.mlb_engine/cache/fg_hist/`` so a re-run is free.

    .venv/bin/python scripts/oos_fangraphs_pull.py --seasons 2024 2025 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import requests

from mlb_engine.output.daily_worksheet import BP_COLS, SP_SKILL_DAYS
from mlb_engine.output.totals_sheet import _FG_HEADERS, _FG_URL

log = logging.getLogger("oos_fangraphs_pull")

DEFAULT_CACHE = Path.home() / ".mlb_engine" / "cache" / "fg_hist"
OPENING_DAY = {2024: Date(2024, 3, 20), 2025: Date(2025, 3, 18), 2026: Date(2026, 3, 26)}
LAST_DAY = {2024: Date(2024, 9, 29), 2025: Date(2025, 9, 28), 2026: Date(2026, 9, 22)}
REL_WINDOWS: tuple[int | None, ...] = tuple(sorted({c[5] for c in BP_COLS}, key=lambda w: w or 0))
STA_WINDOWS: tuple[int | None, ...] = (None, SP_SKILL_DAYS)


def table_range(day: Date, window: int | None, season: int) -> tuple[Date, Date]:
    """Inclusive (start, end) of the leaderboard a game on ``day`` may read."""
    end = day - timedelta(days=1)
    start = OPENING_DAY[season] if window is None else day - timedelta(days=window)
    return max(start, OPENING_DAY[season]), end


def cache_path(cache_dir: Path, stats: str, start: Date, end: Date) -> Path:
    return cache_dir / f"{stats}_{start.isoformat()}_{end.isoformat()}.json"


def fetch_table(stats: str, season: int, start: Date, end: Date, cache_dir: Path,
                session: requests.Session, pause: float = 1.0) -> list[dict]:
    path = cache_path(cache_dir, stats, start, end)
    if path.exists():
        return json.loads(path.read_text())
    team = "0%2Cts" if stats == "rel" else "0"
    url = _FG_URL.format(stats=stats, season=season, month=1000, team=team)
    url += f"&startdate={start.isoformat()}&enddate={end.isoformat()}"
    for attempt in range(5):
        try:
            resp = session.get(url, headers=_FG_HEADERS, timeout=90)
            if resp.status_code == 200:
                data = resp.json().get("data")
                if isinstance(data, list):
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                    time.sleep(pause)
                    return data
            log.warning("%s %s..%s HTTP %s", stats, start, end, resp.status_code)
            if resp.status_code == 429:
                time.sleep(60 * (attempt + 1))
        except (requests.RequestException, ValueError) as exc:
            log.warning("%s %s..%s %s", stats, start, end, type(exc).__name__)
        time.sleep(3 * (attempt + 1))
    return []


def load_day(day: Date, cache_dir: Path = DEFAULT_CACHE, session: requests.Session | None = None,
             fetch: bool = True) -> dict[str, dict[int | None, list[dict]]]:
    """{'rel': {window: rows}, 'sta': {window: rows}} for a game day."""
    sess = session or requests.Session()
    out: dict[str, dict[int | None, list[dict]]] = {"rel": {}, "sta": {}}
    for stats, windows in (("rel", REL_WINDOWS), ("sta", STA_WINDOWS)):
        for w in windows:
            start, end = table_range(day, w, day.year)
            if end < start:
                out[stats][w] = []
                continue
            if fetch:
                out[stats][w] = fetch_table(stats, day.year, start, end, cache_dir, sess)
            else:
                p = cache_path(cache_dir, stats, start, end)
                out[stats][w] = json.loads(p.read_text()) if p.exists() else []
    return out


def game_days(season: int) -> list[Date]:
    params: dict[str, str | int] = {
        "sportId": 1, "gameType": "R", "startDate": OPENING_DAY[season].isoformat(),
        "endDate": LAST_DAY[season].isoformat()}
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params=params, timeout=60)
    r.raise_for_status()
    return [Date.fromisoformat(d["date"]) for d in r.json().get("dates", []) if d.get("games")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=[2024, 2025, 2026])
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sess = requests.Session()
    for s in args.seasons:
        days = game_days(s)
        for i, d in enumerate(days):
            load_day(d, args.cache_dir, sess)
            if i % 20 == 0:
                log.info("%s: %s/%s days", s, i + 1, len(days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
