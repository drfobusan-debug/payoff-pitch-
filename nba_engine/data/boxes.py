"""A local archive of ESPN finals and box lines, one JSON file per slate date.

Grading and every fit read finals from here, not from ESPN again: a season is
about 240 days of one scoreboard and ~5 summaries each, and the files never
change once a day is final. ``<data>/results/<YYYY-MM-DD>.json`` holds a list of
``GameResult`` records with their player lines.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from dataclasses import asdict
from datetime import date as Date
from pathlib import Path

from nba_engine.data.espn import ESPNClient
from nba_engine.schemas import GameResult, PlayerLine, TeamBox

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b")


def results_path(data_dir: Path, day: Date) -> Path:
    return data_dir / "results" / f"{day.isoformat()}.json"


def raw_summary_path(data_dir: Path, day: Date, espn_id: str) -> Path:
    return data_dir / "espn_raw" / day.isoformat() / f"{espn_id}.json.gz"


def write_raw_summary(data_dir: Path, day: Date, espn_id: str, summary: dict) -> Path:
    """Keep ESPN's whole summary (play-by-play included) for the stint work."""
    path = raw_summary_path(data_dir, day, espn_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(summary).encode()))
    tmp.replace(path)
    return path


def read_raw_summary(data_dir: Path, day: Date, espn_id: str) -> dict | None:
    path = raw_summary_path(data_dir, day, espn_id)
    if not path.exists():
        return None
    payload = json.loads(gzip.decompress(path.read_bytes()))
    return payload if isinstance(payload, dict) else None


def norm_name(name: str) -> str:
    """Book and ESPN spellings of one player to one key (accents, suffixes, punctuation)."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = _SUFFIX.sub("", re.sub(r"[^a-z ]", "", s.replace("-", " ")))
    return " ".join(s.split())


def _to_json(game: GameResult) -> dict:
    out = asdict(game)
    out["game_date"] = game.game_date.isoformat()
    return out


def _from_json(raw: dict) -> GameResult | None:
    try:
        return GameResult(
            espn_id=str(raw["espn_id"]),
            game_date=Date.fromisoformat(str(raw["game_date"])),
            away=str(raw["away"]),
            home=str(raw["home"]),
            state=str(raw["state"]),
            away_q=tuple(int(q) for q in raw.get("away_q", ())),
            home_q=tuple(int(q) for q in raw.get("home_q", ())),
            players=tuple(PlayerLine(**p) for p in raw.get("players", ())),
            away_box=TeamBox(**raw["away_box"]) if raw.get("away_box") else None,
            home_box=TeamBox(**raw["home_box"]) if raw.get("home_box") else None,
            neutral=bool(raw.get("neutral", False)),
            venue=str(raw.get("venue", "")),
            city=str(raw.get("city", "")),
            season_type=int(raw.get("season_type", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_results(data_dir: Path, day: Date) -> list[GameResult] | None:
    """The archived day, or ``None`` if it was never archived."""
    path = results_path(data_dir, day)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return None
    games = [_from_json(g) for g in payload if isinstance(g, dict)]
    return [g for g in games if g is not None]


def write_results(data_dir: Path, day: Date, games: list[GameResult]) -> Path:
    path = results_path(data_dir, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([_to_json(g) for g in games]))
    tmp.replace(path)
    return path


def ensure_results(data_dir: Path, day: Date, client: ESPNClient | None = None) -> list[GameResult]:
    """The day's results, fetched once and kept only when every game on it is final.

    A final whose box summary failed comes back with no player lines or team
    totals; the day is then returned but not archived, so the next run fetches
    it again. Each summary that did arrive is kept whole under ``espn_raw/``
    (it carries the play-by-play), whether or not the day is complete.
    """
    held = read_results(data_dir, day)
    if held is not None:
        return held
    games = (client or ESPNClient()).results(
        day, on_summary=lambda espn_id, summary: write_raw_summary(data_dir, day, espn_id, summary)
    )
    if games and all(g.is_final and g.players and g.home_box and g.away_box for g in games):
        write_results(data_dir, day, games)
    return games


def by_matchup(games: list[GameResult]) -> dict[tuple[str, str], GameResult]:
    """(slate date, ``AWAY @ HOME``) -> final, the key every archived quote carries."""
    return {(g.game_date.isoformat(), f"{g.away} @ {g.home}"): g for g in games if g.is_final}


__all__ = [
    "by_matchup",
    "ensure_results",
    "norm_name",
    "read_raw_summary",
    "read_results",
    "raw_summary_path",
    "results_path",
    "write_raw_summary",
    "write_results",
]
