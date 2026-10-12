"""Bart Torvik T-Rank files, archived daily and read **as-of** a date.

``team_results`` (ratings), ``getgamestats`` (game logs) and ``super_sked``
(Torvik's own pre-game projections) are plain CSVs. Every fetch is stored under
``<data>/torvik/<season>/<file>/<YYYY-MM-DD>.csv``; ``as_of`` returns the last
file captured on or before a date, or ``None`` with its staleness, so the
model channel can switch itself off (``torvik_stale_days`` in preregistered.yaml)
while capture and the market-only channel continue.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import requests

from mlb_engine.data import http

log = logging.getLogger(__name__)
BASE = "https://barttorvik.com"
FILES: dict[str, str] = {
    "team_results": "{season}_team_results.csv",
    "super_sked": "{season}_super_sked.csv",
    "game_logs": "getgamestats.php?year={season}&csv=1",
}


def season_of(day: Date) -> int:
    """Torvik names a season by its spring year: Nov 2026 -> 2027."""
    return day.year + 1 if day.month >= 7 else day.year


@dataclass(frozen=True)
class AsOf:
    path: Path | None
    captured: Date | None
    stale_days: int | None

    def usable(self, max_stale_days: int) -> bool:
        return (
            self.path is not None
            and self.stale_days is not None
            and self.stale_days <= max_stale_days
        )


class TorvikClient:
    def __init__(self, timeout: int = 40) -> None:
        self._session = http.session(user_agent="cbb-engine/0.1", timeout=timeout)

    def fetch(self, name: str, season: int) -> str | None:
        url = f"{BASE}/{FILES[name].format(season=season)}"
        try:
            resp = self._session.get(url)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Torvik %s failed: %s", name, exc)
            return None
        text = resp.text
        if not text or text.lstrip().startswith("<"):
            log.warning("Torvik %s returned a page, not a CSV (browser check?)", name)
            return None
        return text

    def archive(self, root: Path, name: str, day: Date) -> Path | None:
        text = self.fetch(name, season_of(day))
        if text is None:
            return None
        path = file_path(root, name, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


def file_path(root: Path, name: str, day: Date) -> Path:
    return root / "torvik" / str(season_of(day)) / name / f"{day.isoformat()}.csv"


def as_of(root: Path, name: str, day: Date) -> AsOf:
    folder = root / "torvik" / str(season_of(day)) / name
    if not folder.exists():
        return AsOf(None, None, None)
    held = sorted(p for p in folder.glob("*.csv") if p.stem <= day.isoformat())
    if not held:
        return AsOf(None, None, None)
    captured = Date.fromisoformat(held[-1].stem)
    return AsOf(held[-1], captured, (day - captured).days)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Header-driven; ``team_results`` ships without a header, so one is supplied."""
    text = path.read_text()
    first = text.splitlines()[0] if text else ""
    if path.parent.name == "team_results" and not first.lower().startswith("rank"):
        return [
            dict(zip(TEAM_RESULTS_COLUMNS, row, strict=False))
            for row in csv.reader(io.StringIO(text))
        ]
    return list(csv.DictReader(io.StringIO(text)))


# Verified column order of <season>_team_results.csv (45 columns, no header row).
TEAM_RESULTS_COLUMNS: tuple[str, ...] = (
    "rank",
    "team",
    "conf",
    "record",
    "adjoe",
    "oe_rank",
    "adjde",
    "de_rank",
    "barthag",
    "proj_w",
    "proj_l",
    "pro_con_w",
    "pro_con_l",
    "con_rec",
    "sos",
    "ncsos",
    "consos",
    "proj_sos",
    "proj_ncsos",
    "proj_consos",
    "elite_sos",
    "elite_ncsos",
    "opp_oe",
    "opp_de",
    "opp_proj_oe",
    "opp_proj_de",
    "con_adj_oe",
    "con_adj_de",
    "qual_o",
    "qual_d",
    "qual_barthag",
    "qual_games",
    "fun",
    "conpf",
    "conpa",
    "conposs",
    "conoe",
    "conde",
    "consosremain",
    "conf_win_pct",
    "wab",
    "wab_rank",
    "fun_rank",
    "adjt",
    "adjt_rank",
)

__all__ = [
    "BASE",
    "FILES",
    "TEAM_RESULTS_COLUMNS",
    "AsOf",
    "TorvikClient",
    "as_of",
    "file_path",
    "read_csv",
    "season_of",
]
