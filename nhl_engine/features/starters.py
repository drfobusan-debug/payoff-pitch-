"""Tonight's goalies: who, how sure, and his GSAx/60 (master plan §5.2, §5.5).

Status ladder, from the source that named him:

* ``confirmed`` -- a starter announced by the team / NHL API game-day roster.
* ``probable`` -- a beat-writer or lines-site expectation.
* ``projected`` -- nobody has said anything; the engine names the goalie who has
  carried the most ice time for the team this season (last season on opening
  night) and stamps the game ``goalie_unconfirmed``. ML, puck line, team-total
  and period ML/PL buys are refused under that stamp; totals are priced on the
  projected pair and say so.

Overrides live in ``~/.nhl_engine/starters/<date>.json`` as
``{"BOS": {"player_id": 8471695, "status": "confirmed", "source": "..."}}`` and
are written by hand (``nhl-engine starter``), by the RotoWire feed
(``nhl-engine lineups``) or by the pre-drop roster pass. The boxscore starter is
the roster of record for grading (``GameResult.home_starter``), never this file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

import pandas as pd

from nhl_engine.config import Config
from nhl_engine.data.moneypuck import MoneyPuckClient
from nhl_engine.data.teamnames import canonical
from nhl_engine.features.goalie import GoalieSkill, goalie_skill

log = logging.getLogger("nhl_engine")

SURE = frozenset({"confirmed", "probable"})
GATE = "goalie_unconfirmed"


@dataclass(frozen=True)
class Starter:
    team: str
    player_id: int
    name: str
    status: str  # confirmed | probable | projected | unknown
    source: str
    skill: GoalieSkill | None

    @property
    def gsax60(self) -> float:
        return self.skill.gsax60 if self.skill is not None else 0.0

    @property
    def confirmed(self) -> bool:
        return self.status in SURE

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "team": self.team,
            "player_id": self.player_id,
            "name": self.name,
            "status": self.status,
            "source": self.source,
        }
        if self.skill is not None:
            out["skill"] = asdict(self.skill)
        return out


def overrides_path(data_dir: Path, slate: Date) -> Path:
    return data_dir / "starters" / f"{slate.isoformat()}.json"


def load_overrides(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {canonical(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def save_override(
    path: Path, team: str, player_id: int, status: str, source: str, *, name: str = ""
) -> None:
    current = load_overrides(path)
    entry: dict[str, object] = {"player_id": int(player_id), "status": status, "source": source}
    if name:
        entry["name"] = name
    current[canonical(team)] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _workhorse(mp: MoneyPuckClient, team: str, season: int) -> tuple[int, str] | None:
    """Most ice time for ``team`` in ``season`` (falls back one season)."""
    for s in (season, season - 1):
        table = mp.season_goalies(s)
        if table.empty or "situation" not in table:
            continue
        rows = table[(table["situation"] == "all") & (table["team"].map(canonical) == team)]
        if rows.empty:
            continue
        best = rows.sort_values("icetime", ascending=False).iloc[0]
        return int(best["playerId"]), str(best["name"])
    return None


def _name_for(mp: MoneyPuckClient, player_id: int, season: int) -> str:
    for s in (season, season - 1):
        table = mp.season_goalies(s)
        if not table.empty and "playerId" in table:
            hit = table[table["playerId"] == player_id]
            if not hit.empty:
                return str(hit["name"].iloc[0])
    return str(player_id)


def _skill(
    mp: MoneyPuckClient, player_id: int, slate: Date, season: int, cfg: Config
) -> GoalieSkill | None:
    games: pd.DataFrame = mp.goalie_games(player_id)
    if games.empty:
        return None
    return goalie_skill(
        games,
        slate,
        season=season,
        league_gsax60=0.0,
        k_season=cfg.shrink.goalie_k_season,
        k_career=cfg.shrink.goalie_k_career,
        r_yy=cfg.shrink.goalie_r_yy,
        min_minutes=cfg.goalie.min_minutes,
        callup_prior=cfg.goalie.callup_prior_gsax60,
    )


def starter_for(
    mp: MoneyPuckClient,
    team: str,
    slate: Date,
    *,
    season: int,
    cfg: Config,
    overrides: dict[str, dict[str, object]],
) -> Starter:
    team = canonical(team)
    ov = overrides.get(team)
    if ov and "player_id" in ov:
        pid = int(str(ov["player_id"]))
        status = str(ov.get("status", "probable"))
        return Starter(
            team,
            pid,
            str(ov.get("name") or _name_for(mp, pid, season)),
            status if status in SURE or status == "projected" else "probable",
            str(ov.get("source", "override")),
            _skill(mp, pid, slate, season, cfg),
        )
    work = _workhorse(mp, team, season)
    if work is None:
        return Starter(team, 0, "", "unknown", "no goalie history", None)
    pid, name = work
    return Starter(
        team, pid, name, "projected", "season ice-time leader", _skill(mp, pid, slate, season, cfg)
    )


__all__ = [
    "GATE",
    "Starter",
    "load_overrides",
    "overrides_path",
    "save_override",
    "starter_for",
]
