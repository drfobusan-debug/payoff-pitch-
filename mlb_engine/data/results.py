"""Fetch final game results and player box-score lines for the nightly audit."""

from __future__ import annotations

from dataclasses import dataclass, field

import requests

BASE = "https://statsapi.mlb.com/api/v1"


@dataclass
class PlayerLine:
    batting: dict[str, int] = field(default_factory=dict)  # H,2B,3B,HR,RBI,R,...
    pitching: dict[str, int] = field(default_factory=dict)  # K,outs,H,BB,ER,...


@dataclass
class GameResult:
    game_pk: int
    final: bool
    home_runs: int
    away_runs: int
    f5_home: int
    f5_away: int
    players: dict[int, PlayerLine] = field(default_factory=dict)  # mlbam_id -> line

    def batter(self, pid: int) -> dict[str, int]:
        return self.players.get(pid, PlayerLine()).batting

    def pitcher(self, pid: int) -> dict[str, int]:
        return self.players.get(pid, PlayerLine()).pitching


def _ip_to_outs(ip: object) -> int:
    """Convert innings-pitched string (e.g. '5.2') to outs."""
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except (ValueError, TypeError):
        return 0


def fetch_result(game_pk: int, session: requests.Session | None = None, timeout: int = 20) -> GameResult:
    s = session or requests.Session()
    ls = s.get(f"{BASE}/game/{game_pk}/linescore", timeout=timeout).json()
    bs = s.get(f"{BASE}/game/{game_pk}/boxscore", timeout=timeout).json()

    innings = ls.get("innings", [])
    home_runs = sum(int((i.get("home", {}) or {}).get("runs", 0) or 0) for i in innings)
    away_runs = sum(int((i.get("away", {}) or {}).get("runs", 0) or 0) for i in innings)
    f5_home = sum(
        int((i.get("home", {}) or {}).get("runs", 0) or 0) for i in innings if i.get("num", 99) <= 5
    )
    f5_away = sum(
        int((i.get("away", {}) or {}).get("runs", 0) or 0) for i in innings if i.get("num", 99) <= 5
    )
    final = str(ls.get("gameStatus", {}).get("abstractGameState", "")).lower() == "final" or bool(
        innings and len(innings) >= 9
    )

    players: dict[int, PlayerLine] = {}
    for side in ("home", "away"):
        team = bs.get("teams", {}).get(side, {})
        for _key, pdata in team.get("players", {}).items():
            pid = (pdata.get("person", {}) or {}).get("id")
            if not pid:
                continue
            stats = pdata.get("stats", {}) or {}
            bat = stats.get("batting", {}) or {}
            pit = stats.get("pitching", {}) or {}
            line = PlayerLine()
            if bat:
                hits = int(bat.get("hits", 0) or 0)
                doubles = int(bat.get("doubles", 0) or 0)
                triples = int(bat.get("triples", 0) or 0)
                hr = int(bat.get("homeRuns", 0) or 0)
                line.batting = {
                    "H": hits,
                    "1B": hits - doubles - triples - hr,
                    "2B": doubles,
                    "3B": triples,
                    "HR": hr,
                    "RBI": int(bat.get("rbi", 0) or 0),
                    "R": int(bat.get("runs", 0) or 0),
                }
            if pit:
                line.pitching = {
                    "K": int(pit.get("strikeOuts", 0) or 0),
                    "outs": _ip_to_outs(pit.get("inningsPitched", "0.0")),
                    "H": int(pit.get("hits", 0) or 0),
                    "BB": int(pit.get("baseOnBalls", 0) or 0),
                    "ER": int(pit.get("earnedRuns", 0) or 0),
                }
            if line.batting or line.pitching:
                players[pid] = line

    return GameResult(
        game_pk=game_pk,
        final=final,
        home_runs=home_runs,
        away_runs=away_runs,
        f5_home=f5_home,
        f5_away=f5_away,
        players=players,
    )
