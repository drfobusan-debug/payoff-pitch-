"""Fetch final game results and player box-score lines for the nightly audit."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

from mlb_engine.data import http

BASE = "https://statsapi.mlb.com/api/v1"

log = logging.getLogger(__name__)


@dataclass
class PlayerLine:
    batting: dict[str, int] = field(default_factory=dict)  # H,2B,3B,HR,RBI,R,...
    pitching: dict[str, int] = field(default_factory=dict)  # K,outs,H,BB,ER,...
    #: The box score's ``battingOrder``: slot times 100, plus one per substitute
    #: into that slot (``400`` started fourth, ``401`` came in for him).
    order: int | None = None

    @property
    def started(self) -> bool:
        return self.order is not None and self.order % 100 == 0

    @property
    def slot(self) -> int | None:
        return None if self.order is None else self.order // 100


@dataclass
class GameResult:
    game_pk: int
    final: bool
    home_runs: int
    away_runs: int
    f5_home: int
    f5_away: int
    players: dict[int, PlayerLine] = field(default_factory=dict)  # mlbam_id -> line
    #: The day the game was actually played (StatsAPI ``officialDate``), and its
    #: scheduled first pitch in UTC. A rainout keeps its ``game_pk`` and is played
    #: on another day, so a row priced for the original date is not a bet on it.
    #: Blank when the schedule could not be read; nothing is voided on a blank.
    official_date: str = ""
    start_utc: str = ""

    def batter(self, pid: int) -> dict[str, int]:
        return self.players.get(pid, PlayerLine()).batting

    def pitcher(self, pid: int) -> dict[str, int]:
        return self.players.get(pid, PlayerLine()).pitching

    def batted(self, pid: int) -> bool:
        """Did this player actually take a plate appearance?

        A scratched hitter is simply absent from the box score, so every counting
        stat reads zero and every over on him looks like a loss. That is not what
        a book does -- the prop is voided -- and it is not what the audit should
        do either, because those phantom losses are indistinguishable from the
        engine picking badly.
        """
        bat = self.batter(pid)
        # A missing PA key means the line predates this field, and a batting line
        # only exists at all for a player the box score lists as having batted --
        # so absence of the counter is not evidence of absence from the game.
        return bool(bat) and bat.get("PA", 1) > 0

    def pitched(self, pid: int) -> bool:
        """Did this player actually face a batter?"""
        pit = self.pitcher(pid)
        return bool(pit) and (pit.get("BF", 1) > 0 or pit.get("outs", 0) > 0)


def _ip_to_outs(ip: object) -> int:
    """Convert innings-pitched string (e.g. '5.2') to outs."""
    try:
        whole, _, frac = str(ip).partition(".")
        return int(whole) * 3 + (int(frac) if frac else 0)
    except (ValueError, TypeError):
        return 0


def _cache_path(cache_dir: Path | None, game_pk: int) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / "boxscores" / f"{game_pk}.json"


def _save_cache(cache_path: Path | None, ls: dict, bs: dict, sched: dict | None = None) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"linescore": ls, "boxscore": bs}
    if sched:
        payload["schedule"] = sched
    try:
        cache_path.write_text(json.dumps(payload, indent=2))
    except OSError:
        log.warning("failed to cache box score for %s", cache_path, exc_info=True)


def _load_cache(cache_path: Path | None) -> tuple[dict | None, dict | None]:
    if cache_path is None or not cache_path.exists():
        return None, None
    try:
        data = json.loads(cache_path.read_text())
        return data.get("linescore"), data.get("boxscore")
    except (OSError, json.JSONDecodeError):
        log.warning("failed to load cached box score for %s", cache_path, exc_info=True)
        return None, None


def _load_cached_schedule(cache_path: Path | None) -> dict:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        return dict(json.loads(cache_path.read_text()).get("schedule") or {})
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def played_entry(schedule: dict) -> dict:
    """The schedule entry for the game as played, not the postponed original.

    A rained-out game keeps its ``game_pk``: the schedule lists it twice, once
    ``Postponed`` on the original date and once on the day it was made up.
    """
    games = [g for d in schedule.get("dates", []) or [] for g in d.get("games", []) or []]
    if not games:
        return {}
    final = [g for g in games if (g.get("status") or {}).get("abstractGameState") == "Final"]
    live = [g for g in games if (g.get("status") or {}).get("detailedState") != "Postponed"]
    return (final or live or games)[-1]


def _fetch_schedule(s: requests.Session, game_pk: int, timeout: int) -> dict:
    try:
        data = s.get(
            f"{BASE}/schedule", params={"sportId": 1, "gamePk": game_pk}, timeout=timeout
        ).json()
    except Exception as exc:  # noqa: BLE001 - a missing schedule voids nothing; the box still grades
        log.warning("MLB Stats API schedule fetch failed for %s: %s", game_pk, exc)
        return {}
    return data if isinstance(data, dict) else {}


def fetch_result(
    game_pk: int,
    session: requests.Session | None = None,
    cache_dir: Path | None = None,
    timeout: int = 20,
) -> GameResult:
    """Fetch a final box score, with a local-cache fallback if the API is unavailable.

    On a successful fetch, the response is written to ``cache_dir/boxscores/{game_pk}.json``
    so a later audit can still grade the game even if the network is down.
    """
    s = session or http.session()
    cache_path = _cache_path(cache_dir, game_pk)
    try:
        ls = s.get(f"{BASE}/game/{game_pk}/linescore", timeout=timeout).json()
        bs = s.get(f"{BASE}/game/{game_pk}/boxscore", timeout=timeout).json()
        sched = _fetch_schedule(s, game_pk, timeout) or _load_cached_schedule(cache_path)
        _save_cache(cache_path, ls, bs, sched)
    except requests.RequestException as exc:
        log.warning("MLB Stats API box score fetch failed for %s: %s", game_pk, exc)
        ls, bs = _load_cache(cache_path)
        if ls is None or bs is None:
            raise RuntimeError(f"No box score available for game_pk={game_pk}") from exc
        sched = _load_cached_schedule(cache_path)
    played = played_entry(sched)

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
            order = str(pdata.get("battingOrder", "") or "")
            if order.isdigit():
                line.order = int(order)
            if bat:
                hits = int(bat.get("hits", 0) or 0)
                doubles = int(bat.get("doubles", 0) or 0)
                triples = int(bat.get("triples", 0) or 0)
                hr = int(bat.get("homeRuns", 0) or 0)
                pa = int(bat.get("plateAppearances", 0) or 0)
                if pa == 0:
                    # Older cached box scores predate this field. Rebuild it rather
                    # than let a missing key void a hitter who demonstrably batted.
                    pa = max(
                        int(bat.get("atBats", 0) or 0)
                        + int(bat.get("baseOnBalls", 0) or 0)
                        + int(bat.get("hitByPitch", 0) or 0)
                        + int(bat.get("sacFlies", 0) or 0)
                        + int(bat.get("sacBunts", 0) or 0),
                        hits,
                    )
                line.batting = {
                    # Plate appearances are not a market, but they are what says
                    # the hitter played at all -- see GameResult.batted.
                    "PA": pa,
                    "H": hits,
                    "1B": hits - doubles - triples - hr,
                    "2B": doubles,
                    "3B": triples,
                    "HR": hr,
                    "RBI": int(bat.get("rbi", 0) or 0),
                    "R": int(bat.get("runs", 0) or 0),
                    # The batter's own walks and strikeouts, which are markets in
                    # their own right: without them here every such prop grades
                    # off a zero and its overs read as losses.
                    "BB": int(bat.get("baseOnBalls", 0) or 0),
                    "K": int(bat.get("strikeOuts", 0) or 0),
                }
            if pit:
                line.pitching = {
                    "BF": int(pit.get("battersFaced", 0) or 0),
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
        official_date=str(played.get("officialDate") or ""),
        start_utc=str(played.get("gameDate") or ""),
    )
