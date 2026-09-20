"""ESPN public scoreboard: live score, clock, period and possession per game."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests

from live_edge.roller import LiveState

log = logging.getLogger(__name__)

SUMMARY = {
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary",
    "cfb": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary",
}
SCOREBOARD = {
    "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "cfb": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=300",
}


def norm_team(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass(frozen=True)
class LiveGame:
    event_id: str
    home: str
    away: str
    status: str  # "pre" | "in" | "post"
    state: LiveState
    espn_home_wp: float | None = None
    kickoff: str = ""
    home_aliases: tuple[str, ...] = ()
    away_aliases: tuple[str, ...] = ()

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"


def _aliases(team: dict) -> tuple[str, ...]:
    out = []
    for k in ("displayName", "shortDisplayName", "location", "name", "abbreviation"):
        v = team.get(k)
        if v:
            out.append(norm_team(str(v)))
    loc, nick = team.get("location"), team.get("name")
    if loc and nick:
        out.append(norm_team(f"{loc} {nick}"))
    return tuple(dict.fromkeys(out))


def _clock_secs(comp: dict) -> int:
    status = comp.get("status", {})
    clock = status.get("clock")
    if isinstance(clock, (int, float)):
        return int(clock)
    m = re.match(r"(\d+):(\d+)", str(status.get("displayClock", "")))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def parse_scoreboard(payload: dict) -> list[LiveGame]:
    games: list[LiveGame] = []
    for ev in payload.get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        home, away = sides.get("home"), sides.get("away")
        if not home or not away:
            continue
        status = comp.get("status", {}).get("type", {})
        state_word = str(status.get("state", "pre"))
        period = int(comp.get("status", {}).get("period") or 0)
        situation = comp.get("situation") or {}
        poss_id = situation.get("possession")
        poss_home: bool | None = None
        if poss_id is not None:
            poss_home = str(poss_id) == str(home.get("id"))
        # ESPN's yardLine is distance from the possessing side's own goal line.
        yl = situation.get("yardLine")
        ytg = 100 - int(yl) if isinstance(yl, (int, float)) and 0 <= yl <= 100 else None
        wp = None
        prob = (situation.get("lastPlay") or {}).get("probability") or {}
        if "homeWinPercentage" in prob:
            wp = float(prob["homeWinPercentage"])
        games.append(
            LiveGame(
                event_id=str(ev.get("id")),
                home=str(home["team"].get("displayName", "")),
                away=str(away["team"].get("displayName", "")),
                status=state_word,
                state=LiveState(
                    home_score=int(home.get("score") or 0),
                    away_score=int(away.get("score") or 0),
                    period=max(period, 1) if state_word != "pre" else 1,
                    clock_secs=_clock_secs(comp)
                    if state_word == "in"
                    else (900 if state_word == "pre" else 0),
                    possession_home=poss_home if state_word == "in" else None,
                    yards_to_goal=ytg if state_word == "in" else None,
                ),
                espn_home_wp=wp,
                kickoff=str(ev.get("date", "")),
                home_aliases=_aliases(home["team"]),
                away_aliases=_aliases(away["team"]),
            )
        )
    return games


def parse_pregame_line(summary: dict) -> tuple[float, float] | None:
    """(home spread, total) as the median over ESPN's pickcenter providers."""
    spreads, totals = [], []
    for row in summary.get("pickcenter") or []:
        sp, ou = row.get("spread"), row.get("overUnder")
        if isinstance(sp, (int, float)) and isinstance(ou, (int, float)):
            spreads.append(float(sp))
            totals.append(float(ou))
    if not spreads:
        return None
    spreads.sort()
    totals.sort()
    mid = len(spreads) // 2
    return spreads[mid], totals[mid]


def fetch_pregame_line(
    sport: str, event_id: str, timeout: float = 15.0
) -> tuple[float, float] | None:
    """Pregame consensus for a game the tape did not see before kickoff."""
    try:
        resp = requests.get(SUMMARY[sport], params={"event": event_id}, timeout=timeout)
        resp.raise_for_status()
        return parse_pregame_line(resp.json())
    except (requests.RequestException, ValueError) as exc:
        log.warning("ESPN summary failed (%s %s): %s", sport, event_id, exc)
        return None


def fetch_scoreboard(sport: str, timeout: float = 15.0) -> list[LiveGame]:
    try:
        resp = requests.get(SCOREBOARD[sport], timeout=timeout)
        resp.raise_for_status()
        return parse_scoreboard(resp.json())
    except (requests.RequestException, ValueError) as exc:
        log.warning("ESPN scoreboard failed (%s): %s", sport, exc)
        return []
