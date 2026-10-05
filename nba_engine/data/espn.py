"""ESPN's public NBA scoreboard and box score: schedule, finals by quarter, player lines.

Grading reads finals from here (game and first-half markets settle on the
quarter scores, props on the box line). A player listed ``didNotPlay`` -- or
with no minutes -- is a DNP, which voids his props rather than losing them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date as Date

import requests

from mlb_engine.data import http
from nba_engine.data import teamnames
from nba_engine.data.oddsapi import slate_date_of
from nba_engine.schemas import GameResult, PlayerLine, TeamBox

log = logging.getLogger(__name__)

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


class ESPNClient:
    def __init__(self, timeout: int = 25) -> None:
        self._session = http.session(user_agent="nba-engine/0.1", timeout=timeout)

    def _get(self, url: str, **params: str) -> dict | None:
        try:
            resp = self._session.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("ESPN request failed (%s): %s", url.rsplit("/", 1)[-1], exc)
            return None
        return payload if isinstance(payload, dict) else None

    def scoreboard(self, day: Date) -> list[GameResult]:
        payload = self._get(f"{SITE}/scoreboard", dates=day.strftime("%Y%m%d"))
        return parse_scoreboard(payload or {}, day)

    def results(
        self, day: Date, on_summary: Callable[[str, dict], None] | None = None
    ) -> list[GameResult]:
        """The day's games, with box lines, team totals and venue attached for every final.

        ``on_summary`` receives each raw summary (it carries the play-by-play),
        so a caller can keep it without a second request.
        """
        out: list[GameResult] = []
        for game in self.scoreboard(day):
            if game.is_final:
                summary = self._get(f"{SITE}/summary", event=game.espn_id)
                if summary is not None:
                    game = with_summary(game, summary)
                    if on_summary is not None:
                        on_summary(game.espn_id, summary)
            out.append(game)
        return out


def _code(team: dict) -> str:
    return teamnames.canonical(str(team.get("abbreviation", "")))


def parse_scoreboard(payload: dict, day: Date) -> list[GameResult]:
    out: list[GameResult] = []
    for event in payload.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        sides = {c.get("homeAway"): c for c in comps[0].get("competitors", [])}
        home, away = sides.get("home"), sides.get("away")
        if home is None or away is None:
            continue

        def quarters(side: dict) -> tuple[int, ...]:
            return tuple(int(float(q.get("value", 0))) for q in side.get("linescores", []))

        out.append(
            GameResult(
                espn_id=str(event.get("id", "")),
                game_date=slate_date_of(str(event.get("date", ""))) or day,
                away=_code(away.get("team", {})),
                home=_code(home.get("team", {})),
                state=str(event.get("status", {}).get("type", {}).get("name", "")),
                away_q=quarters(away),
                home_q=quarters(home),
            )
        )
    return out


def _int(raw: str) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return 0


def parse_box(summary: dict) -> tuple[PlayerLine, ...]:
    """Every listed player's line, DNPs included (flagged)."""
    out: list[PlayerLine] = []
    for team in summary.get("boxscore", {}).get("players", []):
        code = _code(team.get("team", {}))
        for block in team.get("statistics", []):
            keys = list(block.get("keys", []))
            for ath in block.get("athletes", []):
                stats = dict(zip(keys, ath.get("stats", []), strict=False))
                minutes = _int(stats.get("minutes", "0"))
                threes = str(
                    stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "0-0")
                )
                person = ath.get("athlete", {})
                out.append(
                    PlayerLine(
                        espn_id=str(person.get("id", "")),
                        team=code,
                        name=str(person.get("displayName", "")),
                        minutes=minutes,
                        points=_int(stats.get("points", "0")),
                        rebounds=_int(stats.get("rebounds", "0")),
                        assists=_int(stats.get("assists", "0")),
                        threes=_int(threes.split("-")[0]),
                        starter=bool(ath.get("starter")),
                        dnp=bool(ath.get("didNotPlay")) or not stats or minutes == 0,
                    )
                )
    return tuple(out)


def _made_att(raw: str) -> tuple[int, int]:
    made, _, att = str(raw).partition("-")
    return _int(made), _int(att)


def parse_team_boxes(summary: dict) -> dict[str, TeamBox]:
    """Team code -> box totals; a team missing any shooting line is left out."""
    out: dict[str, TeamBox] = {}
    for team in summary.get("boxscore", {}).get("teams", []):
        code = _code(team.get("team", {}))
        stats = {s.get("name"): s.get("displayValue", "") for s in team.get("statistics", [])}
        need = (
            "fieldGoalsMade-fieldGoalsAttempted",
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
            "freeThrowsMade-freeThrowsAttempted",
            "offensiveRebounds",
        )
        if not code or any(k not in stats for k in need):
            continue
        fgm, fga = _made_att(stats[need[0]])
        fg3m, fg3a = _made_att(stats[need[1]])
        ftm, fta = _made_att(stats[need[2]])
        out[code] = TeamBox(
            team=code,
            fgm=fgm,
            fga=fga,
            fg3m=fg3m,
            fg3a=fg3a,
            ftm=ftm,
            fta=fta,
            oreb=_int(stats["offensiveRebounds"]),
            dreb=_int(stats.get("defensiveRebounds", "0")),
            tov=_int(stats.get("totalTurnovers", stats.get("turnovers", "0"))),
            fouls=_int(stats.get("fouls", "0")),
        )
    return out


def with_summary(game: GameResult, summary: dict) -> GameResult:
    """The scoreboard final with the summary's box lines, team totals and venue."""
    teams = parse_team_boxes(summary)
    header = summary.get("header", {})
    comp = (header.get("competitions") or [{}])[0]
    venue = summary.get("gameInfo", {}).get("venue", {})
    return GameResult(
        **{
            **game.__dict__,
            "players": parse_box(summary),
            "away_box": teams.get(game.away),
            "home_box": teams.get(game.home),
            "neutral": bool(comp.get("neutralSite")),
            "venue": str(venue.get("fullName", "")),
            "city": str(venue.get("address", {}).get("city", "")),
            "season_type": _int(header.get("season", {}).get("type", 0)),
        }
    )


__all__ = ["ESPNClient", "parse_box", "parse_scoreboard", "parse_team_boxes", "with_summary"]
