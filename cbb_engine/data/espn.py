"""ESPN men's college basketball scoreboard: schedule, tips, neutral site, finals, half scores.

Free; the grading feed. Raw payloads are archived per fetch so an as-of replay
reads the original, not today's ESPN.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import requests

from mlb_engine.data import http

log = logging.getLogger(__name__)
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
D1_GROUP = "50"


@dataclass(frozen=True)
class Game:
    espn_id: str
    start_utc: str
    away: str
    home: str
    state: str  # STATUS_SCHEDULED | STATUS_IN_PROGRESS | STATUS_FINAL ...
    neutral: bool
    conference_game: bool
    away_halves: tuple[int, ...] = ()
    home_halves: tuple[int, ...] = ()

    @property
    def is_final(self) -> bool:
        return self.state == "STATUS_FINAL"

    @property
    def final(self) -> tuple[int, int] | None:
        if not self.is_final or not self.away_halves:
            return None
        return sum(self.away_halves), sum(self.home_halves)


class ESPNClient:
    def __init__(self, timeout: int = 25) -> None:
        self._session = http.session(user_agent="cbb-engine/0.1", timeout=timeout)

    def _get(self, url: str, **params: str) -> dict[str, object] | None:
        try:
            resp = self._session.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("ESPN request failed (%s): %s", url.rsplit("/", 1)[-1], exc)
            return None
        return payload if isinstance(payload, dict) else None

    def scoreboard_raw(self, day: Date) -> dict[str, object] | None:
        return self._get(
            f"{SITE}/scoreboard", dates=day.strftime("%Y%m%d"), groups=D1_GROUP, limit="400"
        )

    def scoreboard(self, day: Date, archive: Path | None = None) -> list[Game]:
        payload = self.scoreboard_raw(day)
        if payload is None:
            return []
        if archive is not None:
            archive_raw(archive, day, payload)
        return parse_scoreboard(payload)


def archive_raw(root: Path, day: Date, payload: dict[str, object]) -> Path:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "espn" / day.isoformat() / f"scoreboard_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _halves(side: dict[str, object]) -> tuple[int, ...]:
    ls = side.get("linescores")
    if not isinstance(ls, list):
        return ()
    out = []
    for q in ls:
        v = q.get("value", 0) if isinstance(q, dict) else 0
        out.append(int(float(v)) if isinstance(v, (int, float, str)) and str(v) != "" else 0)
    return tuple(out)


def parse_scoreboard(payload: dict[str, object]) -> list[Game]:
    out: list[Game] = []
    events = payload.get("events")
    if not isinstance(events, list):
        return out
    for ev in events:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        home, away = sides.get("home"), sides.get("away")
        if home is None or away is None:
            continue
        out.append(
            Game(
                espn_id=str(ev.get("id", "")),
                start_utc=str(ev.get("date", "")),
                away=str(away.get("team", {}).get("displayName", "")),
                home=str(home.get("team", {}).get("displayName", "")),
                state=str(ev.get("status", {}).get("type", {}).get("name", "")),
                neutral=bool(comp.get("neutralSite", False)),
                conference_game=bool(comp.get("conferenceCompetition", False)),
                away_halves=_halves(away),
                home_halves=_halves(home),
            )
        )
    return out


__all__ = ["D1_GROUP", "SITE", "ESPNClient", "Game", "archive_raw", "parse_scoreboard"]
