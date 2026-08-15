"""Client for the free official MLB Stats API (statsapi.mlb.com).

Provides the daily slate: matchups, probable pitchers, confirmed/expected
lineups, and venue metadata. No authentication required.
"""

from __future__ import annotations

import logging
from datetime import date as Date
from datetime import timedelta

import requests

from mlb_engine.data import http
from mlb_engine.schemas import (
    BatterSlot,
    Game,
    Hand,
    Pitcher,
    Player,
    Slate,
    TeamGameInfo,
    Venue,
)

log = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
SPORT_ID = 1  # MLB


def _utc_hour(iso: str | None) -> float | None:
    """Fractional UTC hour from an ISO game-start string (e.g. 2026-07-19T23:05:00Z)."""
    if not iso:
        return None
    try:
        t = iso.split("T", 1)[1]
        hh, mm = t[:2], t[3:5]
        return int(hh) + int(mm) / 60.0
    except (IndexError, ValueError):
        return None


class MLBStatsClient:
    def __init__(self, session: requests.Session | None = None, timeout: int = 20) -> None:
        self.session = session or http.session()
        self.timeout = timeout

    def _get(self, path: str, **params: str | int) -> dict:
        resp = self.session.get(f"{BASE}/{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _people_handedness(self, ids: set[int]) -> dict[int, tuple[Hand | None, Hand | None]]:
        """Return {mlbam_id: (bats, throws)} for the given player ids."""
        out: dict[int, tuple[Hand | None, Hand | None]] = {}
        ids = {i for i in ids if i}
        if not ids:
            return out
        # people endpoint accepts a comma-separated list.
        chunk = ",".join(str(i) for i in sorted(ids))
        data = self._get("people", personIds=chunk)
        for person in data.get("people", []):
            pid = person.get("id")
            bats = (person.get("batSide") or {}).get("code")
            throws = (person.get("pitchHand") or {}).get("code")
            out[pid] = (
                Hand(bats) if bats in Hand._value2member_map_ else None,
                Hand(throws) if throws in Hand._value2member_map_ else None,
            )
        return out

    def get_slate(self, slate_date: Date) -> Slate:
        """Fetch the full slate for a date with pitchers, lineups, and venues."""
        data = self._get(
            "schedule",
            sportId=SPORT_ID,
            date=slate_date.isoformat(),
            hydrate="probablePitcher,lineups,team,venue",
        )

        games: list[Game] = []
        pending_ids: set[int] = set()
        raw_games: list[dict] = []
        for date_block in data.get("dates", []):
            raw_games.extend(date_block.get("games", []))

        # First pass: collect all player ids needing handedness.
        for g in raw_games:
            teams = g.get("teams", {})
            for side in ("home", "away"):
                pp = teams.get(side, {}).get("probablePitcher") or {}
                if pp.get("id"):
                    pending_ids.add(pp["id"])
            lineups = g.get("lineups", {})
            for key in ("homePlayers", "awayPlayers"):
                for pl in lineups.get(key, []) or []:
                    if pl.get("id"):
                        pending_ids.add(pl["id"])

        hand = self._people_handedness(pending_ids)

        for g in raw_games:
            games.append(self._parse_game(g, slate_date, hand))

        return Slate(slate_date=slate_date, games=games)

    def _parse_game(
        self,
        g: dict,
        slate_date: Date,
        hand: dict[int, tuple[Hand | None, Hand | None]],
    ) -> Game:
        teams = g.get("teams", {})
        venue_raw = g.get("venue", {})
        venue = Venue(
            venue_id=venue_raw.get("id", 0),
            name=venue_raw.get("name", "Unknown"),
        )
        lineups = g.get("lineups", {})

        home = self._parse_team(teams.get("home", {}), lineups.get("homePlayers"), True, hand)
        away = self._parse_team(teams.get("away", {}), lineups.get("awayPlayers"), False, hand)

        return Game(
            game_pk=g.get("gamePk", 0),
            game_date=slate_date,
            game_datetime_utc=g.get("gameDate"),
            status=(g.get("status", {}) or {}).get("detailedState", "Unknown"),
            venue=venue,
            home=home,
            away=away,
        )

    def _parse_team(
        self,
        side: dict,
        lineup_players: list[dict] | None,
        is_home: bool,
        hand: dict[int, tuple[Hand | None, Hand | None]],
    ) -> TeamGameInfo:
        team = side.get("team", {})
        pp_raw = side.get("probablePitcher") or {}
        probable = None
        if pp_raw.get("id"):
            _, throws = hand.get(pp_raw["id"], (None, None))
            probable = Pitcher(
                mlbam_id=pp_raw["id"],
                name=pp_raw.get("fullName", "TBD"),
                throws=throws,
            )

        lineup: list[BatterSlot] = []
        for order, pl in enumerate(lineup_players or [], start=1):
            pid = pl.get("id")
            bats, _ = hand.get(pid, (None, None)) if pid else (None, None)
            pos = (pl.get("primaryPosition") or {}).get("abbreviation")
            lineup.append(
                BatterSlot(
                    order=order,
                    player=Player(
                        mlbam_id=pid or 0, name=pl.get("fullName", "?"), bats=bats, position=pos
                    ),
                )
            )

        return TeamGameInfo(
            team_id=team.get("id", 0),
            name=team.get("name", "Unknown"),
            abbrev=team.get("abbreviation") or team.get("teamCode", "").upper() or "UNK",
            is_home=is_home,
            probable_pitcher=probable,
            lineup=lineup,
        )

    def team_roster(
        self, team_id: int, season: int
    ) -> list[tuple[int, str, Hand | None, Hand | None]]:
        """Return ``(mlbam_id, full_name, bats, throws)`` for a team's roster.

        Used to resolve expected-lineup names (e.g. from Rotowire) to the MLBAM
        ids the rest of the engine keys on. Returns ``[]`` on failure.
        """
        try:
            data = self._get(f"teams/{team_id}/roster", rosterType="fullRoster", season=season)
        except requests.RequestException as exc:
            log.warning("roster fetch failed for team %s: %s", team_id, exc)
            return []
        people = [p.get("person", {}) for p in data.get("roster", [])]
        ids = {p["id"] for p in people if p.get("id")}
        hand = self._people_handedness(ids)
        out: list[tuple[int, str, Hand | None, Hand | None]] = []
        for p in people:
            pid = p.get("id")
            if not pid:
                continue
            bats, throws = hand.get(pid, (None, None))
            out.append((pid, p.get("fullName", ""), bats, throws))
        return out

    def get_today_and_tomorrow(self, today: Date | None = None) -> tuple[Slate, Slate]:
        today = today or Date.today()
        return self.get_slate(today), self.get_slate(today + timedelta(days=1))

    def bullpen_fatigue(self, team_id: int, before: Date, lookback_days: int = 3) -> float | None:
        """Return a 0-100 bullpen-fatigue proxy from recent real pitch counts.

        Transparent proxy for the proprietary fatigue trackers the run-line and
        comeback layers reference: over the team's last ``lookback_days`` game
        days it counts relievers who are "gassed" -- appeared on back-to-back
        days, or threw a heavy recent workload -- and scales that to 0-100.
        Higher = more depleted high-leverage depth. ``None`` if no data.
        """
        start = before - timedelta(days=lookback_days)
        end = before - timedelta(days=1)
        try:
            data = self._get(
                "schedule", sportId=SPORT_ID, teamId=team_id,
                startDate=start.isoformat(), endDate=end.isoformat(),
            )
        except requests.RequestException as exc:
            log.warning("bullpen fatigue schedule failed for %s: %s", team_id, exc)
            return None

        game_days: list[tuple[Date, int]] = []
        for block in data.get("dates", []):
            d = Date.fromisoformat(block["date"])
            for g in block.get("games", []):
                pk = g.get("gamePk")
                if pk and (g.get("status", {}) or {}).get("abstractGameState") == "Final":
                    game_days.append((d, int(pk)))
        if not game_days:
            return None
        game_days.sort()

        # pitches[reliever_id] = {day: pitches}
        pitches: dict[int, dict[Date, int]] = {}
        for d, pk in game_days:
            try:
                box = self._get(f"game/{pk}/boxscore")
            except requests.RequestException:
                continue
            for side in ("home", "away"):
                team_block = (box.get("teams", {}) or {}).get(side, {}) or {}
                if (team_block.get("team", {}) or {}).get("id") != team_id:
                    continue
                players = team_block.get("players", {}) or {}
                for pid in team_block.get("pitchers", []):
                    pdata = players.get(f"ID{pid}", {}) or {}
                    pstats = (pdata.get("stats", {}) or {}).get("pitching", {}) or {}
                    if int(pstats.get("gamesStarted", 0) or 0) > 0:
                        continue  # starter, not a reliever
                    npitch = int(pstats.get("numberOfPitches", 0) or 0)
                    if npitch > 0:
                        pitches.setdefault(int(pid), {})[d] = npitch

        if not pitches:
            return None
        recent_days = [d for d, _ in game_days][-2:]
        gassed = 0
        for _, by_day in pitches.items():
            days_used = [d for d in recent_days if d in by_day]
            two_day = sum(by_day.get(d, 0) for d in recent_days)
            last_day = by_day.get(recent_days[-1], 0) if recent_days else 0
            if len(days_used) >= 2 or two_day >= 40 or last_day >= 30:
                gassed += 1
        return float(min(100, gassed * 20))

    def last_game_venue(self, team_id: int, before: Date, lookback_days: int = 8):
        """Return (game_date, venue_id, start_hour_utc) of the team's most recent
        game before ``before`` (``start_hour_utc`` is None if unparseable).

        Returns None if none found in the lookback window.
        """
        start = before - timedelta(days=lookback_days)
        end = before - timedelta(days=1)
        data = self._get(
            "schedule",
            sportId=SPORT_ID,
            teamId=team_id,
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            hydrate="venue",
        )
        best: tuple[Date, int, float | None] | None = None
        for date_block in data.get("dates", []):
            d = Date.fromisoformat(date_block["date"])
            for g in date_block.get("games", []):
                venue_id = (g.get("venue", {}) or {}).get("id")
                if venue_id and (best is None or d > best[0]):
                    best = (d, venue_id, _utc_hour(g.get("gameDate")))
        return best

    def team_run_differentials(self, season: int) -> dict[str, tuple[float, int]]:
        """Season {team_abbrev: (actual_rd_per_game, games_played)} from standings.

        Feeds the run-line luck-gap baseline: a team's *actual* run differential,
        compared against its xwOBA-based expected differential, flags sequencing
        luck. Keyed by the same team abbreviation the Statcast frame uses.
        """
        teams = self._get("teams", sportId=SPORT_ID, season=season).get("teams", [])
        id_to_abbr = {
            t["id"]: t["abbreviation"]
            for t in teams
            if t.get("id") and t.get("abbreviation")
        }
        data = self._get(
            "standings", leagueId="103,104", season=season, standingsTypes="regularSeason"
        )
        out: dict[str, tuple[float, int]] = {}
        for rec in data.get("records", []):
            for tr in rec.get("teamRecords", []):
                abbr = id_to_abbr.get((tr.get("team") or {}).get("id"))
                games = int(tr.get("gamesPlayed") or 0)
                if abbr and games:
                    out[abbr] = (float(tr.get("runDifferential", 0)) / games, games)
        return out
