"""What the card shows beside each game's prices: injuries, rest, travel, line moves.

Read off disk from what capture, ``watch`` and ``injuries`` already archived
(plus, if asked, ESPN finals for the last few days, which are free). Display
only: nothing here reaches ``ledger.price``, so a missing source costs the
reader one line of colour and changes no row, no price and no gate.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from nba_engine import alarm
from nba_engine.audit.ledger import LedgerRow
from nba_engine.data import boxes, capture, espn_injuries, injuries, teamnames
from nba_engine.data.espn import SITE, ESPNClient
from nba_engine.data.oddsapi import parse_utc
from nba_engine.models.schedule import REST_CAP, schedule, season_of
from nba_engine.schemas import GameResult

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
# Rest is capped at REST_CAP days, so finals for that many days back decide
# rest, back-to-backs and three-in-four.
LOOKBACK_DAYS = REST_CAP
OFFICIAL, ESPN = "official", "espn"

# A day's games, [] for a day without games, None when the source could not be read.
Fetch = Callable[[Date], list[GameResult] | None]


def et_clock(stamp: str) -> str:
    """``2026-10-05T23:30:00Z`` -> ``7:30 PM ET``; ``""`` when unparseable."""
    moment = parse_utc(stamp) if stamp else None
    if moment is None:
        return ""
    return moment.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")


@dataclass(frozen=True)
class Injury:
    source: str  # official | espn
    team: str
    player: str
    status: str
    detail: str
    as_of: str  # the official report's stamp, or ESPN's update time


@dataclass(frozen=True)
class Rest:
    """One team's schedule going into tonight; ``known`` False when the finals
    needed to read it are not held (shown as unknown, never guessed)."""

    team: str
    known: bool
    rest_days: int | None = None
    b2b: bool = False
    three_in_four: bool = False
    miles: float | None = None
    tz_shift: float | None = None
    altitude: bool = False

    def text(self) -> str:
        if not self.known or self.rest_days is None:
            return f"{self.team}: rest unknown (recent finals not held)"
        if self.b2b:
            lead = "back-to-back"
        elif self.rest_days >= REST_CAP:
            lead = f"{REST_CAP}+ days since last game"
        else:
            lead = f"{self.rest_days} days since last game"
        bits = [f"{self.team}: {lead}"]
        if self.three_in_four:
            bits.append("3rd game in 4 nights")
        if self.miles is not None:
            bits.append("no travel" if self.miles < 1 else f"{self.miles:,.0f} mi travel")
        if self.tz_shift:
            bits.append(f"{self.tz_shift:+.0f}h time zone")
        if self.altitude:
            bits.append("altitude")
        return ", ".join(bits)


@dataclass(frozen=True)
class Move:
    key: str  # ml | spread | total
    open: float
    now: float

    def text(self, home: str) -> str:
        if self.key == "ml":
            return f"ML {home} {self.open * 100:.1f}% → {self.now * 100:.1f}%"
        if self.key == "spread":
            return f"spread {home} {self.open:+g} → {self.now:+g}"
        return f"total {self.open:g} → {self.now:g}"


@dataclass(frozen=True)
class GameContext:
    injuries: tuple[Injury, ...] = ()
    rest: tuple[Rest, ...] = ()  # away, home
    moves: tuple[Move, ...] = ()
    opened_at: str = ""  # UTC stamp of the first board, the "open"
    now_at: str = ""  # UTC stamp of the latest pre-tip board, the "now"
    official_report: str = ""  # stamp of the official report read; "" when none held
    espn_feed: str = ""  # file stamp of the ESPN feed read
    alarm: str = ""  # the news alarm's day for this game, one line

    def moves_text(self, home: str) -> str:
        """``12:43 PM ET → 6:05 PM ET: ML LAL 57.1% → 60.3% · ...``; "" with no board held."""
        if not self.moves:
            return ""
        span = f"{et_clock(self.opened_at)} → {et_clock(self.now_at)}: "
        return span + " · ".join(m.text(home) for m in self.moves)


@dataclass(frozen=True)
class Matchup:
    event_id: str
    away: str
    home: str
    tip_utc: str


def games_of(rows: Iterable[LedgerRow]) -> list[Matchup]:
    seen: dict[str, Matchup] = {}
    for r in rows:
        if r.event_id in seen:
            continue
        away, _, home = r.matchup.partition(" @ ")
        seen[r.event_id] = Matchup(r.event_id, away, home, r.tip_utc)
    return list(seen.values())


# -- injuries ----------------------------------------------------------------
def official(root: Path, day: Date) -> tuple[str, dict[str, list[Injury]]]:
    """The day's newest parsed official report, by team code; Available omitted."""
    rows = injuries.read_latest(root, day)
    out: dict[str, list[Injury]] = {}
    for r in rows:
        code = teamnames.code_for(r.team) or teamnames.canonical(r.team)
        if r.status in ("", "Available"):
            continue
        out.setdefault(code, []).append(
            Injury(OFFICIAL, code, _first_last(r.player), r.status, r.reason, r.report)
        )
    return (rows[0].report if rows else ""), out


def _first_last(name: str) -> str:
    last, sep, first = name.partition(", ")
    return f"{first} {last}" if sep else name


def espn(root: Path, day: Date) -> tuple[str, dict[str, list[Injury]]]:
    """The newest archived ESPN feed on or before the day, by team; active players omitted."""
    path = espn_injuries.latest_feed_path(root, day)
    if path is None:
        return "", {}
    out: dict[str, list[Injury]] = {}
    for r in espn_injuries.read_feed(path):
        if r.status in alarm.ACTIVE:
            continue
        team = teamnames.canonical(r.team)
        out.setdefault(team, []).append(
            Injury(ESPN, team, r.player, r.status, r.comment, r.updated)
        )
    stamp = path.stem.split("_")[1] if path.stem.count("_") >= 1 else ""
    return stamp, out


# -- rest and travel ---------------------------------------------------------
def prior_finals(root: Path, day: Date, fetch: Fetch | None) -> tuple[list[GameResult], set[Date]]:
    """Games on the ``LOOKBACK_DAYS`` before ``day`` and the days actually read.

    A day that could not be read leaves every team whose rest depends on it
    shown unknown: a team called rested because a feed failed is worse than one
    shown as unknown.
    """
    games: list[GameResult] = []
    known: set[Date] = set()
    for back in range(1, LOOKBACK_DAYS + 1):
        d = day - timedelta(days=back)
        held = boxes.read_results(root, d)
        if held is None and fetch is not None:
            try:
                held = fetch(d)
            except Exception as exc:  # noqa: BLE001 - context only; the card goes out regardless
                log.warning("ESPN finals for %s not read: %s", d, exc)
                held = None
        if held is not None:
            known.add(d)
            games.extend(held)
    return games, known


def rest(
    day: Date, games: list[Matchup], prior: list[GameResult], known: set[Date]
) -> dict[tuple[str, str], Rest]:
    """``(event_id, team)`` -> :class:`Rest` going into tonight."""
    tonight = [
        GameResult(f"tonight:{g.event_id}", day, g.away, g.home, "STATUS_SCHEDULED") for g in games
    ]
    states = schedule([*prior, *tonight])
    out: dict[tuple[str, str], Rest] = {}
    for g in games:
        for team in (g.away, g.home):
            ts = states.get((f"tonight:{g.event_id}", team))
            played = [
                p.game_date
                for p in prior
                if team in (p.away, p.home) and season_of(p.game_date) == season_of(day)
            ]
            last = max(played) if played else None
            start = last + timedelta(days=1) if last else day - timedelta(days=REST_CAP - 1)
            span = {start + timedelta(days=i) for i in range((day - start).days)}
            if ts is None or not span <= known:
                out[(g.event_id, team)] = Rest(team, known=False)
                continue
            out[(g.event_id, team)] = Rest(
                team,
                known=True,
                rest_days=ts.rest_days,
                b2b=ts.b2b,
                three_in_four=ts.three_in_four,
                # Travel needs the previous site; with no game in the window the
                # team may be coming off a road trip, so it is not guessed.
                miles=ts.miles if last else None,
                tz_shift=ts.tz_shift if last else None,
                altitude=ts.altitude,
            )
    return out


# -- line movement and the news alarm ---------------------------------------
def moves(root: Path, day: Date, games: list[Matchup]) -> dict[str, tuple[list[Move], str, str]]:
    """Per event: consensus at the day's first board vs the latest board before tip."""
    boards = [capture.read_snapshot(p) for p in capture.snapshot_paths(root, day, label="board")]
    boards = [b for b in boards if b]
    out: dict[str, tuple[list[Move], str, str]] = {}
    for g in games:
        tip = parse_utc(g.tip_utc) if g.tip_utc else None
        first: tuple[str, dict[str, float]] | None = None
        last: tuple[str, dict[str, float]] | None = None
        for rows in boards:
            taken = rows[0].captured_at
            stamp = parse_utc(taken)
            if tip is not None and stamp is not None and stamp >= tip:
                continue
            cons = alarm.consensus(r for r in rows if r.event_id == g.event_id).get(g.event_id)
            if not cons:
                continue
            if first is None:
                first = (taken, cons)
            last = (taken, cons)
        if first is None or last is None:
            continue
        mv = [
            Move(k, first[1][k], last[1][k])
            for k in ("ml", "spread", "total")
            if k in first[1] and k in last[1]
        ]
        out[g.event_id] = (mv, first[0], last[0])
    return out


def alarm_lines(root: Path, day: Date) -> dict[str, str]:
    alerts = alarm.read_alerts(root, day)
    unsettled = alarm.unsettled(alerts)
    out: dict[str, str] = {}
    for eid in sorted({a.event_id for a in alerts if a.event_id}):
        mine = [a for a in alerts if a.event_id == eid]
        kinds = Counter(a.kind for a in mine if a.kind != alarm.SETTLED)
        names = {
            "ml_move": "ML move",
            "spread_move": "spread move",
            "total_move": "total move",
            "injury": "ESPN injury change",
            "injury_official": "official report change",
        }
        bits = [f"{n} {names.get(k, k)}{'s' if n > 1 else ''}" for k, n in sorted(kinds.items())]
        settled = [a.detected_at for a in mine if a.kind == alarm.SETTLED]
        if eid in unsettled:
            bits.append("line not yet settled after the latest news")
        elif settled:
            bits.append(f"line settled {et_clock(max(settled))}")
        if bits:
            out[eid] = "; ".join(bits)
    return out


def gather(
    root: Path, day: Date, rows: Iterable[LedgerRow], *, fetch: Fetch | None = None
) -> dict[str, GameContext]:
    """Per event id, everything the card prints beside the prices."""
    games = games_of(rows)
    report, by_team = official(root, day)
    feed_at, feed = espn(root, day)
    prior, known = prior_finals(root, day, fetch)
    rests = rest(day, games, prior, known)
    moved = moves(root, day, games)
    alarms = alarm_lines(root, day)
    out: dict[str, GameContext] = {}
    for g in games:
        teams = (g.away, g.home)
        hurt = [i for t in teams for i in by_team.get(t, [])] + [
            i for t in teams for i in feed.get(t, [])
        ]
        mv, opened, now = moved.get(g.event_id, ([], "", ""))
        out[g.event_id] = GameContext(
            injuries=tuple(hurt),
            rest=tuple(rests[(g.event_id, t)] for t in teams),
            moves=tuple(mv),
            opened_at=opened,
            now_at=now,
            official_report=report,
            espn_feed=feed_at,
            alarm=alarms.get(g.event_id, ""),
        )
    return out


def espn_fetch(root: Path) -> Fetch:
    """Finals for a past day from ESPN (free), archived once the day is settled.

    The scoreboard is read first so a day without games ([]) is told apart
    from a feed that failed (None).
    """
    client = ESPNClient()

    def fetch(d: Date) -> list[GameResult] | None:
        try:
            resp = requests.get(
                f"{SITE}/scoreboard", params={"dates": d.strftime("%Y%m%d")}, timeout=20
            )
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("ESPN scoreboard %s not read: %s", d, exc)
            return None
        if not isinstance(payload, dict) or not payload.get("events"):
            return [] if isinstance(payload, dict) else None
        return boxes.ensure_results(root, d, client) or None

    return fetch


__all__ = [
    "ESPN",
    "ET",
    "LOOKBACK_DAYS",
    "OFFICIAL",
    "GameContext",
    "Injury",
    "Matchup",
    "Move",
    "Rest",
    "alarm_lines",
    "espn",
    "espn_fetch",
    "et_clock",
    "games_of",
    "gather",
    "moves",
    "official",
    "prior_finals",
    "rest",
]
