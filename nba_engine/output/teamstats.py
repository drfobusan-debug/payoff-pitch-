"""Season-to-date team ratings for the card: home/away efficiency and the four factors.

Built from the ESPN box totals already archived under ``results/`` (fetched free
when a day is missing). Display only: nothing here reaches ``ledger.price``.
Regular season, play-in and playoffs count; preseason does not. Every figure
carries its games, because a split three games in ranks mostly on noise.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from nba_engine.data import boxes
from nba_engine.models.schedule import season_of
from nba_engine.schemas import GameResult, TeamBox

log = logging.getLogger(__name__)

COUNTED = (2, 3, 5)  # ESPN season types: regular season, playoffs, play-in
Fetch = Callable[[Date], list[GameResult] | None]


@dataclass(frozen=True)
class Rated:
    value: float | None = None
    rank: int | None = None


@dataclass(frozen=True)
class Split:
    """Points scored and allowed per 100 possessions at one venue."""

    games: int = 0
    off: Rated = field(default_factory=Rated)
    dfn: Rated = field(default_factory=Rated)
    net: Rated = field(default_factory=Rated)


@dataclass(frozen=True)
class Factors:
    """Dean Oliver's four factors for one side of the ball.

    On defense they are what opponents did, and rank 1 is the stingiest.
    """

    efg: Rated = field(default_factory=Rated)
    tov: Rated = field(default_factory=Rated)
    orb: Rated = field(default_factory=Rated)
    ftr: Rated = field(default_factory=Rated)


@dataclass(frozen=True)
class TeamStats:
    team: str
    games: int
    home: Split
    away: Split
    offense: Factors
    defense: Factors


def season_finals(root: Path, day: Date, fetch: Fetch | None = None) -> list[GameResult]:
    """Finals with both box totals, from 1 October of ``day``'s season to the day before."""
    out: list[GameResult] = []
    d = Date(season_of(day), 10, 1)
    while d < day:
        held = boxes.read_results(root, d)
        if held is None and fetch is not None:
            try:
                held = fetch(d)
            except Exception as exc:  # noqa: BLE001 - context only; the card goes out regardless
                log.warning("ESPN finals for %s not read: %s", d, exc)
        out.extend(g for g in held or [] if g.is_final and g.home_box and g.away_box)
        d += timedelta(days=1)
    return out


@dataclass
class _Venue:
    games: int = 0
    pts: float = 0.0
    opp: float = 0.0
    poss: float = 0.0


@dataclass
class _Box:
    fgm: float = 0.0
    fg3m: float = 0.0
    fga: float = 0.0
    fta: float = 0.0
    ftm: float = 0.0
    tov: float = 0.0
    oreb: float = 0.0
    opp_dreb: float = 0.0

    def add(self, mine: TeamBox, theirs: TeamBox) -> None:
        self.fgm += mine.fgm
        self.fg3m += mine.fg3m
        self.fga += mine.fga
        self.fta += mine.fta
        self.ftm += mine.ftm
        self.tov += mine.tov
        self.oreb += mine.oreb
        self.opp_dreb += theirs.dreb

    def factors(self) -> tuple[float | None, float | None, float | None, float | None]:
        plays = self.fga + 0.44 * self.fta + self.tov
        boards = self.oreb + self.opp_dreb
        return (
            (self.fgm + 0.5 * self.fg3m) / self.fga if self.fga else None,
            self.tov / plays if plays else None,
            self.oreb / boards if boards else None,
            self.ftm / self.fga if self.fga else None,
        )


def _ranks(values: dict[str, float | None], *, high_is_good: bool) -> dict[str, Rated]:
    have = sorted(((v, t) for t, v in values.items() if v is not None), reverse=high_is_good)
    out = {t: Rated(v) for t, v in values.items()}
    rank, prev = 0, None
    for i, (v, t) in enumerate(have, start=1):
        if v != prev:
            rank, prev = i, v
        out[t] = Rated(v, rank)
    return out


def season_table(games: Iterable[GameResult]) -> dict[str, TeamStats]:
    """Team code -> its ratings and four factors, ranked against every team with games."""
    venues: dict[tuple[str, str], _Venue] = {}
    off: dict[str, _Box] = {}
    dfn: dict[str, _Box] = {}
    played: dict[str, int] = {}
    for g in games:
        poss = g.possessions
        if g.season_type not in COUNTED or g.home_box is None or g.away_box is None or not poss:
            continue
        sides = (
            (g.home, "home", g.final_home, g.final_away, g.home_box, g.away_box),
            (g.away, "away", g.final_away, g.final_home, g.away_box, g.home_box),
        )
        for team, where, pts, opp, mine, theirs in sides:
            played[team] = played.get(team, 0) + 1
            off.setdefault(team, _Box()).add(mine, theirs)
            dfn.setdefault(team, _Box()).add(theirs, mine)
            if g.neutral:
                continue
            v = venues.setdefault((team, where), _Venue())
            v.games += 1
            v.pts += pts
            v.opp += opp
            v.poss += poss

    def rate(where: str) -> dict[str, Split]:
        held = {t: v for (t, w), v in venues.items() if w == where}
        o = _ranks({t: 100 * v.pts / v.poss for t, v in held.items()}, high_is_good=True)
        d = _ranks({t: 100 * v.opp / v.poss for t, v in held.items()}, high_is_good=False)
        n = _ranks({t: (o[t].value or 0.0) - (d[t].value or 0.0) for t in held}, high_is_good=True)
        return {t: Split(held[t].games, o[t], d[t], n[t]) for t in held}

    def four(tally: dict[str, _Box], *, defense: bool) -> dict[str, Factors]:
        raw = {t: b.factors() for t, b in tally.items()}
        # Offense wants shooting, boards and free throws up and turnovers down; defense the reverse.
        cols = [
            _ranks({t: f[i] for t, f in raw.items()}, high_is_good=good != defense)
            for i, good in enumerate((True, False, True, True))
        ]
        return {t: Factors(*(c[t] for c in cols)) for t in tally}

    home, away = rate("home"), rate("away")
    o4, d4 = four(off, defense=False), four(dfn, defense=True)
    return {
        t: TeamStats(t, n, home.get(t, Split()), away.get(t, Split()), o4[t], d4[t])
        for t, n in played.items()
    }


__all__ = ["COUNTED", "Factors", "Rated", "Split", "TeamStats", "season_finals", "season_table"]
