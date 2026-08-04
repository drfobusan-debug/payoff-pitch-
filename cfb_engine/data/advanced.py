"""CFBD advanced season stats: the play-level efficiency metrics that carry the
documented predictive value for CFB markets.

Sourced from ``/stats/season/advanced`` (PPA/EPA, success rate, explosiveness,
havoc, finishing drives, pace) plus ``/stats/season`` (turnover components).
Each team's row is normalized to a :class:`TeamAdvanced`, collected into an
:class:`AdvancedBook` that also carries the slate's league means so a metric can
be judged "above / below average" without hard-coded cutoffs.

Everything here is optional and fail-soft: a missing key or an unentitled
endpoint yields an empty book, and the marking layer / Markov engine both no-op
back to the ratings-only path when a team's stats are absent.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from cfb_engine.data.teamnames import school_key


@dataclass(frozen=True)
class TeamAdvanced:
    """One team's season efficiency profile (higher = better, unless noted)."""

    team: str
    games: int
    # Net PPA per play = offense PPA minus defense PPA allowed. The single
    # strongest usable ML/ATS signal (overall Big-Play Impact).
    off_ppa: float
    def_ppa: float  # points allowed per defensive play (lower is better)
    off_success: float
    def_success: float  # opponent success rate allowed (lower is better)
    off_explosive: float
    def_explosive: float  # opponent explosiveness allowed (lower is better)
    off_finishing: float  # points per scoring opportunity (drive past the 40)
    def_finishing: float  # points per opportunity allowed (lower is better)
    havoc: float  # defensive disruption rate created (higher is better)
    plays_per_game: float  # offensive tempo proxy (pace)
    drives_per_game: float
    turnover_margin_pg: float  # (takeaways - giveaways) per game

    @property
    def net_ppa(self) -> float:
        return self.off_ppa - self.def_ppa

    @property
    def net_success(self) -> float:
        return self.off_success - self.def_success

    @property
    def net_explosive(self) -> float:
        return self.off_explosive - self.def_explosive


@dataclass(frozen=True)
class AdvancedBook:
    """Per-team advanced stats plus the slate-wide means used as thresholds."""

    teams: dict[str, TeamAdvanced]  # keyed by school_key
    mean_off_ppa: float
    mean_off_explosive: float
    mean_off_finishing: float
    mean_plays_per_game: float

    def get(self, team_name: str) -> TeamAdvanced | None:
        return self.teams.get(school_key(team_name))


def _num(row: object, *path: str) -> float | None:
    cur: object = row
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _turnover_margin(stats: dict[str, float]) -> tuple[float, int]:
    """Turnover margin per game from season component stats.

    takeaways = defensive INTs + opponent fumbles recovered;
    giveaways = own passes intercepted + own fumbles lost.
    An extreme margin is the documented regression (fade) signal.
    """
    games = int(stats.get("games", 0) or 0)
    if games <= 0:
        return 0.0, 0
    takeaways = stats.get("interceptions", 0.0) + stats.get("fumblesRecovered", 0.0)
    giveaways = stats.get("passesIntercepted", 0.0) + stats.get("fumblesLost", 0.0)
    return (takeaways - giveaways) / games, games


def parse_advanced(
    advanced_rows: list[dict[str, object]],
    season_stats: dict[str, dict[str, float]],
) -> AdvancedBook:
    """Build the book from raw ``/stats/season/advanced`` + ``/stats/season`` rows."""
    teams: dict[str, TeamAdvanced] = {}
    for row in advanced_rows:
        team = row.get("team")
        if not team:
            continue
        off_ppa = _num(row, "offense", "ppa")
        def_ppa = _num(row, "defense", "ppa")
        off_plays = _num(row, "offense", "plays")
        off_drives = _num(row, "offense", "drives")
        if off_ppa is None or def_ppa is None:
            continue
        key = school_key(str(team))
        to_margin, so_games = _turnover_margin(season_stats.get(key, {}))
        # Prefer the advanced row's own game count; fall back to the season-stats
        # count. With no reliable count we leave pace at 0.0 -- the "unknown"
        # value MarkovSim treats as "use the default drive count" -- rather than
        # dividing season totals by 1 and simulating at absurd tempo.
        row_games = _num(row, "games")
        games = int(row_games) if row_games and row_games > 0 else so_games
        per_game = games if games > 0 else None
        teams[key] = TeamAdvanced(
            team=str(team),
            games=games,
            off_ppa=off_ppa,
            def_ppa=def_ppa,
            off_success=_num(row, "offense", "successRate") or 0.0,
            def_success=_num(row, "defense", "successRate") or 0.0,
            off_explosive=_num(row, "offense", "explosiveness") or 0.0,
            def_explosive=_num(row, "defense", "explosiveness") or 0.0,
            off_finishing=_num(row, "offense", "pointsPerOpportunity") or 0.0,
            def_finishing=_num(row, "defense", "pointsPerOpportunity") or 0.0,
            havoc=_num(row, "defense", "havoc", "total") or 0.0,
            plays_per_game=(off_plays / per_game) if (off_plays and per_game) else 0.0,
            drives_per_game=(off_drives / per_game) if (off_drives and per_game) else 0.0,
            turnover_margin_pg=to_margin,
        )
    return _finalize(teams)


def _finalize(teams: dict[str, TeamAdvanced]) -> AdvancedBook:
    def mean(vals: list[float], default: float) -> float:
        known = [v for v in vals if v > 0.0]
        return statistics.fmean(known) if known else default

    return AdvancedBook(
        teams=teams,
        mean_off_ppa=statistics.fmean([t.off_ppa for t in teams.values()]) if teams else 0.0,
        mean_off_explosive=mean([t.off_explosive for t in teams.values()], 1.2),
        mean_off_finishing=mean([t.off_finishing for t in teams.values()], 4.3),
        mean_plays_per_game=mean([t.plays_per_game for t in teams.values()], 68.0),
    )
