"""Shared record types for the NBA engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date


@dataclass(frozen=True)
class Game:
    """One scheduled game as the odds board or ESPN describes it."""

    game_date: Date  # slate date in US Eastern
    home: str  # NBA tricode
    away: str
    start_utc: str = ""
    event_id: str = ""  # Odds API event id
    espn_id: str = ""

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"


@dataclass
class Slate:
    slate_date: Date
    games: list[Game] = field(default_factory=list)


@dataclass(frozen=True)
class PlayerLine:
    """One player's box line; what a prop settles on."""

    espn_id: str
    team: str
    name: str
    minutes: int
    points: int
    rebounds: int
    assists: int
    threes: int
    starter: bool = False
    dnp: bool = False

    @property
    def pra(self) -> int:
        return self.points + self.rebounds + self.assists


@dataclass(frozen=True)
class TeamBox:
    """One team's box totals: what possessions and per-100 rates are built from."""

    team: str
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    tov: int  # total turnovers, team turnovers included
    fouls: int

    @property
    def possessions(self) -> float:
        """The standard box estimate: FGA - OREB + TOV + 0.44 * FTA."""
        return self.fga - self.oreb + self.tov + 0.44 * self.fta


@dataclass(frozen=True)
class GameResult:
    """Final as ESPN reports it, per quarter, with the box lines props settle on.

    ``away_q``/``home_q`` hold one entry per period played: four quarters, then
    one per overtime.
    """

    espn_id: str
    game_date: Date
    away: str
    home: str
    state: str  # STATUS_FINAL | STATUS_SCHEDULED | ...
    away_q: tuple[int, ...] = ()
    home_q: tuple[int, ...] = ()
    players: tuple[PlayerLine, ...] = ()
    away_box: TeamBox | None = None
    home_box: TeamBox | None = None
    neutral: bool = False  # ESPN's neutral-site flag
    venue: str = ""
    city: str = ""
    season_type: int = 0  # ESPN: 1 preseason, 2 regular, 3 playoffs, 5 play-in

    @property
    def possessions(self) -> float | None:
        """Game possessions per team: the mean of both teams' box estimates."""
        if self.away_box is None or self.home_box is None:
            return None
        return (self.away_box.possessions + self.home_box.possessions) / 2.0

    @property
    def is_final(self) -> bool:
        return self.state == "STATUS_FINAL"

    @property
    def final_away(self) -> int:
        return sum(self.away_q)

    @property
    def final_home(self) -> int:
        return sum(self.home_q)

    @property
    def h1_away(self) -> int:
        return sum(self.away_q[:2])

    @property
    def h1_home(self) -> int:
        return sum(self.home_q[:2])

    @property
    def overtimes(self) -> int:
        return max(0, len(self.home_q) - 4)
