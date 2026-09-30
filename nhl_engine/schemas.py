"""Shared record types for the NHL engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date


@dataclass(frozen=True)
class Game:
    """One scheduled game as the odds board or the NHL API describes it."""

    game_date: Date  # slate date in US Eastern
    home: str  # NHL API abbrev
    away: str
    start_utc: str = ""
    event_id: str = ""  # Odds API event id
    nhl_game_id: int | None = None

    @property
    def matchup(self) -> str:
        return f"{self.away} @ {self.home}"


@dataclass
class Slate:
    slate_date: Date
    games: list[Game] = field(default_factory=list)


@dataclass(frozen=True)
class PeriodScore:
    number: int
    kind: str  # REG | OT | SO
    away: int
    home: int


@dataclass(frozen=True)
class GameResult:
    """Official final as the NHL API reports it, with what settlement needs.

    ``reg_away``/``reg_home`` are the 60-minute scores. ``final_*`` include the
    credited OT goal, and for a shootout the one credited goal to the winner.
    ``decided`` is ``REG`` | ``OT`` | ``SO``.
    """

    nhl_game_id: int
    game_date: Date
    away: str
    home: str
    state: str  # FUT | LIVE | OFF | FINAL ...
    periods: tuple[PeriodScore, ...] = ()
    decided: str = ""
    away_starter: str = ""
    home_starter: str = ""

    @property
    def is_final(self) -> bool:
        return self.state in ("OFF", "FINAL")

    def period(self, n: int) -> PeriodScore | None:
        return next((p for p in self.periods if p.number == n and p.kind == "REG"), None)

    @property
    def reg_away(self) -> int:
        return sum(p.away for p in self.periods if p.kind == "REG")

    @property
    def reg_home(self) -> int:
        return sum(p.home for p in self.periods if p.kind == "REG")

    @property
    def final_away(self) -> int:
        return (
            self.reg_away + sum(p.away for p in self.periods if p.kind == "OT") + self._so("away")
        )

    @property
    def final_home(self) -> int:
        return (
            self.reg_home + sum(p.home for p in self.periods if p.kind == "OT") + self._so("home")
        )

    def _so(self, side: str) -> int:
        """A shootout credits exactly one goal to its winner, whatever the round count."""
        so = next((p for p in self.periods if p.kind == "SO"), None)
        if so is None:
            return 0
        won = so.away > so.home if side == "away" else so.home > so.away
        return 1 if won else 0
