"""Typed data structures passed between pipeline stages."""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field


class TeamGameInfo(BaseModel):
    """One team's participation in a game."""

    name: str  # canonical full name from the odds board (e.g. "Kansas City Chiefs")
    abbrev: str  # nflverse team code (e.g. "KC")
    is_home: bool


class GameEnvironment(BaseModel):
    """Venue and weather, all optional: every consumer degrades to a no-op.

    Sourced from the historical game file for a backtest and from the schedule
    for a live slate. ``roof`` is the nflverse vocabulary ("outdoors", "dome",
    "closed", "open"); a dome or closed roof means the weather fields are moot.
    """

    roof: str | None = None
    surface: str | None = None
    temp_f: float | None = None
    wind_mph: float | None = None
    neutral_site: bool = False

    def is_indoors(self) -> bool:
        return (self.roof or "").lower() in ("dome", "closed")


class Game(BaseModel):
    game_id: str  # The Odds API event id for a live slate, nflverse id historically
    season: int
    week: int
    game_date: Date
    kickoff_utc: str | None = None
    home: TeamGameInfo
    away: TeamGameInfo
    env: GameEnvironment = Field(default_factory=GameEnvironment)
    # Rest days since each team's previous game; a bye shows up as ~13.
    home_rest: int | None = None
    away_rest: int | None = None

    def matchup(self) -> str:
        return f"{self.away.abbrev} @ {self.home.abbrev}"


class Slate(BaseModel):
    """One week's games. The NFL slate is a week, not a day: Thursday night,
    Sunday's windows and Monday night are priced together off the same ratings.
    """

    season: int
    week: int
    slate_date: Date
    games: list[Game] = Field(default_factory=list)
