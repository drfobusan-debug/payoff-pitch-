"""Typed data structures passed between pipeline stages."""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field


class TeamGameInfo(BaseModel):
    """One team's participation in a game."""

    name: str  # canonical full name from the odds board (e.g. "Alabama Crimson Tide")
    abbrev: str  # short display code (e.g. "ALA")
    is_home: bool


class Game(BaseModel):
    game_id: str  # The Odds API event id (stable per game)
    game_date: Date
    commence_time_utc: str | None = None
    home: TeamGameInfo
    away: TeamGameInfo

    def matchup(self) -> str:
        return f"{self.away.abbrev} @ {self.home.abbrev}"


class Slate(BaseModel):
    slate_date: Date
    games: list[Game] = Field(default_factory=list)
