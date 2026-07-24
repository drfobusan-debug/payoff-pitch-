"""Typed data structures passed between pipeline stages."""

from __future__ import annotations

from datetime import date as Date
from enum import Enum

from pydantic import BaseModel, Field


class Hand(str, Enum):
    L = "L"
    R = "R"
    S = "S"  # switch (batters only)


class Player(BaseModel):
    mlbam_id: int
    name: str
    bats: Hand | None = None
    throws: Hand | None = None
    position: str | None = None  # primary position abbreviation (e.g. "C")


class Pitcher(Player):
    pass


class BatterSlot(BaseModel):
    order: int  # 1-9
    player: Player


class TeamGameInfo(BaseModel):
    team_id: int
    name: str
    abbrev: str
    is_home: bool
    probable_pitcher: Pitcher | None = None
    lineup: list[BatterSlot] = Field(default_factory=list)

    def lineup_confirmed(self) -> bool:
        return len(self.lineup) >= 9


class Venue(BaseModel):
    venue_id: int
    name: str
    lat: float | None = None
    lon: float | None = None
    orientation_deg: float | None = None  # home plate -> center field bearing
    roof: str | None = None  # "open", "closed", "retractable", None


class Game(BaseModel):
    game_pk: int
    game_date: Date
    game_datetime_utc: str | None = None
    status: str
    venue: Venue
    home: TeamGameInfo
    away: TeamGameInfo

    def matchup(self) -> str:
        return f"{self.away.abbrev} @ {self.home.abbrev}"


class Slate(BaseModel):
    slate_date: Date
    games: list[Game] = Field(default_factory=list)
