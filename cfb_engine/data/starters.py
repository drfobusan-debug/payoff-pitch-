"""Who has been taking the snaps, from earlier weeks only.

An injury designation is worth nothing on its own -- the measured effect depends
entirely on how much of a team's work the missing man was doing. A quarterback who
had thrown 90% of the passes is worth -2.2 points against the closing spread; one
splitting time under 50% is worth *plus* 1.5, because losing a rotation player is
not bad news. So the feed has to be read against usage, and quarterbacks are the
position box scores identify cleanly.

Usage is counted from weeks strictly before the slate, which is both leak-free and
the only view a Saturday morning actually has. ``missed_last_week`` is the other
half of the design: an absence the market has already seen priced correctly in the
holdout (47.1%), so it marks the games where the news is stale.
"""

from __future__ import annotations

from dataclasses import dataclass

from cfb_engine.data.teamnames import school_key

# Below this many prior attempts nobody is an established starter yet, so an
# absence carries no measurable signal.
MIN_PRIOR_ATTEMPTS = 30
# The share of prior attempts at which the effect turned negative and stayed
# monotone; under it the sign flips positive.
STARTER_SHARE = 0.75


@dataclass(frozen=True)
class PasserGame:
    """One quarterback's attempts in one team-game."""

    week: int
    team: str  # school_key
    player_id: str
    name: str
    attempts: int


@dataclass(frozen=True)
class Starter:
    player_id: str
    name: str
    attempts: int
    share: float
    missed_last_week: bool

    @property
    def established(self) -> bool:
        return self.attempts >= MIN_PRIOR_ATTEMPTS and self.share >= STARTER_SHARE


StarterBook = dict[str, Starter]


def build_starter_book(rows: list[PasserGame], *, through_week: int) -> StarterBook:
    """The primary passer per team, counting only weeks before ``through_week``."""
    by_team: dict[str, list[PasserGame]] = {}
    for row in rows:
        if row.week >= through_week or row.attempts <= 0:
            continue
        by_team.setdefault(school_key(row.team), []).append(row)

    book: StarterBook = {}
    for team, games in by_team.items():
        totals: dict[str, int] = {}
        names: dict[str, str] = {}
        for game in games:
            totals[game.player_id] = totals.get(game.player_id, 0) + game.attempts
            names[game.player_id] = game.name
        overall = sum(totals.values())
        if overall <= 0:
            continue
        top = max(totals, key=lambda pid: totals[pid])
        last_week = max(game.week for game in games)
        threw_last_week = any(
            game.week == last_week and game.player_id == top for game in games
        )
        book[team] = Starter(
            player_id=top,
            name=names[top],
            attempts=totals[top],
            share=totals[top] / overall,
            missed_last_week=not threw_last_week,
        )
    return book


def starter_absent(starter: Starter | None, out_names: list[str]) -> bool:
    """Is the established starter among the players listed unavailable?"""
    if starter is None or not starter.established:
        return False
    wanted = _key(starter.name)
    return any(_key(name) == wanted for name in out_names)


def _key(name: str) -> str:
    return " ".join(name.replace(".", " ").lower().split())
