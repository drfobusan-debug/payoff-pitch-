"""Reading batting order out of the play-by-play feed.

Batting order is recovered by counting: the k-th plate appearance a team takes
belongs to slot ``k % 9`` whoever is standing there, so the slot's original
occupant is whoever batted there the first time through and a substitution is the
first appearance that belongs to somebody else.

That is exact, and it is also fragile in one specific way. A caught stealing, a
pickoff or a runner thrown out arrives as its own play with ``result.type ==
"atBat"`` -- the same type a real plate appearance carries -- but the batter is
still standing at the plate and his appearance finishes on the *next* play.
Counting one of those advances the pointer, and from that moment every slot in
the game is attributed to the wrong man: the rest of the game reads as nine
simultaneous substitutions that never happened.

It is rare per play and common per game. Across 563 games the feed carries 89 such
plays, which is one in roughly every six games, and each one corrupts everything
after it. Measured against the box score's own ``battingOrder`` codes on 120
games, counting them puts the starter's share of his slot's appearances at 92.9%
against a true 95.8%, and agrees with the box score on 94.5% of slot-games.
Dropping them agrees on 99.9%.
"""

from __future__ import annotations

from typing import Any

# Typed ``atBat`` by the feed, but the batter's appearance does not end here, so
# these must not advance the batting-order pointer. Keyed on ``eventType``, which
# is stable, rather than the display ``event``.
RUNNER_EVENT_TYPES = frozenset(
    {
        "caught_stealing_2b",
        "caught_stealing_3b",
        "caught_stealing_home",
        "pickoff_1b",
        "pickoff_2b",
        "pickoff_3b",
        "pickoff_caught_stealing_2b",
        "pickoff_caught_stealing_3b",
        "pickoff_caught_stealing_home",
        "stolen_base_2b",
        "stolen_base_3b",
        "stolen_base_home",
        "other_out",
        "balk",
        "passed_ball",
        "wild_pitch",
        "defensive_indiff",
    }
)


def is_plate_appearance(play: dict[str, Any]) -> bool:
    """Does this play end a batter's turn at the plate?

    A sacrifice, a fielder's choice, a reached-on-error and a catcher's
    interference all do, and all are plate appearances. A runner being thrown out
    does not, however the feed types it.
    """
    result = play.get("result") or {}
    if result.get("type") != "atBat":
        return False
    return str(result.get("eventType", "")) not in RUNNER_EVENT_TYPES


def plate_appearances(plays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The plays that are plate appearances, in feed order."""
    return [p for p in plays if is_plate_appearance(p)]
