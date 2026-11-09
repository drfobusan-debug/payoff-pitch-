"""Where the spread sits, as three graded-only splits of the spread buys.

Through 2026-10-17 the engine's 139 graded ATS buys returned +18.7% on dogs
(n=91) and +18.6% on favourites (n=46), but the line ranges inside each did not
agree: dogs of +14.5 or more went 21-9 (+35.2%, se 16.4) while the unbet dogs
there lost 17.0% on 133 rows, while dogs of +7.5 to +14 went 9-12 (-16.1%) and
favourites of -3.5 to -7 went 9-11 (-11.2%), each about 20 rows with one good
half. They were three of about twenty cuts tried, so none acts on a bet: the two
losing ranges accrue as candidate screens and the big dog as a candidate
upgrade, and the three probation tests decide whether any ships.

Through 2026-10-31 the dog buys split by venue went 46-28-1 (+15.7u) for the
listed road team and 26-20-3 (+4.6u) for the listed home team, both worth less
than their price at the close (-1.1% and -2.9%). Road and home dogs accrue as two
more candidate upgrades so the split is graded rather than argued; neither acts.
"""

from __future__ import annotations

BIG_DOG_RULE = "ats_upgrade_dog_14.5_plus"
MID_DOG_RULE = "ats_refuse_dog_7.5_to_14"
MID_FAV_RULE = "ats_refuse_fav_3.5_to_7"
ROAD_DOG_RULE = "ats_upgrade_road_dog"
HOME_DOG_RULE = "ats_upgrade_home_dog"


def _spread(market: str, line: float | None) -> float | None:
    return line if market == "game_ats" and line is not None else None


def big_dog(market: str, line: float | None) -> bool:
    """Getting 14.5 points or more."""
    s = _spread(market, line)
    return s is not None and s >= 14.5


def mid_dog(market: str, line: float | None) -> bool:
    """Getting 7.5 to 14 points."""
    s = _spread(market, line)
    return s is not None and 7.5 <= s <= 14.0


def mid_favourite(market: str, line: float | None) -> bool:
    """Laying 3.5 to 7 points."""
    s = _spread(market, line)
    return s is not None and -7.0 <= s <= -3.5


def venue(matchup: str, selection: str) -> str | None:
    """``"home"`` or ``"away"`` for the listed side a spread selection backs."""
    away, sep, home = matchup.partition(" @ ")
    team = selection.rsplit(" ", 1)[0] if selection else ""
    if not sep or not team:
        return None
    if team == home:
        return "home"
    if team == away:
        return "away"
    return None


def _dog(market: str, line: float | None) -> bool:
    s = _spread(market, line)
    return s is not None and s > 0


def road_dog(market: str, line: float | None, matchup: str, selection: str) -> bool:
    """Getting points as the listed road team."""
    return _dog(market, line) and venue(matchup, selection) == "away"


def home_dog(market: str, line: float | None, matchup: str, selection: str) -> bool:
    """Getting points as the listed home team."""
    return _dog(market, line) and venue(matchup, selection) == "home"
