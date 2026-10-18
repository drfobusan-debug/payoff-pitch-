"""Who is playing, as two graded-only splits of the spread buys.

Through 2026-10-10 the engine's 117 graded ATS buys went 38-17 (+35.2% ROI) in
Power 4 vs Power 4 games and 19-16-2 (+4.9%) in G5 vs G5 games (z=1.5), and
30-10-1 (+44.6%) when an AP Top 25 team played against 41-34-1 (+6.6%) when none
did (z=2.2) -- two of about ten cuts tried. Neither split acts on a bet: the
G5-only spread accrues as a candidate screen and the ranked-team spread as a
candidate upgrade, so the three probation tests decide whether either ships.
"""

from __future__ import annotations

from cfb_engine.data.cfbd import GameResult
from cfb_engine.output.brief import GameBrief

G5_ONLY_RULE = "ats_refuse_g5_only"
TOP25_RULE = "ats_upgrade_top25_game"

POWER4 = frozenset({"SEC", "Big Ten", "Big 12", "ACC"})
_P4_INDEPENDENTS = frozenset({"Notre Dame"})


def team_tier(team: str, conference: str, division: str) -> str:
    """``P4``, ``G5`` or ``FCS``; ``""`` when the feed did not say."""
    if not division:
        return ""
    if division.lower() != "fbs":
        return "FCS"
    if conference in POWER4 or team in _P4_INDEPENDENTS:
        return "P4"
    return "G5"


def game_tier(home: str, away: str) -> str | None:
    """Both sides' tiers, sorted (``G5-P4``); None if either is unknown."""
    if not home or not away:
        return None
    return "-".join(sorted((home, away)))


def result_tier(res: GameResult) -> str | None:
    """:func:`game_tier` for a graded CFBD final."""
    return game_tier(
        team_tier(res.home, res.home_conference, res.home_division),
        team_tier(res.away, res.away_conference, res.away_division),
    )


def top25_game(brief: GameBrief | None) -> bool | None:
    """Whether either side was in the AP Top 25 the card was priced on."""
    if brief is None:
        return None
    return brief.home.poll_rank is not None or brief.away.poll_rank is not None


def g5_only(market: str, conf_tier: str | None) -> bool:
    return market == "game_ats" and conf_tier == "G5-G5"


def ranked_game(market: str, top25: bool | None) -> bool:
    return market == "game_ats" and top25 is True
