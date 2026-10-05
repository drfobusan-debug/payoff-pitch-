"""Settle one NBA selection against the ESPN final (plan §7 step 1).

Game markets include overtime; first-half markets use the first two quarters,
so they never see it. A two-way first-half moneyline tied at the half is a
push (DraftKings and BetMGM both refund it). A postponed or cancelled game
voids every row on it.

Props settle on the player's box line. A book's name for a player is matched
to the box exactly, then by a unique same-surname player with the same first
initial, then through ``ALIASES``. A matched player who did not play is void
(``dnp``); a player missing from a complete box was inactive and is void
(``absent``), as both books void a prop on a player who never took the floor.
The audit lists the ``absent`` names, so a spelling the matcher missed shows
up as a regular who is "absent" every night rather than as a hidden loss.
"""

from __future__ import annotations

from collections.abc import Callable

from nba_engine.data.boxes import norm_name
from nba_engine.schemas import GameResult, PlayerLine

WIN, LOSS, PUSH, VOID = "win", "loss", "push", "void"
DNP, ABSENT, NOT_PLAYED = "dnp", "absent", "not_played"
PROP_VALUE: dict[str, Callable[[PlayerLine], int]] = {
    "pl_pts": lambda p: p.points,
    "pl_3pm": lambda p: p.threes,
    "pl_reb": lambda p: p.rebounds,
    "pl_ast": lambda p: p.assists,
    "pl_pra": lambda p: p.pra,
}
# Book spelling -> ESPN spelling, both as ``norm_name`` keys, where neither the
# exact nor the surname-and-initial match can bridge them.
ALIASES: dict[str, str] = {
    "carlton carrington": "bub carrington",
}

Settled = tuple[str | None, str]  # (outcome, void reason)


def _ou(value: float, line: float, side: str) -> str | None:
    if side not in ("over", "under"):
        return None
    if value == line:
        return PUSH
    return WIN if (value > line) == (side == "over") else LOSS


def _team(side: str, game: GameResult, home: int, away: int) -> tuple[int, int] | None:
    if side == game.home:
        return home, away
    if side == game.away:
        return away, home
    return None


def find_player(entity: str, players: tuple[PlayerLine, ...]) -> PlayerLine | None:
    want = norm_name(entity)
    want = ALIASES.get(want, want)
    keys = [(norm_name(p.name), p) for p in players]
    exact = [p for k, p in keys if k == want]
    if exact:
        return exact[0]
    parts = want.split()
    if len(parts) < 2:
        return None
    near = [
        p for k, p in keys if len(k.split()) >= 2 and k.split()[-1] == parts[-1] and k[0] == want[0]
    ]
    return near[0] if len(near) == 1 else None


def settle_row(
    market: str, side: str, entity: str, line: float | None, game: GameResult
) -> Settled:
    """``(outcome, void_reason)``; outcome ``None`` means it cannot be graded yet."""
    if game.not_played:
        return VOID, NOT_PLAYED
    if not game.is_final:
        return None, ""
    if market.startswith("h1_"):
        home, away = game.h1_home, game.h1_away
    else:
        home, away = game.final_home, game.final_away

    if market in ("game_ml", "h1_ml"):
        pts = _team(side, game, home, away)
        if pts is None:
            return None, ""
        if pts[0] == pts[1]:
            return PUSH, ""
        return (WIN if pts[0] > pts[1] else LOSS), ""
    if market in ("game_ats", "h1_ats"):
        pts = _team(side, game, home, away)
        if pts is None or line is None:
            return None, ""
        adj = pts[0] - pts[1] + line
        return (PUSH if adj == 0 else (WIN if adj > 0 else LOSS)), ""
    if market in ("game_total", "h1_total"):
        return (None if line is None else _ou(float(home + away), line, side)), ""

    value_of = PROP_VALUE.get(market)
    if value_of is None or line is None:
        return None, ""
    if not game.players:
        return None, ""
    player = find_player(entity, game.players)
    if player is None:
        return VOID, ABSENT
    if player.dnp or player.minutes <= 0:
        return VOID, DNP
    return _ou(float(value_of(player)), line, side), ""


def settle(market: str, side: str, entity: str, line: float | None, game: GameResult) -> str | None:
    return settle_row(market, side, entity, line, game)[0]


def pnl(outcome: str | None, american: float, stake: float = 1.0) -> float | None:
    """Profit at ``american``: push and void refund the stake."""
    if outcome is None:
        return None
    if outcome in (PUSH, VOID):
        return 0.0
    if outcome == LOSS:
        return -stake
    return stake * (american / 100.0 if american > 0 else 100.0 / -american)


__all__ = [
    "ABSENT",
    "ALIASES",
    "DNP",
    "LOSS",
    "NOT_PLAYED",
    "PROP_VALUE",
    "PUSH",
    "VOID",
    "WIN",
    "find_player",
    "pnl",
    "settle",
    "settle_row",
]
