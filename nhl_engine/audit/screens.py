"""Graded-only refusal screens (injury/rest master plan, Phase 3).

Each screen names the buys it *would* have refused. Nothing here touches a
price, a gate or a tier: the audit reads the refused buys' graded P&L as units
saved (``-pnl``), with its SE and the two halves of the ledger in date order.
Tickets on one game win and lose together, so the SE and halves are taken over
per-game sums, not per-ticket P&L.
A screen earns a promotion call only at >= ``MIN_N`` affected buys with saved
units above one SE and both halves positive; promotion itself is a separate,
explicit change that removes the old rule it replaces.

Context the ledger row does not carry is passed in:

* ``b2b`` -- ``"slate|TEAM"`` for every team playing the second night of a
  back-to-back (it played the calendar day before);
* ``late_news`` -- ``"slate|TEAM"`` for every team with an out/doubtful/scratched
  skater first logged after the card was priced.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import timedelta

from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.data.availability import Availability

MIN_N = 100
GAME_MARKETS = frozenset(
    {
        "game_ml",
        "game_ml3",
        "game_pl",
        "game_pl_alt",
        "game_total",
        "game_total_alt",
        "team_total",
        "team_total_alt",
    }
)
SIDE_MARKETS = frozenset({"game_ml", "game_ml3", "game_pl", "game_pl_alt"})
LATE_STATUSES = frozenset({"out", "doubtful", "scratched"})


@dataclass(frozen=True)
class Context:
    b2b: frozenset[str] = frozenset()
    late_news: frozenset[str] = frozenset()


def _statuses(r: LedgerRow) -> tuple[str, str]:
    away, home = (r.goalie_status.split("/") + ["", ""])[:2]
    return away, home


def goalie_probable(r: LedgerRow, ctx: Context) -> bool:
    return r.market in GAME_MARKETS and "probable" in _statuses(r)


def b2b_goalie_unconfirmed(r: LedgerRow, ctx: Context) -> bool:
    if r.market not in GAME_MARKETS:
        return False
    away_st, home_st = _statuses(r)
    return any(
        f"{r.slate_date}|{team}" in ctx.b2b and st != "confirmed"
        for team, st in ((r.away, away_st), (r.home, home_st))
    )


def late_skater_news(r: LedgerRow, ctx: Context) -> bool:
    return r.market in GAME_MARKETS and any(
        f"{r.slate_date}|{t}" in ctx.late_news for t in (r.away, r.home)
    )


def road_b2b_vs_rested_home(r: LedgerRow, ctx: Context) -> bool:
    return (
        r.market in SIDE_MARKETS
        and r.side == r.away
        and f"{r.slate_date}|{r.away}" in ctx.b2b
        and f"{r.slate_date}|{r.home}" not in ctx.b2b
    )


SCREENS: dict[str, tuple[str, Callable[[LedgerRow, Context], bool]]] = {
    "goalie_probable": ("game buy with a 'probable' (not confirmed) goalie", goalie_probable),
    "b2b_goalie_unconfirmed": (
        "game buy where a team on a back-to-back has no confirmed goalie",
        b2b_goalie_unconfirmed,
    ),
    "late_skater_news": (
        "game buy where an out/doubtful/scratched skater was first logged after pricing",
        late_skater_news,
    ),
    "road_b2b_vs_rested_home": (
        "side buy on a road team on a back-to-back vs a rested home team",
        road_b2b_vs_rested_home,
    ),
}


@dataclass
class ScreenRead:
    name: str
    rule: str
    pnls: list[float] = field(default_factory=list)
    games: list[float] = field(default_factory=list)  # P&L summed per game, date order
    halves: tuple[float, float] = (0.0, 0.0)

    @property
    def n(self) -> int:
        return len(self.pnls)

    @property
    def saved(self) -> float:
        return -sum(self.pnls)

    @property
    def se(self) -> float:
        g = len(self.games)
        if g < 2:
            return float("nan")
        m = sum(self.games) / g
        var = sum((p - m) ** 2 for p in self.games) / (g - 1)
        return math.sqrt(var * g)

    @property
    def record(self) -> str:
        w = sum(p > 0 for p in self.pnls)
        losses = sum(p < 0 for p in self.pnls)
        return f"{w}-{losses}"

    @property
    def verdict(self) -> str:
        if self.n < MIN_N:
            return f"probation ({self.n}/{MIN_N})"
        if self.saved > self.se and min(self.halves) > 0:
            return "clears the bar: promote explicitly"
        return "fails: retire"


def read_screens(rows: Iterable[LedgerRow], ctx: Context) -> list[ScreenRead]:
    buys = sorted(
        (r for r in rows if r.is_buy and r.outcome is not None),
        key=lambda r: (r.slate_date, r.priced_at),
    )
    out: list[ScreenRead] = []
    for name, (rule, hit) in SCREENS.items():
        hits = [r for r in buys if hit(r, ctx)]
        per_game: dict[tuple[str, str], float] = {}
        for r in hits:
            k = (r.slate_date, r.matchup)
            per_game[k] = per_game.get(k, 0.0) + (r.pnl or 0.0)
        s = ScreenRead(name, rule, [r.pnl or 0.0 for r in hits], list(per_game.values()))
        half = len(s.games) // 2
        s.halves = (-sum(s.games[:half]), -sum(s.games[half:]))
        out.append(s)
    return out


def b2b_teams(nights: Iterable[str], played: Callable[[Date], set[str]]) -> frozenset[str]:
    """``slate|TEAM`` for teams that also played the day before; ``played(day)`` lists teams."""
    out: set[str] = set()
    for night in sorted(set(nights)):
        day = Date.fromisoformat(night)
        tonight, before = played(day), played(day - timedelta(days=1))
        out |= {f"{night}|{t}" for t in tonight & before}
    return frozenset(out)


def late_news_teams(rows: Iterable[LedgerRow], records: Iterable[Availability]) -> frozenset[str]:
    """``slate|TEAM`` with a skater first logged out/doubtful/scratched after that slate's card.

    "First" is across every slate passed in, so a long-term absence logged on an
    earlier night is not late news tonight.
    """
    priced: dict[str, str] = {}
    for r in rows:
        if r.priced_at and (r.slate_date not in priced or r.priced_at < priced[r.slate_date]):
            priced[r.slate_date] = r.priced_at
    first: dict[tuple[str, str], Availability] = {}
    for a in records:
        if a.status not in LATE_STATUSES or a.role in ("G1", "G2"):
            continue
        k = (a.team, str(a.player_id or a.name))
        if k not in first or (a.slate, a.seen_at) < (first[k].slate, first[k].seen_at):
            first[k] = a
    return frozenset(
        f"{a.slate}|{a.team}"
        for a in first.values()
        if a.slate in priced and a.seen_at > priced[a.slate]
    )


__all__ = [
    "MIN_N",
    "SCREENS",
    "Context",
    "ScreenRead",
    "b2b_teams",
    "late_news_teams",
    "read_screens",
]
