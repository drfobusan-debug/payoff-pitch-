"""Team-level groups for the CBB audit: how well the closing line called each kind of game.

Graded only; this changes no price, gate or buy. Each game counts once, grouped on what
was known before tip (``bands.context``): each team's adjusted-tempo third from the Torvik
team_results archived on or before the slate day, and conference membership (``bands.POWER``).

Per group and market (spread, total):

* close miss: mean |result - close|, the close being the median line across books in the
  event's last close pass before tip and within 45 min (``audit._closes``). Spread result
  is the favourite's cover margin; total result is final points less the closing total.
* bias: mean (result - close); positive means favourites / Overs beat the close.
* CLV and close EV on the favourite / Over rows (the underdog / Under rows mirror them).

A group is read only at ``MIN_N`` games, and a difference is called only past ``CALL_SE``.
Roster continuity is not grouped: it needs a returning-minutes source.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from cbb_engine import audit, history
from cbb_engine.bands import GameCtx, main_lines, main_rows
from cbb_engine.data import espn
from engine_common.bands import CALL_SE, MIN_N
from engine_runtime import ledger, quotes

MARKETS = (("spread", "game_ats"), ("total", "game_total"))
CONTINUITY_NOTE = "Roster continuity: not grouped yet; needs a returning-minutes source."


@dataclass(frozen=True)
class Called:
    """A finished game's closing consensus lines and final, home perspective."""

    event_id: str
    home_spread: float | None
    total: float | None
    home_margin: int
    points: int

    def result(self, market: str) -> float | None:
        """Result less the close: favourite cover margin (spread) or points over the total."""
        if market == "spread":
            h = self.home_spread
            if h is None or h == 0:
                return None
            return self.home_margin + h if h < 0 else -(self.home_margin + h)
        return None if self.total is None else self.points - self.total


def _pairs(c: GameCtx, attr: str, a: object, b: object) -> bool:
    x, y = getattr(c.home, attr), getattr(c.away, attr)
    return {x, y} == {a, b} if a != b else x == y == a


GROUPS: tuple[tuple[str, str, Callable[[GameCtx], bool]], ...] = (
    ("tempo", "both fast", lambda c: _pairs(c, "pace", "fast", "fast")),
    ("tempo", "both slow", lambda c: _pairs(c, "pace", "slow", "slow")),
    ("tempo", "fast vs slow", lambda c: _pairs(c, "pace", "fast", "slow")),
    ("conference", "Power vs Power", lambda c: _pairs(c, "power", True, True)),
    ("conference", "Power vs mid-major", lambda c: _pairs(c, "power", True, False)),
    ("conference", "mid-major vs mid-major", lambda c: _pairs(c, "power", False, False)),
)


def called(
    shards: Iterable[quotes.Quote], teams: Mapping[str, tuple[str, str]], games: Sequence[espn.Game]
) -> dict[str, Called]:
    """Every event with a timely close and a matched ESPN final."""
    closes, _ = audit._closes(shards)
    out: dict[str, Called] = {}
    for eid, qs in closes.items():
        ha = teams.get(eid)
        hit = history.scores(ha[0], ha[1], games) if ha else None
        if hit is None:
            continue
        (away, home), _ = hit
        lines = main_lines(qs)
        out[eid] = Called(eid, lines.get(("game_ats", "home")), lines.get(("game_total", "over")), home - away, home + away)
    return out


def _mean_se(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def _decimal(american: float) -> float:
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


@dataclass(frozen=True)
class GroupRead:
    dimension: str
    group: str
    market: str
    games: int
    miss: float
    miss_se: float
    vs_rest: float
    vs_rest_se: float
    bias: float
    bias_se: float
    clv_n: int
    clv: float
    clv_se: float
    close_ev: float
    close_ev_se: float

    @property
    def verdict(self) -> str:
        if self.games < 2:
            return "no sample"
        if self.games < MIN_N:
            return f"underpowered ({self.games}/{MIN_N} games)"
        lean = "favourites" if self.market == "spread" else "Overs"
        other = "underdogs" if self.market == "spread" else "Unders"
        if self.vs_rest_se and self.vs_rest > CALL_SE * self.vs_rest_se:
            return "close less accurate than the rest"
        if self.vs_rest_se and self.vs_rest < -CALL_SE * self.vs_rest_se:
            return "close more accurate than the rest"
        if self.bias_se and abs(self.bias) > CALL_SE * self.bias_se:
            return f"{lean if self.bias > 0 else other} beat the close"
        return "no difference"


def _lean_rows(rows: Iterable[ledger.Row], market: str) -> dict[str, list[ledger.Row]]:
    """Per event, the favourite (spread) or Over (total) row(s) with a timely close."""
    out: dict[str, list[ledger.Row]] = {}
    for r in main_rows(rows):
        if r.close_status != "pre_tip" or r.close_consensus is None or r.american is None:
            continue
        if (market == "spread" and r.market == "game_ats" and r.line is not None and r.line < 0) or (
            market == "total" and r.market == "game_total" and r.side == "over"
        ):
            out.setdefault(r.event_id, []).append(r)
    return out


def read(
    rows: Iterable[ledger.Row], ctx: Mapping[str, GameCtx], games: Mapping[str, Called]
) -> list[GroupRead]:
    rows = list(rows)
    out: list[GroupRead] = []
    for market, _ in MARKETS:
        res = {e: x for e, g in games.items() if e in ctx and (x := g.result(market)) is not None}
        lean = _lean_rows(rows, market)
        for dim, name, hit in (("all", "all games", lambda c: True), *GROUPS):
            ins = [e for e in res if hit(ctx[e])]
            rest = [abs(res[e]) for e in res if e not in set(ins)]
            miss, miss_se = _mean_se([abs(res[e]) for e in ins])
            r_m, r_se = _mean_se(rest)
            bias, bias_se = _mean_se([res[e] for e in ins])
            clv = [sum(r.clv_consensus or 0.0 for r in lean[e]) / len(lean[e]) for e in ins if e in lean]
            ev = [sum((r.close_consensus or 0.0) * _decimal(float(r.american or 0)) - 1 for r in lean[e]) / len(lean[e])
                  for e in ins if e in lean]
            c, c_se = _mean_se(clv)
            v, v_se = _mean_se(ev)
            out.append(GroupRead(
                dim, name, market, len(ins), miss, miss_se,
                miss - r_m if rest and dim != "all" else 0.0,
                math.hypot(miss_se, r_se) if rest and dim != "all" else 0.0,
                bias, bias_se, len(clv), c, c_se, v, v_se,
            ))
    return out


HEAD = ("group", "market", "games", "close miss (pts)", "vs rest", "bias (pts)", "CLV fav/Over", "close EV fav/Over", "verdict")


def table(reads: Sequence[GroupRead]) -> list[tuple[str, ...]]:
    def pm(m: float, se: float, n: int, fmt: str = "+.2f") -> str:
        return f"{m:{fmt}} ± {se:.2f}" if n > 1 else "-"

    return [
        (
            g.group,
            g.market,
            str(g.games),
            f"{g.miss:.2f} ± {g.miss_se:.2f}" if g.games > 1 else "-",
            pm(g.vs_rest, g.vs_rest_se, g.games) if g.dimension != "all" else "-",
            pm(g.bias, g.bias_se, g.games),
            f"{pm(g.clv, g.clv_se, g.clv_n)} (n={g.clv_n})",
            f"{100 * g.close_ev:+.1f}¢ ± {100 * g.close_ev_se:.1f}" if g.clv_n > 1 else "-",
            g.verdict if g.dimension != "all" else "",
        )
        for g in reads
    ]


__all__ = ["CONTINUITY_NOTE", "Called", "GROUPS", "GroupRead", "HEAD", "called", "read", "table"]
