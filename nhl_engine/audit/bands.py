"""Graded-only market bands: flat 1u on every bet in a band, at the card's price.

A band is a slice of the board (home favourites, road teams on a back-to-back,
Unders when a team is on a back-to-back, ...). Every graded ledger row is
priced, not only buys, so a band reads "bet every one of these" at the price
the card had. Nothing here touches a price, a gate or a tier.

One row per game per band: the moneyline and 60-minute sides, the +-1.5 puck
line, and the full-game total at the line whose no-vig Over is nearest 50%.
Favourite/underdog comes from the row's no-vig ``consensus``. Two prop families
are read too, one row per player per line: goalie saves Unders and player
no-goal (goals Under 0.5). Rows on one game move together, so the SE and the
halves are taken over per-game sums.

A band earns a call only at >= ``MIN_N`` games: "wins" needs units above
``CALL_SE`` standard errors with both halves of the ledger (date order)
positive; "loses" is the mirror, a refusal candidate. Many bands are read at
once, so about one in twenty will sit past 2 SE by chance; the report says how
many were read.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from engine_common.bands import CALL_SE, MIN_N, BandRead
from nhl_engine.audit.grade import PUSH
from nhl_engine.audit.grade import pnl as payout
from nhl_engine.audit.ledger import LedgerRow
from nhl_engine.audit.screens import Context

SIDES = ("game_ml", "game_pl", "game_ml3")
PROPS = ("g_saves", "sk_g")
#: Starting-goalie GSAx/60 gap for the goalie lean: about the top fifth of the
#: home-minus-away gap in the 2023-25 close study (a default, not a fit).
GSAX_GAP = 0.17


def _b2b(r: LedgerRow, team: str, ctx: Context) -> bool:
    return f"{r.slate_date}|{team}" in ctx.b2b


def _opp(r: LedgerRow) -> str:
    return r.away if r.side == r.home else r.home


def main_rows(rows: Iterable[LedgerRow]) -> list[LedgerRow]:
    """Graded rows a band can hold: one per (game, market, side), main lines only."""
    seen: set[tuple[str, str, str, str]] = set()
    by_total: dict[tuple[str, str], list[LedgerRow]] = {}
    out: list[LedgerRow] = []
    for r in rows:
        if r.outcome is None:
            continue
        if r.entity:
            pk = (r.slate_date, r.matchup, r.market, f"{r.side}|{r.entity}|{r.line}")
            if r.market in PROPS and r.side.lower() == "under" and pk not in seen:
                seen.add(pk)
                out.append(r)
            continue
        k = (r.slate_date, r.matchup, r.market, r.side)
        if r.market == "game_total":
            by_total.setdefault((r.slate_date, r.matchup), []).append(r)
            continue
        if r.market not in SIDES or k in seen:
            continue
        if r.market == "game_pl" and (r.line is None or abs(r.line) != 1.5):
            continue
        seen.add(k)
        out.append(r)
    for game in by_total.values():
        overs = [r for r in game if r.side.lower() == "over"]
        if not overs:
            continue
        line = min(overs, key=lambda r: abs(r.consensus - 0.5)).line
        done: set[str] = set()
        for r in game:
            if r.line == line and r.side.lower() not in done:
                done.add(r.side.lower())
                out.append(r)
    return sorted(out, key=lambda r: (r.slate_date, r.matchup))


def _ml3_rank(rows: list[LedgerRow]) -> dict[tuple[str, str, str], int]:
    """0 = 60-minute favourite, 1 = the other team, 2 = the draw, by no-vig consensus."""
    games: dict[tuple[str, str], list[LedgerRow]] = {}
    for r in rows:
        if r.market == "game_ml3":
            games.setdefault((r.slate_date, r.matchup), []).append(r)
    out: dict[tuple[str, str, str], int] = {}
    for (night, matchup), rs in games.items():
        teams = sorted((r for r in rs if r.side in (r.home, r.away)), key=lambda r: -r.consensus)
        for i, r in enumerate(teams):
            out[(night, matchup, r.side)] = i
        for r in rs:
            if r.side not in (r.home, r.away):
                out[(night, matchup, r.side)] = 2
    return out


Hit = Callable[[LedgerRow, Context, int], bool]


def _side(market: str, f: Callable[[LedgerRow, Context], bool]) -> Hit:
    return lambda r, ctx, _rank: r.market == market and r.side in (r.home, r.away) and f(r, ctx)


def _total(side: str, f: Callable[[LedgerRow, Context], bool]) -> Hit:
    return lambda r, ctx, _rank: r.market == "game_total" and r.side.lower() == side and f(r, ctx)


def _ml3(rank: int, f: Callable[[LedgerRow, Context], bool]) -> Hit:
    return lambda r, ctx, rk: r.market == "game_ml3" and rk == rank and f(r, ctx)


def _prop(market: str, f: Callable[[LedgerRow], bool]) -> Hit:
    return lambda r, _ctx, _rank: (
        r.market == market and bool(r.entity) and r.side.lower() == "under" and f(r)
    )


def _no_goal(r: LedgerRow) -> bool:
    return r.line == 0.5


def _goalie_edge(r: LedgerRow) -> float | None:
    """The side's starter GSAx/60 minus the other starter's, as priced."""
    h, a = r.home_goalie_gsax60, r.away_goalie_gsax60
    if h is None or a is None:
        return None
    return h - a if r.side == r.home else a - h


def _any_b2b(r: LedgerRow, ctx: Context) -> bool:
    return _b2b(r, r.home, ctx) or _b2b(r, r.away, ctx)


def _side_bands(market: str, label: str) -> list[tuple[str, Hit]]:
    def fav(r: LedgerRow) -> bool:
        return r.consensus > 0.5

    def home(r: LedgerRow) -> bool:
        return r.side == r.home

    bands: list[tuple[str, Callable[[LedgerRow, Context], bool]]] = [
        ("home favourite", lambda r, c: home(r) and fav(r)),
        ("home underdog", lambda r, c: home(r) and not fav(r)),
        ("away favourite", lambda r, c: not home(r) and fav(r)),
        ("away underdog", lambda r, c: not home(r) and not fav(r)),
        ("team on a back-to-back", lambda r, c: _b2b(r, r.side, c)),
        (
            "rested team vs a back-to-back",
            lambda r, c: _b2b(r, _opp(r), c) and not _b2b(r, r.side, c),
        ),
        ("road team on a back-to-back", lambda r, c: not home(r) and _b2b(r, r.side, c)),
        (
            "lean: home team vs a road back-to-back",
            lambda r, c: home(r) and _b2b(r, r.away, c) and not _b2b(r, r.home, c),
        ),
        (
            f"lean (fade): starting goalie {GSAX_GAP:.2f}+ GSAx/60 better",
            lambda r, c: (e := _goalie_edge(r)) is not None and e >= GSAX_GAP,
        ),
    ]
    if market == "game_ml":
        bands += [
            ("favourite >= 65%", lambda r, c: r.consensus >= 0.65),
            ("near pick'em 45-55%", lambda r, c: 0.45 <= r.consensus < 0.55),
            ("underdog < 35%", lambda r, c: r.consensus < 0.35),
            ("lean: home favourite >= 65%", lambda r, c: home(r) and r.consensus >= 0.65),
        ]
    return [(f"{label} {name}", _side(market, f)) for name, f in bands]


BANDS: list[tuple[str, Hit]] = [
    *_side_bands("game_ml", "ML"),
    *_side_bands("game_pl", "Puck line"),
    ("Total Over", _total("over", lambda r, c: True)),
    ("Total Under", _total("under", lambda r, c: True)),
    ("Total Over, a team on a back-to-back", _total("over", _any_b2b)),
    ("Total Under, a team on a back-to-back", _total("under", _any_b2b)),
    ("60-min favourite", _ml3(0, lambda r, c: True)),
    ("60-min home favourite", _ml3(0, lambda r, c: r.side == r.home)),
    ("60-min away favourite", _ml3(0, lambda r, c: r.side == r.away)),
    ("60-min underdog", _ml3(1, lambda r, c: True)),
    ("60-min draw", _ml3(2, lambda r, c: True)),
    ("Goalie saves Under", _prop("g_saves", lambda r: True)),
    (
        "Goalie saves Under, sim 10+ pts above the market",
        _prop("g_saves", lambda r: r.model_prob - r.consensus >= 0.10),
    ),
    ("Player no goal (goals Under 0.5)", _prop("sk_g", _no_goal)),
    (
        "Player no goal, sim rates it a buy",
        _prop("sk_g", lambda r: _no_goal(r) and r.tier != "Pass"),
    ),
]


#: The 2023-25 close-study leans (none cleared |z| >= 3 in both seasons), graded
#: here like any band and printed on the slate PDF; never a gate or a buy.
LEANS: tuple[str, ...] = tuple(name for name, _ in BANDS if "lean" in name)
_HITS = dict(BANDS)


def tonight(rows: Iterable[LedgerRow], ctx: Context) -> dict[str, list[str]]:
    """Each lean's qualifying sides on an ungraded card: ML and the +-1.5 puck line."""
    seen: set[tuple[str, str, str]] = set()
    main: list[LedgerRow] = []
    for r in rows:
        if r.entity or r.market not in ("game_ml", "game_pl") or r.side not in (r.home, r.away):
            continue
        if r.market == "game_pl" and r.line not in (-1.5, 1.5):
            continue
        k = (r.matchup, r.market, r.side)
        if k not in seen:
            seen.add(k)
            main.append(r)
    out: dict[str, list[str]] = {}
    for name in LEANS:
        hit = _HITS[name]
        picks = [
            f"{r.side} {'ML' if r.market == 'game_ml' else f'{r.line:+g}'} ({r.matchup})"
            for r in main
            if hit(r, ctx, -1)
        ]
        if picks:
            out[name] = picks
    return out


def read_bands(rows: Iterable[LedgerRow], ctx: Context) -> list[BandRead]:
    main = main_rows(rows)
    rank = _ml3_rank(main)
    out: list[BandRead] = []
    for name, hit in BANDS:
        rs = [r for r in main if hit(r, ctx, rank.get((r.slate_date, r.matchup, r.side), -1))]
        per_game: dict[tuple[str, str], float] = {}
        exp_game: dict[tuple[str, str], float] = {}
        for r in rs:
            k = (r.slate_date, r.matchup)
            per_game[k] = per_game.get(k, 0.0) + (r.pnl or 0.0)
            exp_game[k] = exp_game.get(k, 0.0) + market_expected(r)
        out.append(
            BandRead(
                name,
                [r.pnl or 0.0 for r in rs],
                list(per_game.values()),
                [market_expected(r) for r in rs],
                list(exp_game.values()),
            )
        )
    return out


def market_expected(r: LedgerRow) -> float:
    """Flat 1u at the card's price, settled at the no-vig consensus: the vig a fair bet pays.

    A push returns the stake, and the consensus is conditional on no push, so a
    pushed row expects (and made) nothing."""
    if r.outcome == PUSH:
        return 0.0
    win = payout("win", r.american) or 0.0
    return r.consensus * win - (1.0 - r.consensus)


__all__ = [
    "BANDS",
    "CALL_SE",
    "LEANS",
    "MIN_N",
    "BandRead",
    "main_rows",
    "market_expected",
    "read_bands",
    "tonight",
]
