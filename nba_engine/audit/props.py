"""Graded-only prop reads: the Under at the main line, players watched by name, exec-book coverage.

Nothing here prices, gates or buys. Each read takes one row per player-game at
the main line (the line whose consensus no-vig Over is nearest a coin flip) and
only rows a DraftKings/BetMGM price was recorded for, so the units are what the
card could actually have bet. Every player in a game moves with that game's
pace and minutes, so the SE is taken over per-game sums (``BandRead``).

The watchlist is a default set of (player, market) pairs; ``NBAE_PROP_WATCH``
replaces it with ``Name:market,Name:market`` (markets ``pl_pts``, ``pl_3pm``,
``pl_reb``, ``pl_ast``, ``pl_pra``).
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from engine_common.bands import CALL_SE, MIN_N, BandRead
from engine_common.promotion import close_roi
from nba_engine.audit.ledger import LedgerRow, timely_close
from nba_engine.data.boxes import norm_name

PROPS = ("pl_pts", "pl_3pm", "pl_reb", "pl_ast", "pl_pra")
LABEL = {"pl_pts": "PTS", "pl_3pm": "3PM", "pl_reb": "REB", "pl_ast": "AST", "pl_pra": "PRA"}
WATCH: tuple[tuple[str, str], ...] = (
    ("Cade Cunningham", "pl_reb"),
    ("Tristan da Silva", "pl_3pm"),
    ("Caleb Wilson", "pl_pra"),
)
LAST = 5  # games listed per watched player

PlayerGame = tuple[str, str, str, str]  # (slate, event, market, player)


def watchlist(raw: str | None = None) -> tuple[tuple[str, str], ...]:
    """``NBAE_PROP_WATCH`` if set (``Name:market,...``), else the default list."""
    raw = os.getenv("NBAE_PROP_WATCH") if raw is None else raw
    if not raw or not raw.strip():
        return WATCH
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        name, _, market = item.rpartition(":")
        market = market.strip()
        if name.strip() and market in PROPS:
            out.append((name.strip(), market))
    return tuple(out) or WATCH


def _payout(american: float) -> float:
    return american / 100.0 if american > 0 else 100.0 / -american


@dataclass
class Main:
    """One player-game at its main line: the Over and the Under rows, when posted."""

    key: PlayerGame
    line: float
    over: LedgerRow | None = None
    under: LedgerRow | None = None


def main_lines(rows: Iterable[LedgerRow]) -> list[Main]:
    """Per player-game, the line nearest a coin flip among bettable two-way rows, date order."""
    by_line: dict[PlayerGame, dict[float, dict[str, LedgerRow]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in rows:
        if r.market not in PROPS or not r.entity or r.line is None:
            continue
        if not r.book or r.american is None or r.fair is None or r.side not in ("over", "under"):
            continue
        by_line[(r.slate_date, r.event_id, r.market, r.entity)][r.line][r.side] = r
    out: list[Main] = []
    for key, lines in sorted(by_line.items()):
        paired = {ln: s for ln, s in lines.items() if "over" in s and "under" in s}
        if not paired:
            continue
        line = min(paired, key=lambda ln: (abs((paired[ln]["over"].fair or 0.5) - 0.5), ln))
        out.append(Main(key, line, paired[line]["over"], paired[line]["under"]))
    return out


def _band(name: str, mains: list[Main], side: str) -> BandRead:
    read = BandRead(name)
    games: dict[tuple[str, str], list[float]] = defaultdict(list)
    expected: dict[tuple[str, str], list[float]] = defaultdict(list)
    for m in mains:
        r = m.over if side == "over" else m.under
        if r is None or not r.graded or r.pnl is None or r.american is None or r.fair is None:
            continue
        exp = r.fair * _payout(r.american) - (1.0 - r.fair)
        read.pnls.append(r.pnl)
        read.expected.append(exp)
        if r.close_fair is not None and timely_close(r) and r.pnl != 0:
            read.close_rois.append(close_roi(r.close_fair, r.american))
        games[(m.key[0], m.key[1])].append(r.pnl)
        expected[(m.key[0], m.key[1])].append(exp)
    for g in games:
        read.games.append(sum(games[g]))
        read.expected_games.append(sum(expected[g]))
    return read


def under_bands(rows: Iterable[LedgerRow]) -> list[BandRead]:
    """The Under at the main line, every prop and each market; the Over beside it for scale."""
    mains = main_lines(rows)
    out = [
        _band("props Under (all five)", mains, "under"),
        _band("props Over (all five)", mains, "over"),
    ]
    for market in PROPS:
        sub = [m for m in mains if m.key[2] == market]
        out.append(_band(f"{LABEL[market]} Under", sub, "under"))
    return out


@dataclass
class Watched:
    player: str
    market: str
    mains: list[Main] = field(default_factory=list)
    voids: int = 0

    def side(self, side: str) -> BandRead:
        return _band(side, self.mains, side)


def watched(rows: Iterable[LedgerRow], watch: Iterable[tuple[str, str]]) -> list[Watched]:
    rows = list(rows)
    out: list[Watched] = []
    for player, market in watch:
        key = norm_name(player)
        mine = [r for r in rows if r.market == market and norm_name(r.entity) == key]
        w = Watched(player, market, main_lines(mine))
        w.voids = len({(r.slate_date, r.event_id) for r in mine if r.outcome == "void"})
        out.append(w)
    return out


def _result(m: Main) -> str:
    o = m.over
    if o is None or not o.outcome:
        return "-"
    return {"win": "Over", "loss": "Under", "push": "push", "void": "void"}.get(
        o.outcome, o.outcome
    )


def coverage(rows: Iterable[LedgerRow]) -> list[tuple[str, int, int, int, int]]:
    """Per market: rows, rows with a DK/BetMGM price, games/player-games, those with one."""
    n: dict[str, int] = defaultdict(int)
    priced: dict[str, int] = defaultdict(int)
    units: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    units_priced: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for r in rows:
        u = (r.slate_date, r.event_id, r.entity)
        n[r.market] += 1
        units[r.market].add(u)
        if r.book and r.american is not None:
            priced[r.market] += 1
            units_priced[r.market].add(u)
    return [(m, n[m], priced[m], len(units[m]), len(units_priced[m])) for m in sorted(n)]


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


def lines(rows: Iterable[LedgerRow], watch: Iterable[tuple[str, str]] | None = None) -> list[str]:
    rows = list(rows)
    out = [
        f"[props at the main line: graded only, flat 1u at the DK/BetMGM price; call at {MIN_N}+ games, "
        f"{CALL_SE:g} SE over the market's own expectation]"
    ]
    for b in under_bands(rows):
        out.append(_band_line(b.name, b))
    out += ["", "[watched players: main line each game, graded only]"]
    for w in watched(rows, watchlist() if watch is None else watch):
        head = f"  {w.player} {LABEL[w.market]}"
        if not w.mains:
            out.append(f"{head}: no DK/BetMGM main line graded yet ({w.voids} void)")
            continue
        avg = sum(m.line for m in w.mains) / len(w.mains)
        out.append(f"{head}: {len(w.mains)} games, avg line {avg:.1f}, {w.voids} void (DNP/absent)")
        out.append(_band_line("    Over", w.side("over")))
        out.append(_band_line("    Under", w.side("under")))
        recent = ", ".join(f"{m.key[0][5:]} {m.line:g} {_result(m)}" for m in w.mains[-LAST:])
        out.append(f"    last {min(LAST, len(w.mains))}: {recent}")
    out += [
        "",
        "[execution-book coverage: rows and games/player-games with a DK/BetMGM price]",
        f"  {'market':11} {'rows':>7} {'priced':>7} {'share':>6} {'units':>6} {'priced':>6} {'share':>6}",
    ]
    tot = [0, 0, 0, 0]
    for m, n, p, u, up in coverage(rows):
        out.append(f"  {m:11} {n:7} {p:7} {_pct(p, n)} {u:6} {up:6} {_pct(up, u)}")
        tot = [tot[0] + n, tot[1] + p, tot[2] + u, tot[3] + up]
    out.append(
        f"  {'all':11} {tot[0]:7} {tot[1]:7} {_pct(tot[1], tot[0])} {tot[2]:6} {tot[3]:6} {_pct(tot[3], tot[2])}"
    )
    return out


def _band_line(name: str, b: BandRead) -> str:
    se = f"{b.se:.1f}" if len(b.games) > 1 else "n/a"
    return (
        f"  {name:24} n {b.n:5d}  games {len(b.games):4d}  {b.record:>9}  units {b.units:+7.1f}"
        f"  ROI {100 * b.roi:+6.1f}%  vs market {b.excess:+6.1f} ± {se}  {b.price_read}  -> {b.verdict}"
    )


__all__ = [
    "LABEL",
    "PROPS",
    "WATCH",
    "Main",
    "Watched",
    "coverage",
    "lines",
    "main_lines",
    "under_bands",
    "watched",
    "watchlist",
]
