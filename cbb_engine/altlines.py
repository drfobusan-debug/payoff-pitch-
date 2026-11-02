"""Alternate spreads and totals beside each game the card bought (graded only).

At card time the books' alternate numbers are fetched for every game with a buy
on the spread or total, live or would-be (``alternate_spreads`` +
``alternate_totals``, two credits a game), into ``alt`` quote shards the card
never prices. The audit grades every alternate on the bought side against the
final, at the best DraftKings/BetMGM price, into ``alt_ledger.csv``; the audit
report tables points bought against points sold. Nothing here prices or buys.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path

from cbb_engine import audit, history
from cbb_engine.config import EXECUTION_BOOKS, SPORT
from cbb_engine.data import capture
from cbb_engine.data.capture import slate_date_of
from cbb_engine.data.espn import Game
from engine_runtime import altlines as rt
from engine_runtime import ledger, quotes
from engine_runtime.odds_client import OddsClient

PROVIDERS: tuple[str, ...] = ("alternate_spreads", "alternate_totals")
ALT: dict[str, str] = {"game_ats_alt": "game_ats", "game_total_alt": "game_total"}
FIELDS = ["date", "event_id", "market", "side", "main_line", "alt_line", "favour", "bought", "book", "american", "result", "pnl"]


CARDS_PER_DAY = 2  # the 11:30 and 16:30 card passes (scripts/cbb/macos/install_schedule.command)


def _utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def daily_cost(n_games: int, max_events: int, regions: int) -> int:
    """Ceiling on alternate-line credits a day: every game bought on every card."""
    return CARDS_PER_DAY * min(n_games, max_events) * len(PROVIDERS) * regions


def tracked(rows: Iterable[ledger.Row]) -> list[str]:
    """Events with a buy on the main spread or total, live or would-be."""
    return sorted({r.event_id for r in rows if r.is_buy and r.market in ALT.values()})


def capture_for(
    client: OddsClient, root: Path, day: str, pass_id: str, rows: Sequence[ledger.Row], *, max_events: int
) -> int:
    """Fetch and shard the alternates for the card's bought games; returns quotes written."""
    n = 0
    for eid in tracked(rows)[:max_events]:
        ev = client.event(eid, PROVIDERS, pass_id=pass_id)
        if ev is None:
            continue
        pid = f"{pass_id}_alt_{eid[:8]}"
        qs = capture.rows_from_events([ev], captured_at=quotes.now_utc(), pass_id=pid, pass_kind="alt")
        if quotes.write_shard(root, SPORT, day, qs, pass_id=pid, pass_kind="alt", extra={"event_id": eid}):
            n += len(qs)
    return n


def favour(market: str, side: str, main: float, alt: float) -> float:
    """Points the alternate gives the side over the main number (negative: sold)."""
    if market == "game_total_alt":
        return round(main - alt if side == "over" else alt - main, 1)
    return round(alt - main, 1)


def grade(
    root: Path,
    day: str,
    games: Sequence[Game],
    teams: dict[str, tuple[str, str]],
    lookback: int = audit.LOOKBACK_DAYS,
) -> list[dict[str, object]]:
    """Every alternate on a bought side whose game tipped on ``day``, graded on the final."""
    finals = [g for g in games if g.final]
    out: list[dict[str, object]] = []
    end = Date.fromisoformat(day)
    for back in range(lookback, -1, -1):
        card_day = (end - timedelta(days=back)).isoformat()
        alts = [
            q
            for q in quotes.read_day(root, SPORT, card_day)
            if q.pass_kind == "alt" and q.market in ALT and q.book in EXECUTION_BOOKS and q.line is not None
            and slate_date_of(q.tip_utc) == day
            and _utc(q.captured_at) < _utc(q.tip_utc)
        ]
        if not alts:
            continue
        # Every pass's own rows: the record keeps only a position's first pass, and a later card's
        # alternates belong to that later card's buy.
        buys = {
            (r.pass_id, r.event_id, r.market, r.side): r
            for r in ledger.read_day(root, SPORT, card_day)
            if r.bet_of_record and r.is_buy and r.market in ALT.values() and r.line is not None
        }
        best: dict[tuple[str, str, str, float], quotes.Quote] = {}
        mains: dict[tuple[str, str, str, float], ledger.Row] = {}
        for q in alts:
            card_pass = q.pass_id.split("_alt_")[0]
            main = buys.get((card_pass, q.event_id, ALT[q.market], q.side))
            if main is None or main.line is None or q.line is None:
                continue
            key = (q.event_id, q.market, q.side, q.line)
            cur = best.get(key)
            if cur is None or rt.best_price([((0,), cur.american, cur), ((0,), q.american, q)])[0] is q:
                best[key], mains[key] = q, main
        for key, q in sorted(best.items()):
            main = mains[key]
            assert main.line is not None and q.line is not None
            f = favour(q.market, q.side, main.line, q.line)
            ha = teams.get(q.event_id)
            game = history.match_game(ha[0], ha[1], finals) if ha else None
            if not f or game is None or game.final is None:
                continue
            row = replace(main, line=q.line, american=q.american, book=q.book)
            res = audit.outcome(row, game.final)
            if not res:
                continue
            out.append(
                {
                    "date": day,
                    "event_id": q.event_id,
                    "market": q.market,
                    "side": q.side,
                    "main_line": main.line,
                    "alt_line": q.line,
                    "favour": f,
                    "bought": main.status == "live",
                    "book": q.book,
                    "american": q.american,
                    "result": res,
                    "pnl": audit.pnl(row, res) or 0.0,
                }
            )
    return out


def save_day(path: Path, day: str, rows: list[dict[str, object]]) -> int:
    """Replace one day's rows in the alt ledger; other days are kept as written."""
    kept: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r["date"] != day]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows)
    return len(rows)


def alts(path: Path, until: str | None = None) -> list[rt.Alt]:
    """The alt ledger as records for the shared table; ``beside_buy`` = beside a live buy."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if until is None or r["date"] <= until]
    return [
        rt.Alt(
            game=r["event_id"],
            kind="spread" if r["market"] == "game_ats_alt" else "total",
            favour=float(r["favour"]),
            beside_buy=r["bought"] == "True",
            outcome=r["result"],
            pnl=float(r["pnl"]),
        )
        for r in rows
    ]


__all__ = ["ALT", "PROVIDERS", "alts", "capture_for", "daily_cost", "favour", "grade", "save_day", "tracked"]
