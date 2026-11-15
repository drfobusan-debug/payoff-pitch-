"""Novig's price on every side at card time, and each graded row re-valued at it.

Novig is a no-vig exchange. The question it answers is whether the card's sides,
taken there instead of at the card's book, would have been worth more at the
close. The card pulls Novig's board once per run (a separate ``bookmakers=novig``
request, 3 credits) and keeps the latest pregame quote per side in
``novig_<date>.json``. The audit then grades each row a second time at Novig's
own number and price, against the same close and the same final.

Nothing here is read by pricing: Novig never enters the fair price, the card or
the execution books. Quotes say nothing about depth, so a Novig price is what was
posted, not what a fill of a given size would have got.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from cfb_engine.audit import snapshot
from cfb_engine.audit.clv import ClosingQuote, closing_quotes, compute_clv
from cfb_engine.audit.grade import grade
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.shop import book_pnl
from cfb_engine.data.cfbd import GameResult
from cfb_engine.market.board import GameOdds
from cfb_engine.recommendations import Recommendation
from cfb_engine.schemas import Slate
from engine_common import novig as common

BOOK = "novig"
BAR_BUYS = 150
CALL_SE = 2.0


def only_book(odds: GameOdds, book: str = BOOK) -> GameOdds:
    """``odds`` with every other book's quotes dropped."""
    out = GameOdds(matchup=odds.matchup)
    for side, quotes in odds.ml.items():
        for q in quotes:
            if q.book == book:
                out.add_ml(side, q)
    for point, sides in odds.spreads.items():
        for side, quotes in sides.items():
            for q in quotes:
                if q.book == book:
                    out.add_spread(point, side, q)
    for line, sides in odds.totals.items():
        for side, quotes in sides.items():
            for q in quotes:
                if q.book == book:
                    out.add_total(line, side == "over", q)
    return out


def capture(
    slate: Slate, board: dict[str, GameOdds], now: datetime | None = None
) -> dict[str, ClosingQuote]:
    """Novig's quote per side of every game not yet kicked off, stamped.

    Empty-book placeholders (-9900, -99900, -100000 on Novig's board) are dropped.
    """
    quotes = closing_quotes(slate, {m: only_book(o) for m, o in board.items()}, now)
    return {k: q for k, q in quotes.items() if common.tradable(q.american, None)}


def save_capture(
    slate: Slate, board: dict[str, GameOdds], path: Path, now: datetime | None = None
) -> int:
    """Fold this pull into the slate's file, the latest quote per side winning."""
    fresh = capture(slate, board, now)
    if fresh:
        snapshot.save(snapshot.merge_last_wins(snapshot.load(path), fresh), path)
    return len(fresh)


def stamp(
    entry: LedgerEntry,
    rec: Recommendation,
    quotes: dict[str, ClosingQuote],
    closing: dict[str, ClosingQuote],
    final: GameResult | None,
    *,
    margin_sd: float,
    total_sd: float,
) -> bool:
    """Grade one ledger row at Novig's number and price; False if Novig had none."""
    q = quotes.get(snapshot.key(entry.matchup, entry.market, entry.selection))
    if q is None:
        return False
    entry.novig_line, entry.novig_odds = q.line, q.american
    if final is not None:
        result = grade(replace(rec, line=q.line), final)
        if result is not None:
            entry.novig_result = result
            entry.novig_pnl = book_pnl(result, q.american)
    revalue(entry, closing, rec.side, margin_sd=margin_sd, total_sd=total_sd)
    return True


def revalue(
    entry: LedgerEntry,
    closing: dict[str, ClosingQuote],
    side: str | None,
    *,
    margin_sd: float,
    total_sd: float,
) -> None:
    """Value the row's Novig price against ``closing`` (blank without a Novig quote)."""
    if entry.novig_odds is None:
        entry.novig_clv_ev = None
        return
    entry.novig_clv_ev = compute_clv(
        entry.matchup,
        entry.market,
        entry.selection,
        entry.novig_odds,
        None,
        closing,
        bet_line=entry.novig_line,
        side=side,
        margin_sd=margin_sd,
        total_sd=total_sd,
    ).clv_ev


@dataclass(frozen=True)
class Compare:
    """The same rows valued at the card's book and at Novig, against one close."""

    label: str
    n: int
    of: int  # rows in the group, with or without a Novig quote
    book_ev: float | None
    novig_ev: float | None
    gain: float | None
    gain_se: float | None
    book_roi: float | None
    novig_roi: float | None
    graded: int

    @property
    def verdict(self) -> str:
        if self.gain is None or self.gain_se is None:
            return "no Novig quotes yet"
        if self.n < BAR_BUYS:
            return f"needs {BAR_BUYS - self.n} more"
        if abs(self.gain) <= CALL_SE * self.gain_se:
            return f"within {CALL_SE:g} se"
        return f"Novig {'ahead' if self.gain > 0 else 'behind'} by > {CALL_SE:g} se"


def _clustered(rows: list[tuple[str, float]]) -> tuple[float, float | None]:
    """Mean and its standard error with rows of one game moving together."""
    vals = [v for _, v in rows]
    mean = sum(vals) / len(vals)
    games: dict[str, float] = defaultdict(float)
    for g, v in rows:
        games[g] += v - mean
    if len(games) < 2:
        return mean, None
    k = len(games)
    return mean, math.sqrt(k / (k - 1) * sum(s * s for s in games.values())) / len(vals)


def _game(e: LedgerEntry) -> str:
    return f"{e.date}|{e.matchup}"


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def compare_rows(label: str, rows: list[LedgerEntry]) -> Compare:
    paired = [e for e in rows if e.clv_ev is not None and e.novig_clv_ev is not None]
    graded = [e for e in rows if e.novig_result is not None and e.novig_pnl is not None]
    gain, se = (
        _clustered([(_game(e), (e.novig_clv_ev or 0.0) - (e.clv_ev or 0.0)) for e in paired])
        if paired
        else (None, None)
    )
    return Compare(
        label,
        len(paired),
        len(rows),
        _mean([e.clv_ev or 0.0 for e in paired]),
        _mean([e.novig_clv_ev or 0.0 for e in paired]),
        gain,
        se,
        _mean([e.pnl for e in graded]),
        _mean([e.novig_pnl or 0.0 for e in graded]),
        len(graded),
    )


def compare(entries: list[LedgerEntry]) -> list[Compare]:
    """Buys (the pre-registered row), by market, and every graded side."""
    buys = [e for e in entries if e.tier != "Pass"]
    out = [compare_rows("Buys", buys)]
    for market, name in (
        ("game_ats", "spreads"),
        ("game_ml", "moneylines"),
        ("game_total", "totals"),
    ):
        out.append(compare_rows(f"Buys, {name}", [e for e in buys if e.market == market]))
    out.append(compare_rows("Every graded side", entries))
    return out


NOTE = (
    f"Each row is valued twice against the same timely close: at the card's book and at "
    f"Novig's quote on the same side from the card's last pull (its own number, moved to the "
    f"close's number at the margin or total slope). Pre-registered: Novig improves EV at the close "
    f"if the buys' gain is beyond {CALL_SE:g} SE over {BAR_BUYS} buys with a Novig quote. Quotes "
    f"carry no depth, so a fill of any size may not have been there. Graded only; nothing is bought "
    f"at Novig."
)
