"""Every book's total, captured per pull, and each graded total re-graded at the
best number an allowed book offered and at Kalshi's ask.

The board snapshots keep one quote per side at the most-quoted number, so they
cannot say whether another book had 56 when the card priced 56.5. This module
keeps every book's number and price for each pregame pull (no extra Odds API
request: it reads the board already fetched) in an append-only
``totals_books_<date>.jsonl``, one line per game per pull.

The audit then grades each total twice more against the real final:

* **best number** -- among ``CFBE_SHOP_BOOKS`` (empty: every book), the quote
  with the highest EV, a different number valued by moving the card's no-vig
  probability along a normal of the model's total SD;
* **Kalshi** -- the nearest rung of the game's Over ladder, bought at the ask
  (Over: Yes ask; Under: 1 - Yes bid) plus the taker fee 0.07 x P x (1 - P).
  The bid and open interest are recorded too, but a resting order's fill
  cannot be graded from quotes, so only the ask is scored.

Nothing here is read by pricing; the card is unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

from cfb_engine.audit.grade import LOSS, PUSH, WIN
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.data.kalshi import Ladder, match
from cfb_engine.market import gradedonly
from cfb_engine.market.board import GameOdds
from cfb_engine.market.odds import american_to_decimal
from cfb_engine.schemas import Slate

log = logging.getLogger(__name__)

CARD, CLOSE, OPENER = "card", "close", "opener"
KALSHI_TAKER_FEE = 0.07
_N = NormalDist()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _kick(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def capture_rows(
    slate: Slate,
    board: dict[str, GameOdds],
    source: str,
    *,
    pulled: datetime | None = None,
    ladders: list[Ladder] | None = None,
) -> list[dict]:
    """One record per pregame game: every book's (line, side, price), plus Kalshi."""
    pulled = pulled or _now()
    rows: list[dict] = []
    for game in slate.games:
        kick = _kick(game.commence_time_utc)
        if kick is not None and kick <= pulled:
            continue
        odds = board.get(game.matchup())
        if odds is None or not odds.totals:
            continue
        quotes = [
            [q.book, line, side, q.american]
            for line, sides in sorted(odds.totals.items())
            for side in ("over", "under")
            for q in sides.get(side, [])
        ]
        row: dict = {
            "pulled": pulled.isoformat(timespec="seconds"),
            "source": source,
            "matchup": game.matchup(),
            "kickoff": game.commence_time_utc,
            "quotes": quotes,
        }
        if ladders is not None:
            lad = match(game, ladders)
            if lad is not None:
                row["kalshi"] = [
                    [s.strike, s.yes_bid, s.yes_ask, s.open_interest] for s in lad.strikes
                ]
        rows.append(row)
    return rows


def append(rows: Iterable[dict], path: Path) -> int:
    lines = [json.dumps(r, sort_keys=True) for r in rows]
    if not lines:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.writelines(f"{line}\n" for line in lines)
    return len(lines)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def card_pull(rows: list[dict], matchup: str) -> dict | None:
    """The latest pregame pull the card itself made for ``matchup``."""
    mine = [r for r in rows if r.get("matchup") == matchup and r.get("source") == CARD]
    return max(mine, key=lambda r: r["pulled"]) if mine else None


@dataclass(frozen=True)
class Offer:
    book: str
    line: float
    american: float
    ev: float


def side_prob(side: str, line: float, card_line: float, fair: float, sd: float) -> float:
    """Win probability of ``side`` at ``line``, given its no-vig ``fair`` at the card's number."""
    fair = min(max(fair, 1e-4), 1 - 1e-4)
    p_over = fair if side == "over" else 1 - fair
    mu = card_line + sd * _N.inv_cdf(p_over)
    over = 1 - _N.cdf((line - mu) / sd)
    return over if side == "over" else 1 - over


def best_offer(
    quotes: list[list],
    side: str,
    card_line: float,
    fair: float,
    sd: float,
    books: tuple[str, ...] | None,
) -> Offer | None:
    best: Offer | None = None
    for book, line, s, american in quotes:
        if s != side or (books and book not in books):
            continue
        p = side_prob(side, float(line), card_line, fair, sd)
        ev = p * american_to_decimal(float(american)) - 1
        if best is None or ev > best.ev + 1e-12:
            best = Offer(book, float(line), float(american), ev)
    return best


def ou(total: float, line: float, side: str) -> str:
    if total == line:
        return PUSH
    return WIN if (total > line) == (side == "over") else LOSS


def kalshi_cost(ask: float) -> float:
    """Price plus taker fee per $1 contract."""
    return ask + KALSHI_TAKER_FEE * ask * (1 - ask)


@dataclass(frozen=True)
class KalshiFill:
    strike: float
    bid: float  # what the side could be sold at
    ask: float  # what the side costs
    open_interest: float


def kalshi_side(ladder: list[list], side: str, card_line: float) -> KalshiFill | None:
    """The rung nearest the card's number, priced for ``side``."""
    rungs = [r for r in ladder if 0 < float(r[1]) < float(r[2]) < 1]
    if not rungs:
        return None
    strike, bid, ask, oi = min(rungs, key=lambda r: (abs(float(r[0]) - card_line), float(r[0])))
    if side == "over":
        return KalshiFill(float(strike), float(bid), float(ask), float(oi))
    return KalshiFill(float(strike), round(1 - float(ask), 4), round(1 - float(bid), 4), float(oi))


def book_pnl(result: str, american: float) -> float:
    if result == WIN:
        return round(american_to_decimal(american) - 1, 4)
    return -1.0 if result == LOSS else 0.0


def kalshi_pnl(result: str, ask: float) -> float:
    if result == WIN:
        return round(1 / kalshi_cost(ask) - 1, 4)
    return -1.0 if result == LOSS else 0.0


def stamp(
    entry: LedgerEntry,
    side: str | None,
    final_total: float | None,
    rows: list[dict],
    books: tuple[str, ...] | None,
    sd: float,
) -> bool:
    """Grade one totals ledger row at the best allowed number and at Kalshi's ask."""
    if entry.market != "game_total" or entry.line is None or side not in ("over", "under"):
        return False
    pull = card_pull(rows, entry.matchup)
    if pull is None or final_total is None:
        return False
    fair = entry.fair_prob if entry.fair_prob is not None else 0.5
    offer = best_offer(pull.get("quotes", []), side, entry.line, fair, sd, books)
    if offer is not None:
        entry.shop_book, entry.shop_line, entry.shop_odds = offer.book, offer.line, offer.american
        entry.shop_result = ou(final_total, offer.line, side)
        entry.shop_pnl = book_pnl(entry.shop_result, offer.american)
    fill = kalshi_side(pull.get("kalshi") or [], side, entry.line)
    if fill is not None:
        entry.kalshi_strike, entry.kalshi_bid = fill.strike, fill.bid
        entry.kalshi_ask, entry.kalshi_oi = fill.ask, fill.open_interest
        entry.kalshi_result = ou(final_total, fill.strike, side)
        entry.kalshi_pnl = kalshi_pnl(entry.kalshi_result, fill.ask)
    return offer is not None or fill is not None


@dataclass(frozen=True)
class Record:
    label: str
    n: int
    wins: int
    losses: int
    pushes: int
    units: float

    @property
    def roi(self) -> float:
        return self.units / self.n if self.n else 0.0


def _record(label: str, pairs: list[tuple[str, float]]) -> Record:
    return Record(
        label,
        len(pairs),
        sum(r == WIN for r, _ in pairs),
        sum(r == LOSS for r, _ in pairs),
        sum(r == PUSH for r, _ in pairs),
        round(sum(p for _, p in pairs), 2),
    )


def records(entries: list[LedgerEntry], graded_only: bool = True) -> list[Record]:
    """Card number vs best number vs Kalshi, on the same totals rows."""
    rows = [
        e
        for e in entries
        if e.market == "game_total"
        and e.shop_result
        and (not graded_only or gradedonly.is_gate(e.pass_gate))
    ]
    kal = [e for e in rows if e.kalshi_result]
    return [
        _record("Card number, best price", [(e.result, e.pnl) for e in rows]),
        _record("Best allowed number", [(e.shop_result or "", e.shop_pnl or 0.0) for e in rows]),
        _record("Card number (Kalshi rows)", [(e.result, e.pnl) for e in kal]),
        _record(
            "Kalshi at the ask + fee", [(e.kalshi_result or "", e.kalshi_pnl or 0.0) for e in kal]
        ),
    ]
