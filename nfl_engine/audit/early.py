"""Tuesday's board priced as a record, graded beside the Thursday/Sunday card.

The question it answers is CFB's (``cfb_engine.audit.early``): does taking the
number days early beat waiting for the card? ``nfl-engine open --early`` prices
Tuesday's opening board with the same model the card uses and writes the rows to
their own ledger; ``close`` and ``grade`` stamp and settle them exactly like the
card's, so both are judged against the same finals and the same close.

The comparison is paired on the model's lean, one per game-market (the side it
rates above the no-vig price, as in ``scripts.nfl.ledger_audit``): the same side
on the same game, once at Tuesday's price and once at the card's, each scored by
its EV under the closing no-vig probability. Pre-registered bar: after
``BAR_GAMES`` paired spread leans with a paired close, Tuesday is "better" only
if the mean paired gain is above zero by more than one SE, "worse" if below by
more than one SE, otherwise not answerable at that n.

Measurement only: nothing here prices, tiers or refuses a bet, and no card,
email or ledger the bets are read from ever sees these rows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nfl_engine.audit.ledger import ENGINE, PUSH, LedgerEntry, load_ledger
from nfl_engine.market.ev import MONEYLINE, SPREAD, TOTAL
from nfl_engine.market.screens import Tier

EARLY_LEDGER_NAME = "nfl_early_ledger.csv"
BAR_MARKET = SPREAD  # pre-registered: the bar is called on spreads
BAR_GAMES = 150  # paired leans before the bar is called
MARKETS = (SPREAD, TOTAL, MONEYLINE)


def early_ledger_path(root: Path) -> Path:
    return root / EARLY_LEDGER_NAME


def has_week(path: Path, season: int, week: int) -> bool:
    """Write-once: the point is the number as it stood that morning."""
    return any((e.season, e.week) == (season, week) for e in load_ledger(path))


def _utc(stamp: str) -> datetime | None:
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def pregame(entry: LedgerEntry) -> bool:
    taken, kickoff = _utc(entry.captured_at), _utc(entry.kickoff_utc)
    return taken is None or kickoff is None or taken < kickoff


def _rank(entry: LedgerEntry) -> tuple[int, float]:
    fair = entry.fair_prob if entry.fair_prob is not None else 0.0
    rung = 0.0 if entry.market == MONEYLINE else -abs(fair - 0.5)
    return entry.paired_books, rung


def _usable(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    return [
        e
        for e in entries
        if e.source == ENGINE and e.fair_prob is not None and e.result != PUSH and pregame(e)
    ]


GameKey = tuple[int, int, str, str]


def leans(entries: list[LedgerEntry]) -> dict[GameKey, LedgerEntry]:
    """One lean per game-market from the first pass that priced it."""
    groups: dict[GameKey, list[LedgerEntry]] = defaultdict(list)
    for e in _usable(entries):
        groups[(e.season, e.week, e.matchup, e.market)].append(e)
    out: dict[GameKey, LedgerEntry] = {}
    for key, rows in groups.items():
        first = min(e.captured_at for e in rows)
        card = [e for e in rows if e.captured_at == first]
        positive = [e for e in card if e.fair_prob is not None and e.model_prob > e.fair_prob]
        if positive:
            out[key] = max(positive, key=_rank)
    return out


def same_side(entries: list[LedgerEntry], lean: LedgerEntry) -> LedgerEntry | None:
    """The card's own quote on the side Tuesday leaned, from its first pass."""
    rows = [
        e
        for e in _usable(entries)
        if (e.season, e.week, e.matchup, e.market, e.side)
        == (lean.season, lean.week, lean.matchup, lean.market, lean.side)
    ]
    if not rows:
        return None
    first = min(e.captured_at for e in rows)
    return max((e for e in rows if e.captured_at == first), key=_rank)


def _closed(entry: LedgerEntry) -> bool:
    return entry.clv_ev is not None and bool(entry.close_captured_at)


@dataclass(frozen=True)
class Pair:
    early: LedgerEntry
    card: LedgerEntry

    @property
    def gain(self) -> float:
        """EV at the close Tuesday's price beat the card's price by, same side."""
        return (self.early.clv_ev or 0.0) - (self.card.clv_ev or 0.0)


def pairs(early: list[LedgerEntry], card: list[LedgerEntry], market: str) -> list[Pair]:
    out = []
    for key, lean in leans(early).items():
        if key[3] != market or not _closed(lean):
            continue
        twin = same_side(card, lean)
        if twin is not None and _closed(twin) and twin.captured_at > lean.captured_at:
            out.append(Pair(lean, twin))
    return out


def _mean_se(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


@dataclass(frozen=True)
class Bar:
    label: str
    n: int
    early_ev: float | None
    card_ev: float | None
    gain: float | None
    se: float | None
    bar: bool = False

    @property
    def verdict(self) -> str:
        if self.bar and self.n < BAR_GAMES:
            return f"needs {BAR_GAMES - self.n} more"
        if self.gain is None or self.se is None:
            return "n/a"
        if self.gain > self.se:
            return "Tuesday better" if self.bar else "Tuesday ahead by > 1 se"
        if self.gain < -self.se:
            return "Tuesday worse" if self.bar else "Tuesday behind by > 1 se"
        return "not answerable at this n" if self.bar else "within 1 se"


def bar(label: str, rows: list[Pair], *, preregistered: bool = False) -> Bar:
    gain, se = _mean_se([p.gain for p in rows])
    early_ev, _ = _mean_se([p.early.clv_ev or 0.0 for p in rows])
    card_ev, _ = _mean_se([p.card.clv_ev or 0.0 for p in rows])
    return Bar(label, len(rows), early_ev, card_ev, gain, se, preregistered)


def bars(early: list[LedgerEntry], card: list[LedgerEntry]) -> list[Bar]:
    """The pre-registered spread bar first, then the other markets for reading only."""
    return [
        bar(f"{m} leans, same side", pairs(early, card, m), preregistered=m == BAR_MARKET)
        for m in MARKETS
    ]


@dataclass(frozen=True)
class Record:
    label: str
    n: int
    wins: int
    losses: int
    units: float
    close_ev: float | None


def buys(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    return [e for e in entries if e.source == ENGINE and e.tier != Tier.PASS.value and e.result]


def record(label: str, rows: list[LedgerEntry]) -> Record:
    ev, _ = _mean_se([e.clv_ev for e in rows if _closed(e) and e.clv_ev is not None])
    return Record(
        label,
        len(rows),
        sum(e.result == "win" for e in rows),
        sum(e.result == "loss" for e in rows),
        round(sum(e.pnl for e in rows), 2),
        ev,
    )


def report(early: list[LedgerEntry], card: list[LedgerEntry]) -> list[str]:
    weeks = {(e.season, e.week) for e in early}
    card = [e for e in card if (e.season, e.week) in weeks]
    out = [f"Tuesday early card vs the card, {len(weeks)} week(s)", ""]
    for r in (
        record("Tuesday buys", buys(early)),
        record("Card buys, same weeks", buys(card)),
    ):
        ev = f"{r.close_ev:+.2%}" if r.close_ev is not None else "n/a"
        out.append(
            f"  {r.label:26s} n {r.n:4d}  {r.wins}-{r.losses}  {r.units:+.2f}u  EV@close {ev}"
        )
    out += ["", f"Paired leans (pre-registered: {BAR_MARKET}, {BAR_GAMES} games, 1 se)"]
    for b in bars(early, card):
        cells = [f"{v:+.2%}" if v is not None else "n/a" for v in (b.early_ev, b.card_ev, b.gain)]
        se = f"± {b.se:.2%}" if b.se is not None else ""
        out.append(
            f"  {b.label:26s} n {b.n:4d}  Tue {cells[0]}  card {cells[1]}  "
            f"gain {cells[2]} {se}  {b.verdict}"
        )
    return out
