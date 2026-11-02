"""Tuesday's card for a Saturday, graded beside Saturday's own card.

The question it answers is whether betting the number days early pays: the same
model prices the same games twice, once on the Tuesday line and once on the
Saturday morning line, and both are graded against the same finals and the same
close. Sides bought both times isolate the line itself; sides bought only once
show what each card finds that the other does not.

Measurement only: nothing here prices, tiers or refuses a bet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

from cfb_engine.audit.grade import LOSS, PUSH, WIN
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.priced import BUY_TIERS
from cfb_engine.market import keys

SATURDAY = 5


def next_saturday(today: Date) -> Date:
    """The first Saturday after ``today`` (a week out when today is Saturday)."""
    return today + timedelta(days=(SATURDAY - today.weekday()) % 7 or 7)


@dataclass(frozen=True)
class Record:
    label: str
    n: int
    wins: int
    losses: int
    pushes: int
    units: float
    roi: float
    clv_pts: float | None  # mean points against the close, spread/total rows only
    pts_gained: float | None = None  # mean points Tuesday's number beat Saturday's


def _key(e: LedgerEntry) -> tuple[str, str, str, str]:
    return (e.date, e.matchup, e.market, keys.side_of(e.selection))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def record(label: str, rows: list[LedgerEntry], pts_gained: list[float] | None = None) -> Record:
    decided = [e for e in rows if e.result != PUSH]
    units = sum(e.pnl for e in rows)
    return Record(
        label=label,
        n=len(rows),
        wins=sum(e.result == WIN for e in rows),
        losses=sum(e.result == LOSS for e in rows),
        pushes=len(rows) - len(decided),
        units=round(units, 2),
        roi=units / len(decided) if decided else 0.0,
        clv_pts=_mean([e.clv_pts for e in rows if e.clv_pts is not None]),
        pts_gained=_mean(pts_gained) if pts_gained is not None else None,
    )


def points_gained(early: LedgerEntry, card: LedgerEntry) -> float | None:
    """Points the earlier number is better by, from the bettor's side; None on a moneyline."""
    if early.line is None or card.line is None:
        return None
    if early.market == "game_ats":
        return early.line - card.line
    if early.market == "game_total":
        over = early.selection.startswith("Over")
        return card.line - early.line if over else early.line - card.line
    return None


def compare(early: list[LedgerEntry], card: list[LedgerEntry]) -> list[Record]:
    """Tuesday's buys against Saturday's buys on the slates both cards priced."""
    days = {e.date for e in early}
    early_buys = [e for e in early if e.tier in BUY_TIERS]
    card_buys = [e for e in card if e.tier in BUY_TIERS and e.date in days]
    by_key = {_key(e): e for e in card_buys}
    early_keys = {_key(e) for e in early_buys}
    both = [(e, by_key[_key(e)]) for e in early_buys if _key(e) in by_key]
    gained = [g for e, c in both if (g := points_gained(e, c)) is not None]

    out = [
        record("Tuesday buys", early_buys),
        record("Saturday buys, same slates", card_buys),
    ]
    for market in ("game_ats", "game_total", "game_ml"):
        out.append(record(f"  Tuesday {market}", [e for e in early_buys if e.market == market]))
        out.append(record(f"  Saturday {market}", [e for e in card_buys if e.market == market]))
    out += [
        record("Bought both days, at Tuesday's number", [e for e, _ in both], gained),
        record("Bought both days, at Saturday's number", [c for _, c in both]),
        record("Tuesday only", [e for e in early_buys if _key(e) not in by_key]),
        record("Saturday only", [c for c in card_buys if _key(c) not in early_keys]),
    ]
    return out
