"""Tuesday's card for a Saturday, graded beside Saturday's own card.

The question it answers is whether betting the number days early pays: the same
model prices the same games twice, once on the Tuesday line and once on the
Saturday morning line, and both are graded against the same finals and the same
close. Sides bought both times isolate the line itself; sides bought only once
show what each card finds that the other does not.

Each card is also judged on what its price was worth at the close (the
closing no-vig probability against the price taken), and Tuesday's spreads face
one pre-registered bar: the sides the model put ``GAP_PTS`` or more off the
card's own line must show a positive average worth at the close over
``BAR_GAMES`` games. On 2026's opener boards (163 games, Oct 6 on) those sides
saw the line move +0.81 pts their way by the close, against +0.07 from the
Saturday 09:00 number, so the question is whether pricing early keeps that.

Measurement only: nothing here prices, tiers or refuses a bet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

from cfb_engine.audit.grade import LOSS, PUSH, WIN
from cfb_engine.audit.ledger import LedgerEntry
from cfb_engine.audit.priced import BUY_TIERS
from cfb_engine.market import keys
from cfb_engine.recommendations import Recommendation

SATURDAY = 5
GAP_PTS = 3.0  # pre-registered: model margin this far off the card's spread
BAR_GAMES = 150  # games before the bar is called


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
    close_ev: float | None = None  # mean EV of the price taken at the closing no-vig


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
        close_ev=_mean([e.clv_ev for e in rows if e.clv_ev is not None]),
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


def model_gap(rec: Recommendation) -> float | None:
    """Points the model expects the side to cover its own line by; spreads only."""
    if rec.market != "game_ats" or rec.line is None or rec.exp_margin is None:
        return None
    if rec.team_side not in ("home", "away"):
        return None
    margin = rec.exp_margin if rec.team_side == "home" else -rec.exp_margin
    return margin + rec.line


def gap_sides(
    cards: dict[str, list[Recommendation]], entries: list[LedgerEntry], gap: float = GAP_PTS
) -> list[LedgerEntry]:
    """Graded spread sides, bought or not, the card put ``gap`` points or more off its line."""
    keep = {
        (day, r.matchup, r.market, keys.side_of(r.selection))
        for day, recs in cards.items()
        for r in recs
        if (g := model_gap(r)) is not None and g >= gap
    }
    return [e for e in entries if _key(e) in keep]


@dataclass(frozen=True)
class CloseBar:
    label: str
    n: int  # sides with a timely close
    close_ev: float | None
    se: float | None

    @property
    def verdict(self) -> str:
        if self.n < BAR_GAMES:
            return f"needs {BAR_GAMES - self.n} more"
        return "pass" if self.close_ev is not None and self.close_ev > 0 else "fail"


def close_bar(label: str, rows: list[LedgerEntry]) -> CloseBar:
    vals = [e.clv_ev for e in rows if e.clv_ev is not None]
    mean = _mean(vals)
    se = None
    if mean is not None and len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        se = math.sqrt(var / len(vals))
    return CloseBar(label=label, n=len(vals), close_ev=mean, se=se)


def bars(
    early_cards: dict[str, list[Recommendation]],
    early_rows: list[LedgerEntry],
    card_cards: dict[str, list[Recommendation]],
    card_rows: list[LedgerEntry],
) -> list[CloseBar]:
    """Tuesday's pre-registered spread bar, and the same rule on Saturday's card."""
    days = set(early_cards)
    same = {d: recs for d, recs in card_cards.items() if d in days}
    label = f"spreads, model {GAP_PTS:g}+ pts off the line"
    return [
        close_bar(f"Tuesday {label}", gap_sides(early_cards, early_rows)),
        close_bar(
            f"Saturday {label}, same slates",
            gap_sides(same, [e for e in card_rows if e.date in days]),
        ),
    ]
