"""Graduation: would the paper record justify going live? Display only.

The NFL engine is paper by construction -- there is no staking code -- so this
decides nothing. It applies the bar a deliberate go-live decision would have to
clear, the CFB probation tests turned around (CFB asks "are these buys losing?";
this asks "are they winning, and is it real?"), and prints the verdict in the
weekly audit so the question is answered by a rule rather than by a good month.

Only buys that are honest evidence count:

* priced before kickoff (``captured_at`` < ``kickoff_utc``) -- a price taken
  after the whistle is not a price anyone could have bet into;
* one per game-market side -- a Thursday game priced on two weekly cards, or the
  same side at two books, is one bet, not two;
* graded win/loss (pushes return nothing and would dilute the ROI).

Four tests, all required:

1. **Volume.** ``MIN_N`` such buys.
2. **Margin.** ROI above zero by more than one standard error.
3. **Consistency.** Both chronological halves profitable.
4. **CLV.** Mean CLV above zero by more than one standard error, over buys whose
   close was stamped within ``CLOSE_WINDOW`` of kickoff. A close taken hours
   early measures beating the morning line, so it does not count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import fmean, stdev

from nfl_engine.audit.ledger import ENGINE, LOSS, WIN, LedgerEntry, kickoff_moment
from nfl_engine.market.screens import Tier

MIN_N = 100
CLOSE_WINDOW = timedelta(minutes=60)
_BUY = frozenset({Tier.STRONG.value, Tier.MODERATE.value})

WATCHING = "WATCHING"
NOT_YET = "NOT YET"
MEETS_BAR = "MEETS BAR"


@dataclass(frozen=True)
class Graduation:
    n: int
    roi: float
    roi_se: float
    first_half: float
    second_half: float
    clv_n: int
    clv: float | None
    clv_se: float
    status: str
    failed: tuple[str, ...]


def _moment(stamp: str) -> datetime | None:
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def priced_before_kickoff(entry: LedgerEntry) -> bool:
    taken, kickoff = _moment(entry.captured_at), kickoff_moment(entry)
    return taken is not None and kickoff is not None and taken < kickoff


def has_true_close(entry: LedgerEntry) -> bool:
    """A CLV stamped from a price seen within ``CLOSE_WINDOW`` before kickoff."""
    seen, kickoff = _moment(entry.close_captured_at), kickoff_moment(entry)
    if entry.clv is None or seen is None or kickoff is None:
        return False
    return kickoff - CLOSE_WINDOW <= seen < kickoff


def eligible_buys(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Engine buys graded win/loss, priced pre-kickoff, one per game-market side."""
    rows = [
        e
        for e in entries
        if e.source == ENGINE
        and e.tier in _BUY
        and e.result in (WIN, LOSS)
        and e.odds is not None
        and priced_before_kickoff(e)
    ]
    first: dict[tuple[object, ...], LedgerEntry] = {}
    for e in sorted(rows, key=lambda e: e.captured_at):
        first.setdefault((e.season, e.kickoff_utc, e.matchup, e.market, e.side), e)
    return sorted(first.values(), key=lambda e: e.kickoff_utc)


def _mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) < 2:
        return values[0], 0.0
    return fmean(values), stdev(values) / len(values) ** 0.5


def _beyond_one_se(mean: float, se: float) -> bool:
    return mean > se if se > 0 else mean > 0


def graduation(entries: list[LedgerEntry], *, min_n: int = MIN_N) -> Graduation:
    buys = eligible_buys(entries)
    roi, roi_se = _mean_se([e.pnl for e in buys])
    mid = len(buys) // 2
    h1 = fmean([e.pnl for e in buys[:mid]]) if mid else 0.0
    h2 = fmean([e.pnl for e in buys[mid:]]) if buys[mid:] else 0.0
    clvs = [e.clv for e in buys if has_true_close(e) and e.clv is not None]
    clv, clv_se = _mean_se(clvs)

    failed: list[str] = []
    if len(buys) < min_n:
        failed.append(f"volume ({len(buys)} of {min_n})")
    if not _beyond_one_se(roi, roi_se):
        failed.append("margin")
    if not (h1 > 0 and h2 > 0):
        failed.append("consistency")
    if len(clvs) < min_n or not _beyond_one_se(clv, clv_se):
        failed.append("clv")
    if len(buys) < min_n:
        status = WATCHING
    else:
        status = NOT_YET if failed else MEETS_BAR
    return Graduation(
        n=len(buys),
        roi=roi,
        roi_se=roi_se,
        first_half=h1,
        second_half=h2,
        clv_n=len(clvs),
        clv=clv if clvs else None,
        clv_se=clv_se,
        status=status,
        failed=tuple(failed),
    )


def graduation_lines(entries: list[LedgerEntry], *, min_n: int = MIN_N) -> list[str]:
    g = graduation(entries, min_n=min_n)
    clv = "n/a" if g.clv is None else f"{g.clv:+.4f} (se {g.clv_se:.4f})"
    lines = [
        f"  status {g.status}  (display only; the engine stays paper either way)",
        f"  eligible buys {g.n} (need {min_n}): pre-kickoff price, one per game-market side",
        f"  ROI {g.roi * 100:+.1f}% (se {g.roi_se * 100:.1f})"
        f"  halves {g.first_half * 100:+.1f}% / {g.second_half * 100:+.1f}%",
        f"  CLV vs a close within {int(CLOSE_WINDOW.total_seconds() // 60)} min of kickoff:"
        f" {clv} on {g.clv_n} buy(s)",
    ]
    if g.failed:
        lines.append(f"  not met: {', '.join(g.failed)}")
    return lines
