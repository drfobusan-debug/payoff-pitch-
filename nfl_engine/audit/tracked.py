"""Graded-only rules: zones a screen refuses, recorded and judged, never bet.

The 2026 weekly audit found two groups of model leans the screens refuse that
went on to win -- spread underdogs the model rated 2-4 points above the
de-vigged fair, and moneyline underdogs longer than the ``longshot`` cap. Both
were cut from six weeks after the results were in, which is how a filter gets
loosened on noise. So neither changes a selection: each is graded every week on
the engine's own model leans and judged by the graduation bar (volume, margin,
both halves, CLV), and the screens stay exactly as they are until one clears it.

History already argues against both (``scripts/nfl/dog_zone_study.py``, one model
lean per game-market at the 2006-2025 closes): the spread zone went 671-679, -2.7%
over 1,350 leans, and the long moneyline dogs 185-875, -10.0% over 1,060.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nfl_engine.audit.graduation import (
    CLOSE_WINDOW,
    MIN_N,
    Graduation,
    judge,
    priced_before_kickoff,
)
from nfl_engine.audit.ledger import LOSS, WIN, LedgerEntry
from nfl_engine.market.ev import MONEYLINE, SPREAD
from nfl_engine.market.screens import MAX_AMERICAN

SPREAD_DOG_GAP_RULE = "spread_dog_gap_2_4"
ML_LONGSHOT_RULE = "ml_dog_over_300"
GAP_LO, GAP_HI = 0.02, 0.04


def model_gap(entry: LedgerEntry) -> float | None:
    return None if entry.fair_prob is None else entry.model_prob - entry.fair_prob


def spread_dog_gap(entry: LedgerEntry) -> bool:
    """A spread lean getting points, the model 2-4 points over the de-vigged fair."""
    gap = model_gap(entry)
    return (
        entry.market == SPREAD
        and entry.line is not None
        and entry.line > 0
        and gap is not None
        and GAP_LO <= gap < GAP_HI
    )


def ml_longshot(entry: LedgerEntry) -> bool:
    """A moneyline lean at a price the ``longshot`` screen refuses."""
    gap = model_gap(entry)
    return (
        entry.market == MONEYLINE
        and entry.odds is not None
        and entry.odds > MAX_AMERICAN
        and gap is not None
        and gap > 0
    )


RULES: tuple[tuple[str, str, Callable[[LedgerEntry], bool]], ...] = (
    (SPREAD_DOG_GAP_RULE, "spread dog, model gap 2-4 pts", spread_dog_gap),
    (ML_LONGSHOT_RULE, f"ML dog above +{MAX_AMERICAN:.0f}", ml_longshot),
)


@dataclass(frozen=True)
class TrackedRule:
    name: str
    label: str
    wins: int
    losses: int
    units: float
    verdict: Graduation


def matches(leans: list[LedgerEntry], rule: Callable[[LedgerEntry], bool]) -> list[LedgerEntry]:
    """Leans the rule would bet, graded win/loss, priced pre-kickoff, in kickoff order."""
    rows = [
        e
        for e in leans
        if rule(e) and e.result in (WIN, LOSS) and e.odds is not None and priced_before_kickoff(e)
    ]
    return sorted(rows, key=lambda e: e.kickoff_utc)


def tracked(leans: list[LedgerEntry], *, min_n: int = MIN_N) -> list[TrackedRule]:
    out = []
    for name, label, rule in RULES:
        rows = matches(leans, rule)
        out.append(
            TrackedRule(
                name=name,
                label=label,
                wins=sum(e.result == WIN for e in rows),
                losses=sum(e.result == LOSS for e in rows),
                units=sum(e.pnl for e in rows),
                verdict=judge(rows, min_n=min_n),
            )
        )
    return out


def tracked_lines(leans: list[LedgerEntry], *, min_n: int = MIN_N) -> list[str]:
    lines = []
    for t in tracked(leans, min_n=min_n):
        g = t.verdict
        clv = "n/a" if g.clv is None else f"{g.clv:+.4f} (se {g.clv_se:.4f}, n {g.clv_n})"
        lines.append(
            f"  {t.name:<20} {g.status:<9} n {g.n:>3} (need {min_n})  {t.wins}-{t.losses}"
            f"  units {t.units:+.2f}  ROI {g.roi * 100:+.1f}% (se {g.roi_se * 100:.1f})"
            f"  halves {g.first_half * 100:+.1f}% / {g.second_half * 100:+.1f}%"
            f"  CLV {clv}  [{t.label}]"
        )
        if g.failed:
            lines.append(f"      not met: {', '.join(g.failed)}")
    window = int(CLOSE_WINDOW.total_seconds() // 60)
    lines.append(
        f"  graded only: no selection changes. CLV counts closes within {window} min of kickoff."
    )
    return lines
