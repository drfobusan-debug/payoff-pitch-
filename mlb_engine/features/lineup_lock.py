"""Lineup-lock staleness read: was this card priced against real, late information?

``TeamGameInfo.lineup_confirmed()`` only asserts that nine hitters are present,
and the Rotowire enrichment fills unposted lineups with *projected* ones so a
game can be priced hours early. Both paths look identical downstream, so a card
carries no record of whether it saw the lineup that will actually bat, and a
run made long before first pitch cannot have seen the information that resolves
late: scratches, the posted lineup, the final weather read, the plate umpire.

That is a moneyline problem specifically -- an early edge on a full game can be
entirely manufactured by a projected star who is scratched -- so this module
turns the two facts the pipeline already knows (lineup provenance and hours to
first pitch) into an auditable status stamped on every recommendation, plus an
optional demotion of ``game_ml`` buys.

The demotion ships **off** (``MLBE_ML_LINEUP_LOCK``): a projected lineup is the
normal state for an early card, and hard-passing on it would empty most slates
before the graded data says the passes were right. The status and note ship on,
so the ledger can measure whether projected-lineup buys actually underperform
posted-lineup ones before the gate is switched on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

# Priced this many hours or more before first pitch, a card predates the window
# in which lineups post, scratches surface and the umpire is announced.
DEFAULT_STALE_HOURS = 3.0

POSTED = "posted"
PROJECTED = "projected"


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val not in ("0", "false", "False", "")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def hours_to_first_pitch(
    game_datetime_utc: str | None,
    now: datetime | None = None,
) -> float | None:
    """Hours between ``now`` and first pitch; negative once the game has started.

    ``None`` when the slate carries no start time (or an unparseable one), which
    keeps the staleness read neutral rather than guessing.
    """
    if not game_datetime_utc:
        return None
    try:
        start = datetime.fromisoformat(game_datetime_utc.replace("Z", "+00:00"))
    except ValueError:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return (start - ref).total_seconds() / 3600.0


@dataclass(frozen=True)
class LineupLock:
    """One game's lineup provenance + timing, as stamped on its recommendations."""

    status: str  # POSTED | PROJECTED
    hours_to_first_pitch: float | None
    stale: bool
    note: str | None


@dataclass(frozen=True)
class LineupLockGate:
    """Config + logic for the lineup-lock staleness read."""

    demote: bool = False
    stale_hours: float = DEFAULT_STALE_HOURS

    @classmethod
    def from_env(cls) -> LineupLockGate:
        return cls(
            demote=_env_flag("MLBE_ML_LINEUP_LOCK", False),
            stale_hours=_env_float("MLBE_LINEUP_STALE_HOURS", DEFAULT_STALE_HOURS),
        )

    def read(self, projected: bool, hours: float | None) -> LineupLock:
        """Classify a game from its lineup provenance and hours to first pitch."""
        early = hours is not None and hours >= self.stale_hours
        parts: list[str] = []
        if projected:
            parts.append("lineup projected, not posted (late-scratch risk)")
        if early:
            assert hours is not None
            parts.append(
                f"priced {hours:.1f}h before first pitch "
                "(late scratches/weather/umpire not captured)"
            )
        return LineupLock(
            status=PROJECTED if projected else POSTED,
            hours_to_first_pitch=hours,
            stale=projected or early,
            note="; ".join(parts) if parts else None,
        )

    def allows(self, lock: LineupLock | None) -> tuple[bool, str]:
        """Return (keep_buy, reason) for a moneyline buy in this game."""
        if lock is None or not lock.stale:
            return True, ""
        note = lock.note or "stale lineup/timing"
        if not self.demote:
            return True, f"lineup-lock: WARN ({note}); re-run near lock"
        return False, f"lineup-lock: PASS ({note})"
