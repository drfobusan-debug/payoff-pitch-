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

The moneyline demotion on lineup *provenance* still ships off
(``MLBE_ML_LINEUP_LOCK``): a projected lineup is the normal state for an early
card, and hard-passing on it would empty most slates before the graded data says
the passes were right.

On *batter props* the sample asked for has arrived, and it says something
weaker than a refusal. Graded buys carrying a provenance stamp:

=====================  =====  =========
lineup                     n        ROI
=====================  =====  =========
posted                   391      -1.6%
projected                774      -9.9%
=====================  =====  =========

an 8.3pp gap at roughly 1.4 standard errors -- and split by date it does not
repeat: the older half runs posted +15.3% against projected -4.7%, the newer
half posted -14.7% against projected -15.9%, which is no gap at all. So a
projected lineup caps a batter prop at Moderate rather than refusing it
(``MLBE_LINEUP_PROVENANCE_CAP``, on): the money follows the posted lineups
without the slate emptying on a difference the second half of the sample cannot
find. Pitcher props are left alone -- both provenances are in profit there
(+4.5% posted on 71, +2.4% projected on 132), and a starter is announced days
before his lineup is.

The *clock* half is a different question, and the ledger has now answered it. On
the 915 graded buys carrying a first-pitch stamp, ROI splits at almost exactly
the three hours this module already called stale:

=====================  =====  =========  =====================
when it was priced         n        ROI  bootstrap 95%
=====================  =====  =========  =====================
inside 3h                369      +5.7%  [-4.5%, +16.0%]
3h or more out           546     -14.9%  [-22.9%, -6.6%]
=====================  =====  =========  =====================

and the sign repeats market by market -- batter hits, RBI, hits+runs+RBI, total
bases, pitcher strikeouts and pitcher hits are all positive inside the window and
negative outside it. So the clock gate ships **on**
(``MLBE_LINEUP_CLOCK_GATE``) and applies to every market, not just the
moneyline: an early price is not a worse *edge*, it is a bet made before the
information that resolves the game exists.

That makes the morning card a preview rather than a bet slip, which is only
honest if something re-prices the slate near lock. ``mlb-engine run
--within-hours`` is that pass, and ``setup_engine_autorun.sh`` schedules it four
times across the day so a split day/night slate is covered; the morning run
still prices, grades and emails everything, with each early row carrying the
reason it was refused.

The measurement needs the status to reach the *ledger*, not just the
recommendation: see :func:`mlb_engine.audit.analysis.lineup_findings` for the
read, which reports an under-powered split as under-powered rather than as a
verdict on this gate.
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

# The provenance cap covers the markets the split was measured on: a batter's
# presence in the lineup is the thing a projection can be wrong about.
PROVENANCE_MARKETS = "batter_"


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
    clock: bool = True
    provenance_cap: bool = True

    @classmethod
    def from_env(cls) -> LineupLockGate:
        return cls(
            demote=_env_flag("MLBE_ML_LINEUP_LOCK", False),
            stale_hours=_env_float("MLBE_LINEUP_STALE_HOURS", DEFAULT_STALE_HOURS),
            clock=_env_flag("MLBE_LINEUP_CLOCK_GATE", True),
            provenance_cap=_env_flag("MLBE_LINEUP_PROVENANCE_CAP", True),
        )

    def caps_at_moderate(self, lock: LineupLock | None, market: str) -> tuple[bool, str]:
        """Return (cap, reason): is this batter prop priced off an unposted lineup?

        Neutral without a lock, which is the state of a backtest: an unrecorded
        provenance is not a projected one.
        """
        if not self.provenance_cap or lock is None:
            return False, ""
        if not market.startswith(PROVENANCE_MARKETS):
            return False, ""
        if lock.status != PROJECTED:
            return False, ""
        return True, (
            "lineup provenance: MODERATE cap (priced off a projected lineup; "
            "projected batter buys returned -9.9% against -1.6% posted) -- "
            "the late pass re-prices this hitter once he is in the lineup"
        )

    def in_window(self, hours: float | None) -> bool:
        """Is this game close enough to first pitch to be bet?

        A game with no start time is in the window: the clock cannot refuse what
        it cannot read. A game already underway is not -- its pre-match board is
        gone, so a price for it is a price for nothing.
        """
        if hours is None:
            return True
        return 0.0 <= hours < self.stale_hours

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

    def clock_allows(self, lock: LineupLock | None) -> tuple[bool, str]:
        """Return (keep_buy, reason) for a buy in *any* market, on the clock alone.

        Neutral without a first-pitch stamp, which is the state of every
        backtest and of a slate whose start times the feed has not published:
        the gate refuses a measured distance from lock, never a missing one.
        """
        if not self.clock or lock is None:
            return True, ""
        hours = lock.hours_to_first_pitch
        if hours is None or hours < self.stale_hours:
            return True, ""
        return False, (
            f"lineup clock: PASS (priced {hours:.1f}h out, before the lineups, "
            f"scratches and weather that resolve it; buys priced "
            f"{self.stale_hours:.0f}h+ out returned -14.9% against +5.7% inside "
            "it) -- the late pass re-prices this game near lock"
        )

    def allows(self, lock: LineupLock | None) -> tuple[bool, str]:
        """Return (keep_buy, reason) for a moneyline buy in this game."""
        if lock is None or not lock.stale:
            return True, ""
        note = lock.note or "stale lineup/timing"
        if not self.demote:
            return True, f"lineup-lock: WARN ({note}); re-run near lock"
        return False, f"lineup-lock: PASS ({note})"
