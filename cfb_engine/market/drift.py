"""Where the market went between the first board and ours -- measured now, acted on later.

Closing line value is the only signal in this engine that can be read before a
season's worth of bets has been graded, which is why it matters here: college
football gives ~800 games a year against baseball's tens of thousands of props,
so waiting for ROI significance means waiting seasons. CLV is scored after
kickoff, though. This module scores the same movement *before* the bet, against
the first board the engine saw for the slate.

The sibling MLB engine turns that into two vetoes, and its ledger is the reason
this one does not -- yet. Over 919 priced buys it found the relationship is not
monotonic: buys the market had already moved **toward** returned -11.2% while
buys it had moved **away from** returned +4.3%, so the favourable pre-move was
the expensive one; but a *large* adverse move was expensive too, which is what
its ``clv_drift`` veto refuses. A signal that is bad at both ends and good in the
middle is a signal whose thresholds are doing all the work, and those thresholds
were fitted on baseball prices, where the number never moves and all movement is
price.

Football's movement is mostly in the handicap instead (see
:mod:`cfb_engine.market.linevalue`), the CFB ledger currently has **zero** graded
bets, and this engine's own history is a warning about porting a fitted number:
the VSiN home-field table and the efficiency marking bumps both looked
convincing and both tested null once they were measured against the closing
spread. So the drift is computed for every priced side, recorded on the
recommendation and in the ledger, and refuses nothing unless
``CFBE_DRIFT_GATE=1`` is set. When a graded season exists,
``screen_probation`` can grade the rows this would have removed and the veto can
be switched on -- or not -- on evidence rather than on baseball's.

Sign convention: **positive drift means the market moved toward the side we are
betting.** ``adverse`` is the tail where it walked away, ``run_up`` the tail where
it arrived before us.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# No-vig probability points the market may move against a side before the (opt-in)
# veto fires. MLB's value, kept as the starting point because there is no CFB
# number yet; 2 points is ~0.8 of a point of spread at the engine's margin SD.
DEFAULT_MAX_ADVERSE = 0.02
# Points the price may already have run our way. MLB uses 0.0 -- any arrival at
# all counts as late -- which is its most aggressive screen and the least
# transferable, so it is separately switchable here.
DEFAULT_MAX_RUN_UP = 0.02


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class DriftGate:
    """Report market movement on a priced side, and optionally refuse it.

    ``enabled`` governs *acting*, not measuring: with it off (the default) the
    drift is still computed and returned as a reason string, so the season
    accumulates the evidence needed to decide whether it should ever have been a
    veto. ``gate`` names which tail fired, for per-screen probation grading.
    """

    enabled: bool = False
    adverse: bool = True
    momentum: bool = False
    max_adverse: float = DEFAULT_MAX_ADVERSE
    max_run_up: float = DEFAULT_MAX_RUN_UP

    @classmethod
    def from_env(cls) -> DriftGate:
        return cls(
            enabled=_flag("CFBE_DRIFT_GATE", False),
            adverse=_flag("CFBE_DRIFT_ADVERSE", True),
            momentum=_flag("CFBE_DRIFT_MOMENTUM", False),
            max_adverse=_num("CFBE_DRIFT_MAX_ADVERSE", DEFAULT_MAX_ADVERSE),
            max_run_up=_num("CFBE_DRIFT_MAX_RUN_UP", DEFAULT_MAX_RUN_UP),
        )

    def verdict(self, drift: float | None) -> tuple[bool, str, str | None]:
        """``(keep, reason, gate)`` for a side whose market has moved ``drift``.

        Neutral when there is no baseline to compare against -- the first run of a
        slate captures the board it would have compared to, and a missing side is
        a data hole, not a betting decision.
        """
        if drift is None:
            return True, "", None
        pts = drift * 100.0
        if self.adverse and drift <= -self.max_adverse:
            reason = f"drift: market moved {pts:+.1f} pts away since first board"
            if self.enabled:
                return False, f"{reason} -> PASS", "clv_drift"
            return True, reason, None
        if self.momentum and drift > self.max_run_up:
            reason = f"drift: market already moved {pts:+.1f} pts to this side"
            if self.enabled:
                return False, f"{reason} -> PASS", "momentum_run_up"
            return True, reason, None
        return True, f"drift: {pts:+.1f} pts since first board", None
