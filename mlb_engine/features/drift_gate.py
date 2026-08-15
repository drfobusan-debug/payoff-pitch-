"""Closing line value, applied before the bet instead of the morning after.

CLV is the only column in the graded ledger that separates a winning buy from a
losing one: over 379 priced buys, the ones that beat the close returned +5.4%
and the ones that lost it -11.8%. It is scored after first pitch, though, when
the money is already down, and on ``game_ml`` the engine beat the close just
28.1% of the time (mean CLV -0.32) -- it is systematically buying sides the
market is still walking away from.

This gate runs the same arithmetic against the *opening* board rather than the
closing one. The pipeline persists the first price it sees for each selection on
a slate (``audit/board_<date>.json``); on any later run, a side whose no-vig
price has drifted down by more than ``min_drift`` since then is one the market
has spent the day moving away from, and buying it now is buying the worse half
of that CLV split.

It is neutral by construction on the first run of a slate, because the board it
compares against is the one it just captured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MIN_DRIFT = 0.02  # no-vig probability points the market may move against us


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw not in ("0", "false", "False")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class DriftGate:
    """Veto a buy whose price has moved against it since the slate opened."""

    enabled: bool = True
    min_drift: float = DEFAULT_MIN_DRIFT

    @classmethod
    def from_env(cls) -> DriftGate:
        return cls(
            enabled=_env_flag("MLBE_CLV_GATE", True),
            min_drift=_env_float("MLBE_CLV_DRIFT", DEFAULT_MIN_DRIFT),
        )

    def allows(self, open_prob: float | None, fair_prob: float) -> tuple[bool, str]:
        """``(keep, reason)`` for a selection now priced at ``fair_prob`` no-vig."""
        if not self.enabled:
            return True, ""
        if open_prob is None:
            return True, ""
        drift = fair_prob - open_prob
        if drift <= -self.min_drift:
            return False, (
                f"clv: PASS (market moved {drift * 100:+.1f} pts against this side "
                "since the open; buys that lose the close return -11.8%)"
            )
        return True, f"clv: OK ({drift * 100:+.1f} pts since open)"
