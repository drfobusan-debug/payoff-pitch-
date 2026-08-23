"""Closing line value, applied before the bet instead of the morning after.

CLV is the only column in the graded ledger that separates a winning buy from a
losing one: over 379 priced buys, the ones that beat the close returned +5.4%
and the ones that lost it -11.8%. It is scored after first pitch, though, when
the money is already down.

The ``game_ml`` figure this gate was first justified by -- beating the close
just 28.1% of the time -- was an artefact of an undevigged Circa quote in the
consensus, not a side-selection failure: the two sides of a game summed to
-0.41 points of CLV in 136 of 136 games, and once that over-round is removed
the moneyline buys beat the close 57.5% of the time. The gate stands on the
+5.4%/-11.8% split alone, which is measured across every market.

This gate runs the same arithmetic against the *opening* board rather than the
closing one. The pipeline persists the first price it sees for each selection on
a slate (``audit/board_<date>.json``); on any later run, a side whose no-vig
price has drifted down by more than ``min_drift`` since then is one the market
has spent the day moving away from, and buying it now is buying the worse half
of that CLV split.

It is neutral by construction on the first run of a slate, because the board it
compares against is the one it just captured.

The same variable also refuses the *opposite* tail, and that sign is the one the
ledger insisted on rather than the one that sounds right. Across the 919 priced
buys that have an opening board, buys the market had already moved **toward**
returned -11.2% while buys it had moved away from returned +4.3%; on top of the
devigged-probability floor the split is -10.1% (n=168) against +11.8% (n=188,
p=0.02). It holds in all five slates with enough of both, in both halves of the
window, in batter and pitcher props separately, and inside every price band, so
it is not the shortened price and not lateness. Movement before our bet and
movement after it correlate -0.45: a price that has run our way is one that gives
the run-up back, which is why buying with the move loses the CLV split that the
adverse-drift veto above is built on. Both ends of the variable therefore refuse
-- the market is allowed to sit still, not to have already decided.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MIN_DRIFT = 0.02  # no-vig probability points the market may move against us
# No-vig points the price may have already moved *our* way before we bet it. Zero
# means a side the market has come to at all is a side we are late to; the band
# that does the damage is the small one (0 to +2 points went -12.5%, n=412), so a
# tolerance would keep exactly the rows worth refusing.
DEFAULT_MAX_RUN_UP = 0.0


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
    """Veto a buy the market has already decided, in either direction.

    ``allows`` refuses a side the market has walked away from; ``momentum_allows``
    refuses one it has already come to. They are named separately in the ledger
    (``clv_drift`` and ``momentum_run_up``) so ``screen_probation`` grades each on
    the rows it actually removed.
    """

    enabled: bool = True
    min_drift: float = DEFAULT_MIN_DRIFT
    momentum: bool = True
    max_run_up: float = DEFAULT_MAX_RUN_UP

    @classmethod
    def from_env(cls) -> DriftGate:
        return cls(
            enabled=_env_flag("MLBE_CLV_GATE", True),
            min_drift=_env_float("MLBE_CLV_DRIFT", DEFAULT_MIN_DRIFT),
            momentum=_env_flag("MLBE_MOMENTUM_GATE", True),
            max_run_up=_env_float("MLBE_MOMENTUM_MAX_RUN_UP", DEFAULT_MAX_RUN_UP),
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

    def momentum_allows(
        self, open_prob: float | None, fair_prob: float
    ) -> tuple[bool, str]:
        """``(keep, reason)`` for a side priced at ``fair_prob`` after the move.

        Refuses the buy when the no-vig price has already run toward us by more
        than ``max_run_up`` since the opening board -- the money is in before
        ours, and the ledger says the run-up comes back.
        """
        if not self.momentum:
            return True, ""
        if open_prob is None:
            return True, ""
        run_up = fair_prob - open_prob
        if run_up > self.max_run_up:
            return False, (
                f"momentum: PASS (market already moved {run_up * 100:+.1f} pts to this "
                "side since the open; buying after the move returned -11.2% against "
                "+4.3% buying before it)"
            )
        return True, f"momentum: OK ({run_up * 100:+.1f} pts to this side since open)"
