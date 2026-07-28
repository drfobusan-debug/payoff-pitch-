"""Sharp-money confirmation gate for moneyline buys.

Backtesting the graded ``game_ml`` props showed that the model's own EV/edge is
the *wrong* way round -- higher-EV moneyline buys won less often (EV AUC 0.33,
p=0.004 over 102 graded rows), because the engine picks the highest-EV side and
that is systematically the losing dog. The metric that actually separates
winning moneyline buys from losers is the VSIN betting split: the share of the
**handle** (money) on the bet side, and especially handle% minus bets% -- the
classic *sharp money* indicator (AUC 0.80, p=0.027 on the buys).

This gate demotes a ``game_ml`` BUY to Pass unless the money split confirms the
side (handle% at least keeps pace with ticket%), which in effect stops the
engine from buying purely on an inverted EV signal. It is a post-model
selection gate (never changes a probability), env-tunable with a kill-switch,
and stays neutral when no VSIN split is available so it cannot punish games we
have no public-money read on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# A moneyline buy is kept only when handle% - bets% >= this threshold, i.e. the
# money is at least keeping pace with the tickets on our side (sharp agreement).
# Winning buys averaged +19.7 here, losers -2.6; 0.0 is the neutral default and
# is meant to be tuned against a longer graded window (see MLBE_ML_* env knobs).
DEFAULT_MIN_DIVERGENCE = 0.0


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


@dataclass(frozen=True)
class MLSharpGate:
    """Config + logic for the moneyline sharp-money confirmation gate."""

    enabled: bool = True
    min_divergence: float = DEFAULT_MIN_DIVERGENCE

    @classmethod
    def from_env(cls) -> MLSharpGate:
        return cls(
            enabled=_env_flag("MLBE_ML_SHARP_GATE", True),
            min_divergence=_env_float("MLBE_ML_MIN_DIVERGENCE", DEFAULT_MIN_DIVERGENCE),
        )

    def allows(
        self,
        handle_pct: float | None,
        bets_pct: float | None,
    ) -> tuple[bool, str]:
        """Return (keep_buy, reason).

        ``keep_buy`` is False only when the gate is enabled, a VSIN split is
        available, and the handle-minus-bets divergence on our side falls below
        the confirmation threshold.
        """
        if not self.enabled:
            return True, ""
        if handle_pct is None or bets_pct is None:
            return True, "ml-gate: neutral (no handle/bets split)"
        div = handle_pct - bets_pct
        if div < self.min_divergence:
            return False, (
                f"ml-gate: PASS (handle-bets {div:+.0f} < {self.min_divergence:+.0f}; "
                f"no sharp confirmation)"
            )
        return True, f"ml-gate: OK (handle-bets {div:+.0f})"
