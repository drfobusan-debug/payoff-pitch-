"""Power gate for batter home-run buys.

Backtesting the graded HR props showed that the *priced* HR probability carries
almost no signal about which home-run longshots actually go over -- the metrics
that separate true home runs from false positives are the batter's raw power
inputs, chiefly **max exit velocity** (the single strongest separator) and
**barrel rate**. Home-run "buys" are picked on EV (long payouts x a small
probability), so a weak-power hitter can clear the EV bar and become a false
positive.

This gate demotes a ``batter_hr`` BUY to Pass when the hitter lacks the power
profile, lifting the PPV of the HR buys without touching the probability model.
It is a post-model selection gate (never changes a probability), env-tunable
with a kill-switch, and stays neutral on thin batted-ball samples so it cannot
punish small-sample noise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Provisional thresholds -- a hitter must clear BOTH to keep an HR buy. These
# are anchored just above league-average power (max EV ~108 mph, barrel ~8%) and
# are meant to be tuned against a longer graded window; see MLBE_HR_* env knobs.
DEFAULT_MIN_MAX_EV = 109.0
DEFAULT_MIN_BARREL = 0.070
DEFAULT_MIN_BBE = 15


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


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class HRPowerGate:
    """Config + logic for the home-run power gate."""

    enabled: bool = True
    min_max_ev: float = DEFAULT_MIN_MAX_EV
    min_barrel: float = DEFAULT_MIN_BARREL
    min_bbe: int = DEFAULT_MIN_BBE

    @classmethod
    def from_env(cls) -> HRPowerGate:
        return cls(
            enabled=_env_flag("MLBE_HR_POWER_GATE", True),
            min_max_ev=_env_float("MLBE_HR_MAX_EV", DEFAULT_MIN_MAX_EV),
            min_barrel=_env_float("MLBE_HR_BARREL", DEFAULT_MIN_BARREL),
            min_bbe=_env_int("MLBE_HR_MIN_BBE", DEFAULT_MIN_BBE),
        )

    def allows(
        self,
        max_ev: float | None,
        barrel: float | None,
        bbe: int | None,
    ) -> tuple[bool, str]:
        """Return (keep_buy, reason).

        ``keep_buy`` is False only when the gate is enabled, the sample is large
        enough to trust, and the hitter fails the max-EV or barrel floor.
        """
        if not self.enabled:
            return True, ""
        if bbe is None or bbe < self.min_bbe:
            return True, "hr-gate: neutral (thin batted-ball sample)"
        if max_ev is None or barrel is None:
            return True, "hr-gate: neutral (no power data)"
        if max_ev < self.min_max_ev or barrel < self.min_barrel:
            return False, (
                f"hr-gate: PASS (max_ev {max_ev:.1f}<{self.min_max_ev:.0f} "
                f"or barrel {barrel:.3f}<{self.min_barrel:.3f})"
            )
        return True, (
            f"hr-gate: OK (max_ev {max_ev:.1f}, barrel {barrel:.3f})"
        )
