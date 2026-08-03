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

# Standing barrel gate (user-requested): on top of the power floor above, an HR
# buy must EITHER carry an elite barrel level, OR show barrel rising over the
# last three weeks vs the rolling six-week rate. This demotes HR longshots on
# hitters who are neither elite-power nor trending up -- even when the EV looks
# good -- which is where the graded HR overs bled money. Set to 0 to disable
# just this gate (the power floor stays).
DEFAULT_BARREL_GATE = 0.15


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
    barrel_gate: float = DEFAULT_BARREL_GATE

    @classmethod
    def from_env(cls) -> HRPowerGate:
        return cls(
            enabled=_env_flag("MLBE_HR_POWER_GATE", True),
            min_max_ev=_env_float("MLBE_HR_MAX_EV", DEFAULT_MIN_MAX_EV),
            min_barrel=_env_float("MLBE_HR_BARREL", DEFAULT_MIN_BARREL),
            min_bbe=_env_int("MLBE_HR_MIN_BBE", DEFAULT_MIN_BBE),
            barrel_gate=_env_float("MLBE_HR_BARREL_GATE", DEFAULT_BARREL_GATE),
        )

    def allows(
        self,
        max_ev: float | None,
        barrel: float | None,
        bbe: int | None,
        barrel_3w: float | None = None,
        barrel_6w: float | None = None,
    ) -> tuple[bool, str]:
        """Return (keep_buy, reason).

        ``keep_buy`` is False only when the gate is enabled, the sample is large
        enough to trust, and the hitter fails either the max-EV/barrel power
        floor or the standing barrel gate (barrel level below ``barrel_gate``
        AND not trending up over the last three weeks).
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
        # Standing barrel gate: keep only if barrel is elite OR rising 3w vs 6w.
        if self.barrel_gate > 0.0 and barrel < self.barrel_gate:
            rising = (
                barrel_3w is not None
                and barrel_6w is not None
                and barrel_3w > barrel_6w
            )
            if not rising:
                trend = (
                    f"3w {barrel_3w:.3f}<={barrel_6w:.3f} 6w"
                    if barrel_3w is not None and barrel_6w is not None
                    else "no 3w/6w trend"
                )
                return False, (
                    f"hr-gate: PASS (barrel {barrel:.3f}<{self.barrel_gate:.2f} "
                    f"and not rising: {trend})"
                )
            return True, (
                f"hr-gate: OK (barrel {barrel:.3f} rising 3w {barrel_3w:.3f}"
                f">{barrel_6w:.3f} 6w)"
            )
        return True, (
            f"hr-gate: OK (max_ev {max_ev:.1f}, barrel {barrel:.3f})"
        )
