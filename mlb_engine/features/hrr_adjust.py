"""Post-model adjustment for H+R+RBI (``batter_hrr``) probabilities.

Backtesting 1,764 graded H+R+RBI props (5 slates) showed two things:

* The model's own probability *is* genuinely predictive here (winners .395 vs
  losers .364, AUC 0.591, p<1e-4) -- unlike the moneyline -- so we keep it.
* But on the **o1.5** line the model is over-confident at the top: its
  highest-probability picks realize *below* the base rate, matching the audit's
  "over-confident by ~18 pts" flag. And the metrics that separate winners are
  **contact quality** (sweet-spot%, xSLG, xBA, max EV; all p<0.01), not raw
  power (barrel/hard-hit% do not separate).

This applies two bounded, env-tunable corrections to the calibrated
``batter_hrr`` probability (never the raw sim, never other markets):

1. an **o1.5 over-confidence shrink** pulling probabilities above a pivot back
   toward it, and
2. a small **contact-quality tilt** nudging the probability by the batter's
   sweet-spot% / xSLG relative to the population means.

Both are gentle by default and fully disableable; the effect sizes are modest
(AUC ~0.55), so this is a calibration nudge, not a hard gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Population means of the graded H+R+RBI sample -- the tilt is centered here so
# above-average contact quality nudges the probability up, below-average down.
HRR_SWEET_CENTER = 0.303
HRR_XSLG_CENTER = 0.495

DEFAULT_PIVOT = 0.45
DEFAULT_SHRINK = 0.70  # o1.5 probs above the pivot are pulled 30% toward it
DEFAULT_TILT_W = 0.30
DEFAULT_TILT_CAP = 0.03


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


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class HRRAdjuster:
    """Config + logic for the H+R+RBI probability adjustment."""

    enabled: bool = True
    pivot: float = DEFAULT_PIVOT
    shrink: float = DEFAULT_SHRINK
    tilt_w: float = DEFAULT_TILT_W
    tilt_cap: float = DEFAULT_TILT_CAP

    @classmethod
    def from_env(cls) -> HRRAdjuster:
        return cls(
            enabled=_env_flag("MLBE_HRR_ADJUST", True),
            pivot=_env_float("MLBE_HRR_PIVOT", DEFAULT_PIVOT),
            shrink=_env_float("MLBE_HRR_SHRINK", DEFAULT_SHRINK),
            tilt_w=_env_float("MLBE_HRR_TILT_W", DEFAULT_TILT_W),
            tilt_cap=_env_float("MLBE_HRR_TILT_CAP", DEFAULT_TILT_CAP),
        )

    def apply(
        self,
        prob: float,
        line: float | None,
        sweet_spot: float | None,
        xslg: float | None,
    ) -> float:
        """Return the adjusted H+R+RBI probability.

        Shrinks the o1.5 tail toward the pivot (over-confidence fix) and applies
        a bounded contact-quality tilt. No-ops when disabled.
        """
        if not self.enabled:
            return prob
        adj = prob
        if line == 1.5 and adj > self.pivot:
            adj = self.pivot + (adj - self.pivot) * self.shrink
        if sweet_spot is not None and xslg is not None:
            tilt = self.tilt_w * (
                (sweet_spot - HRR_SWEET_CENTER) + (xslg - HRR_XSLG_CENTER)
            )
            adj += _clip(tilt, -self.tilt_cap, self.tilt_cap)
        return _clip(adj, 1e-6, 1.0 - 1e-6)
