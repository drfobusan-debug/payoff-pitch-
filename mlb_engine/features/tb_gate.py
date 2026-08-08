"""Power + matchup gate for batter total-bases buys.

Total bases is the worst-performing graded market: 31.8% PPV against the ~52.4%
break-even at -110, and it is overconfident exactly where it buys (the 0.50-0.60
band priced 53.9% and cashed 31.0%). Its only filter was the shared contact
floor, which excludes a bat purely on **xSLG** -- yet the total-bases backtest
that set the selector weights found max exit velocity and barrel rate separate
the over winners from the losers while "xSLG/xBA carry little signal". The
market was being gated on the one metric its own testing says does not predict
it.

Two gates, both applied post-model (neither ever changes a probability):

* **Batter power** -- keep a TB buy only when the hitter clears a barrel-rate
  *and* a max-EV floor. Total bases is a power market; a below-average-contact
  bat clearing the EV bar is the false positive that bled the market.
* **Opposing starter** -- drop the TB over when the batter faces a starter who
  suppresses contact quality on *both* barrels and hard contact. TB was the only
  power market with no opposing-pitcher gate at all; hits/singles already get the
  SIERA "vs ace" veto.

Both stay neutral when the batted-ball sample is too thin to trust, so the gate
never punishes small-sample noise, and both are env-tunable with kill switches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Batter power floors, anchored at league average (max EV ~108 mph, barrel ~8%).
# Total bases needs extra-base contact, so a bat below the league's power line
# should not be bought in it at any price.
DEFAULT_MIN_BARREL = 0.060
DEFAULT_MIN_MAX_EV = 107.0
DEFAULT_MIN_BBE = 15

# Opposing-starter contact suppression. A starter is treated as a suppressor
# only when he is below baseline on BOTH barrels allowed (~8%) and hard-hit
# allowed (~40%) -- one alone is too noisy to veto a bet on. Set either floor to
# 0 to disable that half of the test.
DEFAULT_MAX_OPP_BARREL = 0.060
DEFAULT_MAX_OPP_HARD_HIT = 0.360
DEFAULT_MIN_OPP_BBE = 30


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
class TBGate:
    """Config + logic for the total-bases power and matchup gates."""

    enabled: bool = True
    min_barrel: float = DEFAULT_MIN_BARREL
    min_max_ev: float = DEFAULT_MIN_MAX_EV
    min_bbe: int = DEFAULT_MIN_BBE
    opp_enabled: bool = True
    max_opp_barrel: float = DEFAULT_MAX_OPP_BARREL
    max_opp_hard_hit: float = DEFAULT_MAX_OPP_HARD_HIT
    min_opp_bbe: int = DEFAULT_MIN_OPP_BBE

    @classmethod
    def from_env(cls) -> TBGate:
        return cls(
            enabled=_env_flag("MLBE_TB_GATE", True),
            min_barrel=_env_float("MLBE_TB_MIN_BARREL", DEFAULT_MIN_BARREL),
            min_max_ev=_env_float("MLBE_TB_MIN_MAX_EV", DEFAULT_MIN_MAX_EV),
            min_bbe=_env_int("MLBE_TB_MIN_BBE", DEFAULT_MIN_BBE),
            opp_enabled=_env_flag("MLBE_TB_OPP_GATE", True),
            max_opp_barrel=_env_float(
                "MLBE_TB_MAX_OPP_BARREL", DEFAULT_MAX_OPP_BARREL
            ),
            max_opp_hard_hit=_env_float(
                "MLBE_TB_MAX_OPP_HARD_HIT", DEFAULT_MAX_OPP_HARD_HIT
            ),
            min_opp_bbe=_env_int("MLBE_TB_MIN_OPP_BBE", DEFAULT_MIN_OPP_BBE),
        )

    def power_reason(
        self,
        barrel: float | None,
        max_ev: float | None,
        bbe: int | None,
    ) -> str | None:
        """Reason the hitter's own power disqualifies a total-bases buy.

        ``None`` means no exclusion -- including when the gate is off or the
        batted-ball sample is too thin/absent to read.
        """
        if not self.enabled:
            return None
        if bbe is None or bbe < self.min_bbe:
            return None
        if barrel is None or max_ev is None:
            return None
        if barrel < self.min_barrel:
            return (
                f"tb-gate: barrel {barrel:.3f} < {self.min_barrel:.3f}"
            )
        if max_ev < self.min_max_ev:
            return (
                f"tb-gate: max_ev {max_ev:.1f} < {self.min_max_ev:.1f}"
            )
        return None

    def opponent_reason(
        self,
        barrel_allowed: float | None,
        hard_hit_allowed: float | None,
        bbe: int | None,
    ) -> str | None:
        """Reason the opposing starter disqualifies a total-bases over.

        Fires only for a starter who suppresses *both* barrels and hard contact;
        stays neutral on a thin sample or missing data.
        """
        if not self.enabled or not self.opp_enabled:
            return None
        if bbe is None or bbe < self.min_opp_bbe:
            return None
        if barrel_allowed is None or hard_hit_allowed is None:
            return None
        if (
            barrel_allowed < self.max_opp_barrel
            and hard_hit_allowed < self.max_opp_hard_hit
        ):
            return (
                f"tb-gate: vs contact suppressor (barrel allowed "
                f"{barrel_allowed:.3f} < {self.max_opp_barrel:.3f}, hard-hit "
                f"{hard_hit_allowed:.3f} < {self.max_opp_hard_hit:.3f})"
            )
        return None
