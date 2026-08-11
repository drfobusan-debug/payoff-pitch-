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

# Soft air contact is a near-absolute negative for home runs: a hitter whose
# fly balls and line drives average under 90 mph cannot carry a ball 350+ feet.
# Unlike the barrel tests this reads air contact only, so it is not satisfied by
# hard ground balls. Set to 0 to disable.
DEFAULT_MIN_FB_LD_EV = 90.0

# Standing barrel gate: on top of the power floor above, an HR buy must carry an
# elite barrel level. This demotes HR longshots on hitters who are not
# elite-power -- even when the EV looks good -- which is where the graded HR
# overs bled money. Set to 0 to disable just this gate (the power floor stays).
#
# It once had an escape hatch: a sub-elite hitter was bought anyway when his
# barrel rate over the last three weeks exceeded the trailing six. Out of time
# that trend earns nothing -- regressing the next fortnight's HR rate on the
# barrel *level* plus the trend puts the trend at coefficient -0.002, t = -0.14
# on 996 batter-windows, and faintly the wrong sign. Because the clause was
# permissive, a worthless trend cost no false negatives; it admitted buys the
# level test meant to block, and roughly half of hitters show a rising three
# weeks by chance, so it voided about half the gate. The graded ledger agrees in
# direction: HR buys admitted only by the trend won 7.4% against 16.2% for those
# clearing on level (n=27 and 37, too few to be decisive on their own).
DEFAULT_BARREL_GATE = 0.15

# The same standing gate expressed per plate appearance. Barrel rate per batted
# ball says nothing about how often a hitter puts the ball in play, so a
# whiff-prone slugger can barrel 16% of his contact and still clear a 15% gate
# while barreling far less often than a contact hitter at 10%. Barrels/PA folds
# contact frequency in and is the form projection systems weight. A hitter must
# clear the level in EITHER form, so this only removes bats that look elite per
# batted ball purely because they rarely make contact.
# 0.065/PA is roughly the 15%-per-BBE hitter at league-average contact.
DEFAULT_BARREL_PA_GATE = 0.065

# Opposing-starter contact suppression, mirroring the total-bases matchup gate.
# Home runs are the most contact-quality-dependent outcome there is, so a
# starter who gives up neither barrels nor hard contact is the wrong arm to buy
# a longshot against however good the hitter. Both floors must be breached --
# one alone is too noisy to decline a bet on. Set either to 0 to disable.
DEFAULT_MAX_OPP_BARREL = 0.060
DEFAULT_MAX_OPP_HARD_HIT = 0.360
DEFAULT_MIN_OPP_BBE = 30

# Home runs are a counting outcome: a hitter cannot hit one in a plate
# appearance he never gets. A leadoff bat averages ~4.6 PAs against ~3.9 for the
# nine hole, so the bottom third of the order is buying ~15% fewer chances at
# the same price -- and usually in a weaker run-scoring context. Slots are
# 1-indexed; 9 disables the test.
DEFAULT_MAX_SLOT = 6


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
    barrel_pa_gate: float = DEFAULT_BARREL_PA_GATE
    min_fb_ld_ev: float = DEFAULT_MIN_FB_LD_EV
    opp_enabled: bool = True
    max_opp_barrel: float = DEFAULT_MAX_OPP_BARREL
    max_opp_hard_hit: float = DEFAULT_MAX_OPP_HARD_HIT
    min_opp_bbe: int = DEFAULT_MIN_OPP_BBE
    max_slot: int = DEFAULT_MAX_SLOT

    @classmethod
    def from_env(cls) -> HRPowerGate:
        return cls(
            enabled=_env_flag("MLBE_HR_POWER_GATE", True),
            min_max_ev=_env_float("MLBE_HR_MAX_EV", DEFAULT_MIN_MAX_EV),
            min_barrel=_env_float("MLBE_HR_BARREL", DEFAULT_MIN_BARREL),
            min_bbe=_env_int("MLBE_HR_MIN_BBE", DEFAULT_MIN_BBE),
            barrel_gate=_env_float("MLBE_HR_BARREL_GATE", DEFAULT_BARREL_GATE),
            barrel_pa_gate=_env_float(
                "MLBE_HR_BARREL_PA_GATE", DEFAULT_BARREL_PA_GATE
            ),
            min_fb_ld_ev=_env_float("MLBE_HR_MIN_FB_LD_EV", DEFAULT_MIN_FB_LD_EV),
            opp_enabled=_env_flag("MLBE_HR_OPP_GATE", True),
            max_opp_barrel=_env_float(
                "MLBE_HR_MAX_OPP_BARREL", DEFAULT_MAX_OPP_BARREL
            ),
            max_opp_hard_hit=_env_float(
                "MLBE_HR_MAX_OPP_HARD_HIT", DEFAULT_MAX_OPP_HARD_HIT
            ),
            min_opp_bbe=_env_int("MLBE_HR_MIN_OPP_BBE", DEFAULT_MIN_OPP_BBE),
            max_slot=_env_int("MLBE_HR_MAX_SLOT", DEFAULT_MAX_SLOT),
        )

    def opponent_reason(
        self,
        barrel_allowed: float | None,
        hard_hit_allowed: float | None,
        bbe: int | None,
    ) -> str | None:
        """Reason the opposing starter disqualifies a home-run over.

        Fires only for a starter who suppresses *both* barrels and hard contact;
        stays neutral on a thin sample or missing data.
        """
        if not self.enabled or not self.opp_enabled:
            return None
        if bbe is None or bbe < self.min_opp_bbe:
            return None
        if barrel_allowed is None or hard_hit_allowed is None:
            return None
        if self.max_opp_barrel <= 0.0 or self.max_opp_hard_hit <= 0.0:
            return None
        if (
            barrel_allowed < self.max_opp_barrel
            and hard_hit_allowed < self.max_opp_hard_hit
        ):
            return (
                f"hr-gate: vs contact suppressor (barrel allowed "
                f"{barrel_allowed:.3f} < {self.max_opp_barrel:.3f}, hard-hit "
                f"{hard_hit_allowed:.3f} < {self.max_opp_hard_hit:.3f})"
            )
        return None

    def slot_reason(self, slot: int | None) -> str | None:
        """Reason the hitter's place in the order disqualifies a home-run over.

        ``slot`` is 1-indexed. Neutral when the lineup spot is unknown.
        """
        if not self.enabled or self.max_slot >= 9 or slot is None:
            return None
        if slot > self.max_slot:
            return f"hr-gate: bats {slot}th (too few PAs, cap {self.max_slot})"
        return None

    def allows(
        self,
        max_ev: float | None,
        barrel: float | None,
        bbe: int | None,
        barrel_pa: float | None = None,
        fb_ld_ev: float | None = None,
    ) -> tuple[bool, str]:
        """Return (keep_buy, reason).

        ``keep_buy`` is False only when the gate is enabled, the sample is large
        enough to trust, and the hitter fails the max-EV/barrel power floor, the
        soft-air-contact floor, or the standing barrel gate (barrel below
        ``barrel_gate`` per batted ball AND below ``barrel_pa_gate`` per plate
        appearance).
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
        # Soft air contact: cannot carry a ball out however hard the grounders.
        if (
            self.min_fb_ld_ev > 0.0
            and fb_ld_ev is not None
            and fb_ld_ev < self.min_fb_ld_ev
        ):
            return False, (
                f"hr-gate: PASS (FB/LD EV {fb_ld_ev:.1f}<{self.min_fb_ld_ev:.0f})"
            )
        # Standing barrel gate: keep only if barrel is elite per batted ball OR
        # per plate appearance.
        elite_per_pa = (
            self.barrel_pa_gate > 0.0
            and barrel_pa is not None
            and barrel_pa >= self.barrel_pa_gate
        )
        if self.barrel_gate > 0.0 and barrel < self.barrel_gate and not elite_per_pa:
            per_pa = (
                f", {barrel_pa:.3f}/PA<{self.barrel_pa_gate:.3f}"
                if barrel_pa is not None
                else ""
            )
            return False, (
                f"hr-gate: PASS (barrel {barrel:.3f}<{self.barrel_gate:.2f}"
                f"{per_pa})"
            )
        return True, (
            f"hr-gate: OK (max_ev {max_ev:.1f}, barrel {barrel:.3f})"
        )
