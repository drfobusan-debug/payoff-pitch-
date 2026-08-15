"""How often the hitter in a lineup slot is gone before the game is.

The simulator batted nine fixed slots to the last out, so every plate appearance
a pinch hitter or defensive sub took was credited to the man who started there.
That is not a rounding error on a binary prop: in the singles panel, a hitter who
got three plate appearances failed to single 69.6% of the time against 56.1% at
four, so one lost turn is 13.5 points of Under.

The hazard here is measured, not assumed. Batting order is recovered by counting
plate appearances -- ``data.pbp`` explains why that needs care -- so the slot's
original occupant is whoever batted there the first time through, and a removal is
the first appearance that belongs to somebody else. Over 42,335 slot appearances
in 559 games, 4.52% of which are somebody else:

    state                              hazard per appearance
    opposing starter still in                  0.27%
    opposing starter out                      10.11%
    inning 7+, starter out, slot 1              7.9%
    inning 7+, starter out, slot 9             19.7%
    inning 7+, starter out, platoon edge        3.9%
    inning 7+, starter out, wrong-handed       11.0%
    ... wrong-handed and batting 9th           22.7%

Fitted jointly (Newton logistic, z in brackets): starter out +2.795 [17.8],
inning 7+ +0.872 [11.2], slot -0.001 [-0.0], same-handed +1.157 [16.1],
same-handed x slot +0.763 [6.9], on an intercept of -6.889.

What the fit says is that a removal is almost entirely a bullpen event. While the
opposing starter is in, it barely happens -- 0.27% an appearance, and 0.00% in the
first inning, which is the sanity check that the extraction is reading real
substitutions rather than a bookkeeping artefact. Once he is gone the hazard is
sixteen times higher.

Two things about the terms. Handedness *does* earn its place -- in four
chronological folds, fitting on the past and scoring the games after it, the
platoon terms beat the slot-and-inning-only model on both Brier and log loss every
time -- but it earns it as handedness, not as a wOBA split spread: the matchup a
manager can see is the hand, and a half-season split is mostly noise. And the slot
main effect is now exactly nothing on its own; what the interaction says is that
batting order only matters *for the platoon-disadvantaged bat*, which is the
defensive-first bottom-of-the-order platoon hitter and nobody else.

The cost to a hitter's prop is a fifth of a plate appearance and it is not spread
evenly. The leadoff man loses 0.14 of his slot's 4.64 appearances and is lifted in
9.2% of games; the ninth-place hitter loses 0.30 of 3.75 and is lifted in 21.7%.
Across the nine he keeps 4.02 of 4.21, which the box score's own ``battingOrder``
codes independently confirm at 3.94 of 4.12.

The replacement is measured too, off the same feed. A substitute's 1,915 plate
appearances ran .1269 singles and 27.1% strikeouts against the starters' .1430 and
22.0%: a worse bat, mostly through contact, which is why the branch moves hits and
total bases down without touching anyone's hit-per-contact skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Outcome rates of a substitute's plate appearance (pinch hitters and defensive
# subs, 1,915 PA). Held here rather than in ``rolling.LEAGUE_RATES`` because it is
# deliberately not the league average: the bench is worse than the lineup.
SUB_RATES = {
    "1B": 0.1269,
    "2B": 0.0256,
    "3B": 0.0031,
    "HR": 0.0329,
    "BB": 0.1191,
    "K": 0.2705,
    "OUT": 0.4219,  # residual, so the seven sum to 1 exactly
}

# Share of plate appearances after the starter's exit that a left-hander throws,
# from the same 765 games (4,959 of 17,714). Used only to decide whether the
# matchup the slot walks into is same-handed, which is a claim about the manager's
# choice rather than about the outcome distribution.
PEN_LHP_SHARE = 0.28

# First inning the fit treats as late. Removals happen earlier too, which the
# ``sp_out`` term carries; this is the separate step up once the bench is live.
LATE_INNING = 7


@dataclass(frozen=True)
class RemovalHazard:
    """Per-appearance chance the slot's original occupant is lifted.

    Coefficients are the joint logistic fit in the module docstring. They are
    fields rather than constants so a refit -- or a sensitivity run -- does not
    need a code change.
    """

    intercept: float = -6.889
    sp_out: float = 2.795
    late: float = 0.872
    slot: float = -0.001
    same_hand: float = 1.157
    same_hand_slot: float = 0.763

    def per_pa(
        self, *, slot: int, inning: int, starter_out: bool, same_hand: bool
    ) -> float:
        """Hazard for one appearance by a slot whose original occupant is still in.

        ``slot`` is 1-indexed and centred on the fifth spot, the scale the fit was
        run on. ``same_hand`` is the batter's stance against the hand of the arm he
        is about to face; unknown handedness should pass ``False`` rather than
        guess, which leaves the hitter on the platoon-edge (lower) hazard.
        """
        centred = (slot - 5) / 4.0
        z = (
            self.intercept
            + self.sp_out * float(starter_out)
            + self.late * float(inning >= LATE_INNING)
            + self.slot * centred
            + self.same_hand * float(same_hand)
            + self.same_hand_slot * (centred if same_hand else 0.0)
        )
        return 1.0 / (1.0 + math.exp(-z))
