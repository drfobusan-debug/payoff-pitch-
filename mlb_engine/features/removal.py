"""How often the hitter in a lineup slot is gone before the game is.

The simulator batted nine fixed slots to the last out, so every plate appearance
a pinch hitter or defensive sub took was credited to the man who started there.
That is not a rounding error on a binary prop: in the singles panel, a hitter who
got three plate appearances failed to single 69.6% of the time against 56.1% at
four, so one lost turn is 13.5 points of Under.

The hazard here is measured, not assumed. From 765 games of play-by-play, batting
order is preserved through substitutions -- the k-th plate appearance a team takes
belongs to slot k % 9 whoever bats -- so the slot's original occupant is whoever
batted there the first time through, and a removal is the first appearance that
belongs to somebody else. Over 56,083 appearances where he was still in:

    state                              hazard per appearance
    opposing starter still in                   1.6%
    opposing starter out                        8.6%
    inning 7+, starter out, slot 1               7.9%
    inning 7+, starter out, slot 9              16.6%
    inning 7+, starter out, platoon edge         5.7%
    inning 7+, starter out, wrong-handed        13.3%
    ... wrong-handed and batting 9th            24.2%

Fitted jointly (Newton logistic, z in brackets): starter out +1.180 [18.1],
inning 7+ +0.670 [12.0], slot -0.079 [-1.4], same-handed +0.631 [14.1],
same-handed x slot +0.681 [10.0].

The two things worth saying about that fit. Handedness *does* earn its place --
in four chronological folds, fitting on the past and scoring the games after it,
the platoon terms beat the slot-and-inning-only model on both Brier and log loss
every time -- but it earns it as handedness, not as a wOBA split spread: the
matchup a manager can see is the hand, and a half-season split is mostly noise.
And the slot main effect is nothing on its own; what the interaction says is that
batting order only matters *for the platoon-disadvantaged bat*, which is the
defensive-first bottom-of-the-order platoon hitter and nobody else.

The replacement is measured too, off the same feed. A substitute's 4,409 plate
appearances ran .1315 singles and 25.0% strikeouts against the starters' .1428
and 21.9%: a worse bat, mostly through contact, which is why the branch moves
hits and total bases down without touching anyone's hit-per-contact skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Outcome rates of a substitute's plate appearance (pinch hitters and defensive
# subs, 4,409 PA). Held here rather than in ``rolling.LEAGUE_RATES`` because it is
# deliberately not the league average: the bench is worse than the lineup.
SUB_RATES = {
    "1B": 0.1315,
    "2B": 0.0336,
    "3B": 0.0027,
    "HR": 0.0329,
    "BB": 0.1061,
    "K": 0.2504,
    "OUT": 0.4428,  # residual, so the seven sum to 1 exactly
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

    intercept: float = -4.469
    sp_out: float = 1.180
    late: float = 0.670
    slot: float = -0.079
    same_hand: float = 0.631
    same_hand_slot: float = 0.681

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
