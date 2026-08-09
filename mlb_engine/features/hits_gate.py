"""Contact gate for batter hit / H+R+RBI buys.

Grading seven slates of ``batter_h`` and ``batter_hrr`` showed the same failure
mode in both: the model's confidence is only weakly related to whether the bet
wins (AUC ~0.55), and what error there is concentrates almost entirely on **weak
hitters**.  Splitting ``batter_h`` o0.5 by batter quality, the bottom quartile
was priced at .577 and won .493 while the top three quartiles all landed within
five points of their number.  The realised gap between the worst and best bats
was 9.3 points; the model priced it at 3.7.

That compression is what manufactures the false positives.  A weak hitter priced
as though he were nearly league average looks underpriced against a market that
has him correctly, so he clears the edge floor and gets bought -- which is why
the graded buy list filled up with backup catchers and utility infielders at
-149 while only 9% of buys landed on a top-quartile bat.

This gate is the selection-side answer: tier the hitter on the inputs that
actually produce hits -- expected BA on his contact, how often he makes contact
at all, and a small credit for speed -- and require the context to justify the
buy. The composite's weights are fitted against hits per plate appearance with a
held-out window, not asserted; see ``W_XBA`` below for what survived and what
did not.

* **elite / good** contact -- buy on the normal edge floor.
* **average** -- only when the park and the weather are working for him.
* **poor** -- never buy the over, and flag the under when the context is
  hostile as well.

It is a post-model selection gate (it never changes a probability), env-tunable
with a kill switch, and neutral on thin samples so it cannot punish noise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from mlb_engine.features.regression import BL_SPRINT, BatterRegression

# Baselines and spreads measured over the 386 batters with 40+ batted balls in a
# six-week Statcast window, read through ``build_batter_regression`` so they are
# the same quantities the gate sees at runtime.
#
# Note these are deliberately *not* ``regression.BL_XBA`` / ``BL_ZONE_CONTACT``.
# Those constants are league rates per plate appearance, while ``breg.xba`` is
# the mean expected BA over *batted balls only* -- a much higher number, since
# it excludes strikeouts. Centring on .250 put the median hitter +2.3 SD above
# baseline and classified 81% of the league "elite".
BL_XBA_CONTACT = 0.320

# League-average strikeout rate. Unlike the contact-quality metrics this one is
# a direct subtraction from hit chances: a strikeout is the one PA outcome that
# can never become a hit, whatever the contact quality behind it.
BL_K_PCT = 0.227

# Weights for the contact composite, in units of "standard deviations of the
# league distribution".
#
# These are *fitted*, and fitted **out of time**, which is not the same thing and
# turned out to matter enormously. The first version of these weights regressed
# hits per PA on inputs measured over *the same window as the outcome*, holding
# out a different set of batters. That design is circular: a ball that falls in
# for a hit raises the hitter's xBA and his hit total simultaneously, so xBA is
# partly a restatement of the target rather than a predictor of it. It came out
# the largest weight in the model.
#
# The honest design is the one the engine actually faces: read a 42-day window,
# predict the games that come *after* it. Stacked over four rolls, n=862
# batter-windows:
#
#     input     same-window    OUT OF TIME    p (out of time)
#     xBA         +2.51          +0.65          .062
#     K%          -1.42          -1.00          .0018
#     GB%         +0.07          +0.85          .0082    <- added
#     sprint      +0.01          +0.37          .218
#     LD%         +0.14          +0.32          .363     <- not added
#     zone-ct     -0.08           n/a           .432     <- dropped earlier
#
# Two reversals. **xBA falls from dominant to marginal** -- most of its apparent
# power was the circularity, and it is now the smaller of the two main terms
# rather than the anchor. **GB% flips from nothing to significant**: the earlier
# same-window test showed p=.60 and was used to justify leaving batted-ball mix
# out of this gate entirely. That conclusion was wrong. A ground ball cannot be
# a home run, but it can very easily be a hit, and the hitters who hit them keep
# hitting them (split-half reliability .68, the most stable input here).
#
# K% remains the one input that never moves under any design, which is what a
# real effect looks like. Zone contact stays out. Line-drive rate stays out of
# *this* gate -- it is a singles effect, and it lives on the singles multiplier
# in ``regression.py`` where it is significant at p<1e-4.
#
# Weights are the standardized coefficients scaled so K% is 1.0, since K% is now
# the most reliable of the four.
W_XBA = 0.65
W_K = 1.0
W_GB = 0.85
W_SPRINT = 0.37

# League standard deviations, used to put the inputs on one scale. Measured over
# the same cohort, except sprint speed, which comes from a season leaderboard
# rather than the Statcast window.
SD_XBA = 0.047
SD_K = 0.079
SD_SPRINT = 1.6
# Ground-ball rate: league mean and spread over the same out-of-time panel
# (10th/90th percentile .321/.528).
BL_GB_RATE_CONTACT = 0.420
SD_GB = 0.081

# Tier cut points, read off the percentiles of the composite pooled over three
# Statcast windows (1,262 batter-windows): elite is the top 15% of bats, good the
# next 30%, poor the bottom 20%. Set as quantiles rather than round numbers so
# the tier mix is a known share of the league instead of an accident of the
# weights -- and re-derived from scratch whenever a weight changes, since adding
# GB% widened the composite's spread and every cut moved with it.
DEFAULT_ELITE = 1.62
DEFAULT_GOOD = 0.41
DEFAULT_POOR = -1.16

# Context floor an "average" bat must clear to be bought: park factor times the
# weather multiplier, where 1.0 is a neutral night in a neutral yard. Elite and
# good bats are not context-gated -- the thesis is that they are worth buying
# when the price is right, and the park is a modifier, not a veto.
DEFAULT_MIN_CONTEXT = 1.00

# Context below which a *poor* bat becomes an active under candidate rather than
# merely a no-buy.
DEFAULT_UNDER_CONTEXT = 0.98

# Minimum batted balls before the composite is trusted at all.
DEFAULT_MIN_BBE = 15

# Plate-appearance risk. A platoon bat batting in the bottom third against the
# hand he is worst on is the likeliest hitter on the card to be lifted for a
# pinch hitter, and he is buying fewer PAs than his price implies either way.
# Slots are 1-indexed; 9 disables the test.
DEFAULT_MAX_PLATOON_SLOT = 6

TIER_ELITE = "elite"
TIER_GOOD = "good"
TIER_AVERAGE = "average"
TIER_POOR = "poor"


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


def contact_score(breg: BatterRegression | None) -> float:
    """Weighted contact composite for a hitter, in league standard deviations.

    Positive is a better hit-producing profile than league average. Returns NaN
    when the batted-ball sample is too thin to read, which every caller treats
    as "stay neutral" rather than "poor".
    """
    if breg is None or breg.bbe < DEFAULT_MIN_BBE:
        return float("nan")
    score = W_XBA * (breg.xba - BL_XBA_CONTACT) / SD_XBA
    # K% is the only input where lower is better, and it is NaN when the slice
    # carries no completed plate appearances.
    if breg.k_pct == breg.k_pct:  # not NaN
        score += W_K * (BL_K_PCT - breg.k_pct) / SD_K
    score += W_SPRINT * (breg.sprint_speed - BL_SPRINT) / SD_SPRINT
    score += W_GB * (breg.gb_rate - BL_GB_RATE_CONTACT) / SD_GB
    return float(score)


@dataclass(frozen=True)
class HitsContactGate:
    """Config + logic for the batter hit/H+R+RBI contact gate."""

    enabled: bool = True
    elite: float = DEFAULT_ELITE
    good: float = DEFAULT_GOOD
    poor: float = DEFAULT_POOR
    min_context: float = DEFAULT_MIN_CONTEXT
    under_context: float = DEFAULT_UNDER_CONTEXT
    min_bbe: int = DEFAULT_MIN_BBE
    max_platoon_slot: int = DEFAULT_MAX_PLATOON_SLOT

    @classmethod
    def from_env(cls) -> HitsContactGate:
        return cls(
            enabled=_env_flag("MLBE_HITS_GATE", True),
            elite=_env_float("MLBE_HITS_ELITE", DEFAULT_ELITE),
            good=_env_float("MLBE_HITS_GOOD", DEFAULT_GOOD),
            poor=_env_float("MLBE_HITS_POOR", DEFAULT_POOR),
            min_context=_env_float("MLBE_HITS_MIN_CONTEXT", DEFAULT_MIN_CONTEXT),
            under_context=_env_float(
                "MLBE_HITS_UNDER_CONTEXT", DEFAULT_UNDER_CONTEXT
            ),
            min_bbe=_env_int("MLBE_HITS_MIN_BBE", DEFAULT_MIN_BBE),
            max_platoon_slot=_env_int(
                "MLBE_HITS_MAX_PLATOON_SLOT", DEFAULT_MAX_PLATOON_SLOT
            ),
        )

    def tier(self, breg: BatterRegression | None) -> str | None:
        """Contact tier for a hitter, or ``None`` on a thin/unreadable sample."""
        score = contact_score(breg)
        if score != score:  # NaN
            return None
        if score >= self.elite:
            return TIER_ELITE
        if score >= self.good:
            return TIER_GOOD
        if score <= self.poor:
            return TIER_POOR
        return TIER_AVERAGE

    def platoon_pa_reason(
        self, slot: int | None, platoon_disadvantage: bool
    ) -> str | None:
        """Reason the hitter's plate-appearance outlook disqualifies the over.

        A hitter on the wrong side of the platoon in the bottom third of the
        order is both the likeliest to be pinch-hit for and the likeliest to bat
        three times instead of four. ``slot`` is 1-indexed; neutral when unknown.
        """
        if not self.enabled or self.max_platoon_slot >= 9 or slot is None:
            return None
        if platoon_disadvantage and slot > self.max_platoon_slot:
            return (
                f"hits-gate: platoon bat batting {slot}th "
                f"(pinch-hit / short-PA risk, cap {self.max_platoon_slot})"
            )
        return None

    def allows(
        self,
        breg: BatterRegression | None,
        context: float | None = None,
    ) -> tuple[bool, str]:
        """Return ``(keep_buy, reason)`` for a hit / H+R+RBI over.

        ``context`` is the park factor times the weather multiplier for tonight
        (1.0 = neutral). A poor bat is never bought; an average bat needs the
        context working for him; elite and good bats clear on price alone.
        """
        if not self.enabled:
            return True, ""
        tier = self.tier(breg)
        if tier is None:
            return True, "hits-gate: neutral (thin batted-ball sample)"
        score = contact_score(breg)
        if tier == TIER_POOR:
            return False, (
                f"hits-gate: PASS (poor contact, score {score:+.2f} "
                f"<= {self.poor:+.2f})"
            )
        if tier == TIER_AVERAGE:
            if context is None:
                return True, "hits-gate: neutral (average bat, no park context)"
            if context < self.min_context:
                return False, (
                    f"hits-gate: PASS (average bat {score:+.2f} without the "
                    f"park/weather: context {context:.3f} < {self.min_context:.2f})"
                )
            return True, (
                f"hits-gate: OK (average bat {score:+.2f}, context {context:.3f})"
            )
        return True, f"hits-gate: OK ({tier} contact, score {score:+.2f})"

    def under_reason(
        self,
        breg: BatterRegression | None,
        context: float | None = None,
        slot: int | None = None,
        platoon_disadvantage: bool = False,
    ) -> str | None:
        """Why this hitter is an active *under* candidate, if he is.

        Fires only for a poor-contact bat whose night is also working against
        him -- a hostile park/weather context, or short plate appearances from
        the bottom of the order on the wrong side of the platoon. A poor bat in
        a neutral or friendly context is a no-buy, not a fade.
        """
        if not self.enabled:
            return None
        if self.tier(breg) != TIER_POOR:
            return None
        score = contact_score(breg)
        hostile = context is not None and context < self.under_context
        short_pa = bool(
            platoon_disadvantage
            and slot is not None
            and self.max_platoon_slot < 9
            and slot > self.max_platoon_slot
        )
        if not hostile and not short_pa:
            return None
        why = []
        if hostile and context is not None:
            why.append(f"context {context:.3f} < {self.under_context:.2f}")
        if short_pa:
            why.append(f"platoon bat batting {slot}th")
        return f"hits-gate: UNDER (poor contact {score:+.2f}; {', '.join(why)})"
