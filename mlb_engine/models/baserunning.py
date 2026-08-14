"""The base-out transition, shared by both run models.

Two things the pooled rates miss.

**The out count is the dominant conditioner.** The man on second scores on a
single 40% of the time with nobody out and 79% with two, because with two outs he
leaves on contact and with none he cannot afford to be erased. One pooled 62% is
wrong in both directions at once, and the same holds for every advancement rate
here -- first to third on a single runs .27/.29/.47.

**The double play was in the simulator and not in the Markov chain.** A ground
ball with a force at second turns two 21% of the time, so on identical inputs the
chain scored 4.2% more than the sim at the rate the pipeline actually passes it,
and 8.7% more at the rate the league actually turns. F5 totals were priced hotter
than full-game totals off the same lineups. Both models now enumerate or sample
this one table, and a test asserts they still agree.

Measured on the MLB play-by-play feed, 566 games (2026-06-01..07-15), reading
each runner's whole journey rather than the next plate appearance's base state --
the feed splits an advance into legs, so a man going first to third appears as
1B->2B then 2B->3B. Sample sizes are per cell, below.
"""

from __future__ import annotations

import os

OUTCOMES = ("1B", "2B", "3B", "HR", "BB", "K", "OUT")

# Rates below are indexed by the out count the batter came to the plate with.

# --- the out ---------------------------------------------------------------
# P(two outs recorded | ball in play for an out, man on first, <2 outs).
# n=1,666 / 2,212 by out; .196 and .214 are within noise of each other, so the
# rate is not indexed by outs -- only by the pitcher, through his ground-ball
# rate, since the force is only available on the ground.
LEAGUE_DP_RATE = 0.207
DP_PER_GB_OUT = 0.391  # a ground-ball out with the force on
DP_PER_AIR_OUT = 0.048  # fly, line, popup
GB_OUT_LIFT = 1.113  # ground balls are outs more often than they are hit

# P(man on third scores | ball in play for an out, <2 outs). The sac fly and the
# run-scoring groundout. n=243 / 651. With two outs the inning ends first.
SCORE_FROM_3B_ON_OUT = (0.584, 0.548, 0.0)
# The same, when the out was a force at second rather than a tag: the throw goes
# to second, not home. Excludes the double play, which is scored separately.
SCORE_FROM_3B_ON_FORCE_OUT = (0.624, 0.562, 0.0)
SCORE_FROM_3B_ON_DP = 0.265  # a run can still score ahead of the second out

# The productive out: P(runner takes the next base | out, <2 outs).
OUT_ADVANCES_2B = (0.557, 0.387, 0.0)  # n=724 / 1,258
OUT_ADVANCES_1B = (0.244, 0.210, 0.0)  # n=1,339 / 1,738, double plays excluded

# --- the hit ---------------------------------------------------------------
SINGLE_SCORES_2B = (0.403, 0.551, 0.786)  # n=206 / 396 / 426
SINGLE_ADVANCES_1B_TO_3B = (0.267, 0.293, 0.466)  # n=580 / 779 / 768
DOUBLE_SCORES_1B = (0.292, 0.363, 0.530)  # n=144 / 212 / 232
# A double scores the man on second .984/.973/1.000 (n=64/110/122) -- held at
# certainty, since the exception is a runner who fell down.

# --- bases given away between plate appearances ----------------------------
# Stolen bases, wild pitches, passed balls, balks and errors: 2,424 advances over
# 24,762 plate appearances with a runner on, half moving every runner and half
# moving one; 287 runners thrown out or picked off.
FREE_ADVANCE_PER_PA = 0.098
FREE_ADVANCE_ALL_SHARE = 0.50
RUNNER_ERASED_PER_PA = 0.012

# Kill switch. MLBE_OUTS_SPLIT_ADVANCE=0 restores the pooled rates that #128
# measured, so a regression can be attributed to the out-count split rather than
# to the double play. It does not restore the deterministic advancement, which
# was wrong in a way nothing here should make configurable.
if os.environ.get("MLBE_OUTS_SPLIT_ADVANCE", "1").strip().lower() in {"0", "false", "no"}:
    SCORE_FROM_3B_ON_OUT = (0.56, 0.56, 0.0)
    SCORE_FROM_3B_ON_FORCE_OUT = (0.56, 0.56, 0.0)
    OUT_ADVANCES_2B = (0.47, 0.47, 0.0)
    OUT_ADVANCES_1B = (0.22, 0.22, 0.0)
    SINGLE_SCORES_2B = (0.62, 0.62, 0.62)
    SINGLE_ADVANCES_1B_TO_3B = (0.35, 0.35, 0.35)
    DOUBLE_SCORES_1B = (0.40, 0.40, 0.40)

Branch = tuple[float, int, int, int]  # probability, base', outs', runs


def dp_rate_from_gb(gb_pct: float) -> float:
    """P(two outs | ball in play for an out, force at second) for one pitcher.

    Measured rather than scaled off an assumed anchor: the force is only there to
    be turned on the ground, so a pitcher's rate follows from his ground-ball
    rate. The previous 12% anchor was little more than half what the league turns.
    """
    gb_outs = min(1.0, max(0.0, gb_pct * GB_OUT_LIFT))
    return DP_PER_GB_OUT * gb_outs + DP_PER_AIR_OUT * (1.0 - gb_outs)


def _merge(branches: list[Branch]) -> tuple[Branch, ...]:
    """Collapse branches that land on the same state, dropping impossible ones."""
    acc: dict[tuple[int, int, int], float] = {}
    for p, base, outs, runs in branches:
        if p > 0.0:
            acc[(base, outs, runs)] = acc.get((base, outs, runs), 0.0) + p
    return tuple((p, b, o, r) for (b, o, r), p in acc.items())


def _out_branches(base: int, outs: int, dp_rate: float) -> tuple[Branch, ...]:
    """A ball in play for an out: the double play, the sac fly, the productive out."""
    on1, on2, on3 = base & 1, (base >> 1) & 1, (base >> 2) & 1
    force = bool(on1 and outs < 2)
    out: list[Branch] = []

    p_dp = dp_rate if force else 0.0
    if p_dp > 0.0:
        # The batter is out and the man on first is forced at second. Runners
        # ahead of the force hold, except the man on third, who can beat the
        # throw across often enough to matter.
        held = (on2 << 1)
        if on3:
            out.append((p_dp * SCORE_FROM_3B_ON_DP, held, outs + 2, 1))
            out.append((p_dp * (1.0 - SCORE_FROM_3B_ON_DP), held | 0b100, outs + 2, 0))
        else:
            out.append((p_dp, held, outs + 2, 0))

    p_single_out = 1.0 - p_dp
    if p_single_out <= 0.0:
        return _merge(out)
    if outs >= 2:
        # The inning is over; nobody advances and nothing scores.
        return _merge([*out, (p_single_out, base, outs + 1, 0)])

    # One out recorded. Who moves is a joint question: the man on third scores or
    # holds, then the man on second takes third if it is open, then the man on
    # first takes second if that is open.
    p_home = (SCORE_FROM_3B_ON_FORCE_OUT if force else SCORE_FROM_3B_ON_OUT)[outs]
    third_paths = (
        [(p_home, 1), (1.0 - p_home, 0)] if on3 else [(1.0, 0)]
    )
    for p3, runs in third_paths:
        held_third = on3 and not runs
        second_paths = (
            [(OUT_ADVANCES_2B[outs], 0, 1), (1.0 - OUT_ADVANCES_2B[outs], 1, 0)]
            if on2 and not held_third
            else [(1.0, on2, 0)]
        )
        for p2, still_second, to_third in second_paths:
            new_third = int(held_third or to_third)
            blocked = bool(still_second)
            first_paths = (
                [(OUT_ADVANCES_1B[outs], 0, 1), (1.0 - OUT_ADVANCES_1B[outs], 1, 0)]
                if on1 and not blocked
                else [(1.0, on1, 0)]
            )
            for p1, still_first, to_second in first_paths:
                nb = (new_third << 2) | ((still_second | to_second) << 1) | still_first
                out.append((p_single_out * p3 * p2 * p1, nb, outs + 1, runs))
    return _merge(out)


def _single_branches(base: int, outs: int) -> tuple[Branch, ...]:
    """A single. The batter is on first; who else moves is probabilistic."""
    on1, on2, on3 = base & 1, (base >> 1) & 1, (base >> 2) & 1
    out: list[Branch] = []
    second_paths = (
        [(SINGLE_SCORES_2B[outs], 1, 0), (1.0 - SINGLE_SCORES_2B[outs], 0, 1)]
        if on2
        else [(1.0, 0, 0)]
    )
    for p2, scored, held_third in second_paths:
        runs = on3 + scored
        if on1 and not held_third:
            p = SINGLE_ADVANCES_1B_TO_3B[outs]
            out.append((p2 * p, 0b101, outs, runs))
            out.append((p2 * (1.0 - p), 0b011, outs, runs))
        else:
            out.append((p2, (held_third << 2) | (on1 << 1) | 1, outs, runs))
    return _merge(out)


def advance_dist(base: int, outs: int, oc: str, dp_rate: float) -> tuple[Branch, ...]:
    """Successor states for one plate appearance, as (prob, base', outs', runs).

    ``base`` is a 3-bit mask: bit0 = first, bit1 = second, bit2 = third. The
    probabilities sum to 1.
    """
    on1, on2, on3 = base & 1, (base >> 1) & 1, (base >> 2) & 1

    if oc == "K":
        # A man on third scores on 0.2% of strikeouts and a force at second turns
        # two on 2.3%, both indistinguishable from zero at this sample.
        return ((1.0, base, outs + 1, 0),)

    if oc == "OUT":
        return _out_branches(base, outs, dp_rate)

    if oc == "BB":
        # A walk forces only: a runner moves up when the man behind takes his base.
        if not on1:
            return ((1.0, base | 1, outs, 0),)
        if not on2:
            return ((1.0, (on3 << 2) | 0b011, outs, 0),)
        if not on3:
            return ((1.0, 0b111, outs, 0),)
        return ((1.0, 0b111, outs, 1),)

    if oc == "1B":
        return _single_branches(base, outs)

    if oc == "2B":
        runs = on3 + on2
        if not on1:
            return ((1.0, 0b010, outs, runs),)
        p = DOUBLE_SCORES_1B[outs]
        return ((p, 0b010, outs, runs + 1), (1.0 - p, 0b110, outs, runs))

    if oc == "3B":
        return ((1.0, 0b100, outs, on3 + on2 + on1),)

    return ((1.0, 0, outs, on3 + on2 + on1 + 1),)  # HR


def free_bases(base: int, outs: int) -> tuple[Branch, ...]:
    """Bases handed over between plate appearances, as (prob, base', outs', runs).

    Steals, wild pitches, passed balls, balks and errors. Leaving them out is not
    conservative: they are the scoring the old certainty on the bases stood in for.
    """
    if base == 0 or outs >= 3:
        return ((1.0, base, outs, 0),)
    on1, on2, on3 = base & 1, (base >> 1) & 1, (base >> 2) & 1
    out: list[Branch] = []

    # Caught stealing or picked off: the man going is the trailing runner.
    if on1:
        erased = base & ~1
    elif on2:
        erased = base & ~0b010
    else:
        erased = 0
    out.append((RUNNER_ERASED_PER_PA, erased, outs + 1, 0))

    p_move = (1.0 - RUNNER_ERASED_PER_PA) * FREE_ADVANCE_PER_PA
    # Wild pitch, passed ball, balk: everybody moves up.
    out.append((p_move * FREE_ADVANCE_ALL_SHARE, ((on2 << 2) | (on1 << 1)) & 0b111, outs, on3))
    # A steal or a single error: the trailing runner takes the next base if it is
    # open, otherwise the man ahead of him goes.
    if on1 and not on2:
        one_up, runs = (base & ~1) | 0b010, 0
    elif on2 and not on3:
        one_up, runs = (base & ~0b010) | 0b100, 0
    elif on3:
        one_up, runs = base & ~0b100, 1
    else:
        one_up, runs = base, 0
    out.append((p_move * (1.0 - FREE_ADVANCE_ALL_SHARE), one_up, outs, runs))
    out.append((1.0 - RUNNER_ERASED_PER_PA - p_move, base, outs, 0))
    return _merge(out)


def transitions(base: int, outs: int, oc: str, dp_rate: float) -> tuple[Branch, ...]:
    """The plate appearance, and then the bases given away after it."""
    out: list[Branch] = []
    for p_pa, nb, no, runs in advance_dist(base, outs, oc, dp_rate):
        if no >= 3:
            out.append((p_pa, nb, no, runs))
            continue
        for p_free, fb, fo, fruns in free_bases(nb, no):
            out.append((p_pa * p_free, fb, fo, runs + fruns))
    return _merge(out)


def sample(base: int, outs: int, oc: str, dp_rate: float, draw: float) -> Branch:
    """Draw one branch of :func:`advance_dist` from a uniform ``draw`` in [0, 1)."""
    acc = 0.0
    branches = advance_dist(base, outs, oc, dp_rate)
    for branch in branches:
        acc += branch[0]
        if draw < acc:
            return branch
    return branches[-1]


def assign(order: list[int], new_base: int, runs: int) -> tuple[list[int], list[int]]:
    """Recover who ended where from a sampled branch, for callers tracking identity.

    ``order`` is the runners lead-first with the batter last. Runners cannot pass
    each other, so the state alone settles this: the ``runs`` lead-most men
    scored, the next fill the occupied bases from third down, and whoever is left
    was retired -- which on a double play is the man forced at second, correctly.
    """
    scored = order[:runs]
    rest = order[runs:]
    bases = [-1, -1, -1]
    for slot in (3, 2, 1):
        if new_base & (1 << (slot - 1)):
            if not rest:
                raise ValueError(f"no runner left for base {slot}: {order} {new_base:03b}")
            bases[slot - 1] = rest.pop(0)
    return bases, scored
