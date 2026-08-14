"""Base-out transitions for one plate appearance.

Runners do not advance on command. A single scores the man on second 40% of the
time with nobody out and 79% with two, not always; a ground ball with a force at
second turns two about 39% of the time, not never; and a man on third scores on an
out with fewer than two outs more than half the time, which the deterministic
rules scored as nothing at all.

Every rate below is measured on 42,203 plate appearances of 2026 Statcast
(2026-06-01..2026-07-15) rather than assumed, and each carries its sample size.
The estimates use runs and outs recorded, which Statcast reports on every plate
appearance including the one that ends the inning, so they are not conditioned on
the inning continuing.

The rates are shared by both run models — the exact F5 Markov chain enumerates
this distribution and the Monte Carlo sim samples the same branches — so the two
cannot drift apart in what they believe a single does.
"""

from __future__ import annotations

import os

OUTCOMES = ("1B", "2B", "3B", "HR", "BB", "K", "OUT")

# --- outs ------------------------------------------------------------------
# P(two outs | batted-ball out, force at second, <2 outs), by contact type.
DP_PER_GB_OUT = 0.391  # n=1,768
DP_PER_AIR_OUT = 0.048  # fly/line/popup pooled, n=1,892
# Ground balls become outs more often than air contact does, so a pitcher's GB%
# share of batted balls understates his share of the outs he records.
GB_OUT_LIFT = 1.113  # 0.4685 of outs vs 0.4209 of batted balls
# League-average blend of the two, for callers with no pitcher in hand.
LEAGUE_DP_RATE = 0.214  # n=3,660

# --- the man on third on an out --------------------------------------------
# Sacrifice flies and run-scoring ground outs. With two outs the inning ends
# before he can do anything and he scores 1.6% of the time, on errors.
SCORE_FROM_3B_ON_OUT = (0.510, 0.553, 0.0)  # no force at second, n=102/295
SCORE_FROM_3B_ON_FORCE_OUT = (0.624, 0.562, 0.0)  # two not turned, n=157/372
SCORE_FROM_3B_ON_DP = 0.265  # n=102

# --- runners moving up on an out ------------------------------------------
OUT_ADVANCES_2B = (0.532, 0.422, 0.0)  # second to third, n=327/493
OUT_ADVANCES_1B = (0.167, 0.225, 0.0)  # first to second, no double play, n=1,081/955

# --- hits, and why these are indexed by the out count ----------------------
# With two outs a runner leaves on contact because there is nothing to lose, and
# with nobody out he holds because being erased costs the inning. Pooling the
# three is not a small simplification: the man on second scores on a single 40%
# of the time with nobody out and 79% of the time with two, and a single pooled
# 61% is wrong in both directions at once. This was the largest single error in
# the deterministic rules after the double play.
SINGLE_SCORES_2B = (0.404, 0.531, 0.793)  # n=183/288/348
SINGLE_ADVANCES_1B_TO_3B = (0.226, 0.221, 0.308)  # n=390/452/390
DOUBLE_SCORES_1B = (0.258, 0.331, 0.475)  # n=97/121/139

# A strikeout freezes the runners: a man on third scores 0.2% of the time and a
# force at second turns two 2.3% of the time, both indistinguishable from zero.

# Kill switch. MLBE_PROB_BASERUNNING=0 restores the deterministic rules exactly:
# every runner in scoring position scores on a single, the man on first always
# takes third, and nobody either advances or scores on an out. The double play is
# a separate dimension, carried by ``dp_rate``, so reproducing the old F5 chain
# also means passing 0 for it. Those errors cancel in the aggregate run
# environment -- which is why they survived this long -- while being wrong in
# every individual base-out state, so this exists to isolate a regression rather
# than as a configuration anyone should run.
if os.environ.get("MLBE_PROB_BASERUNNING", "1").strip().lower() in {"0", "false", "no"}:
    SCORE_FROM_3B_ON_OUT = (0.0, 0.0, 0.0)
    SCORE_FROM_3B_ON_FORCE_OUT = (0.0, 0.0, 0.0)
    SCORE_FROM_3B_ON_DP = 0.0
    OUT_ADVANCES_2B = (0.0, 0.0, 0.0)
    OUT_ADVANCES_1B = (0.0, 0.0, 0.0)
    SINGLE_SCORES_2B = (1.0, 1.0, 1.0)
    SINGLE_ADVANCES_1B_TO_3B = (1.0, 1.0, 1.0)
    DOUBLE_SCORES_1B = (1.0, 1.0, 1.0)

HOME = 4

Branch = tuple[int, int, int, float]


def dp_rate_from_gb(gb_pct: float) -> float:
    """P(two outs | batted-ball out, force at second) for a given ground-ball rate."""
    gb_outs = min(1.0, max(0.0, gb_pct * GB_OUT_LIFT))
    return DP_PER_GB_OUT * gb_outs + DP_PER_AIR_OUT * (1.0 - gb_outs)


def sample(base: int, outs: int, oc: str, dp_rate: float, draw: float) -> Branch:
    """Draw one branch of :func:`advance_dist` from a uniform ``draw`` in [0, 1)."""
    acc = 0.0
    branches = advance_dist(base, outs, oc, dp_rate)
    for branch in branches:
        acc += branch[3]
        if draw < acc:
            return branch
    return branches[-1]


def assign(order: list[int], new_base: int, runs: int) -> tuple[list[int], list[int]]:
    """Recover who is where from a sampled branch, for callers tracking identity.

    ``order`` is the runners lead-first with the batter last. Runners cannot pass
    each other, so the state alone determines this: the ``runs`` lead-most men
    scored, the next fill the occupied bases from third down, and whoever is left
    was retired — which for a double play is the man forced at second, correctly.
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


def _place(dests: tuple[int, ...]) -> tuple[int, int]:
    """Fold intended destinations, lead runner first, into a (base mask, runs).

    A runner cannot pass the man ahead of him, so anyone whose intended base is
    occupied stops behind it. This is what keeps the branch enumerations honest
    when two runners both want third.
    """
    mask = 0
    runs = 0
    limit = 3
    for dest in dests:
        if dest >= HOME:
            runs += 1
            continue
        base = min(dest, limit)
        if base < 1:
            raise ValueError(f"no base left for a runner: {dests}")
        mask |= 1 << (base - 1)
        limit = base - 1
    return mask, runs


def _merge(branches: list[Branch]) -> tuple[Branch, ...]:
    """Collapse branches that land in the same state, dropping impossible ones."""
    seen: dict[tuple[int, int, int], float] = {}
    for base, outs, runs, prob in branches:
        if prob <= 0.0:
            continue
        key = (base, outs, runs)
        seen[key] = seen.get(key, 0.0) + prob
    return tuple((b, o, r, p) for (b, o, r), p in sorted(seen.items()))


def advance_dist(base: int, outs: int, oc: str, dp_rate: float) -> tuple[Branch, ...]:
    """Distribution over (base', outs', runs) for outcome ``oc`` from (base, outs).

    ``base`` is a 3-bit mask: bit0 = first, bit1 = second, bit2 = third. The
    returned probabilities sum to 1.
    """
    on1 = base & 1
    on2 = (base >> 1) & 1
    on3 = (base >> 2) & 1

    if oc == "K":
        return ((base, outs + 1, 0, 1.0),)

    if oc == "OUT":
        return _merge(_out_branches(on1, on2, on3, outs, dp_rate))

    if oc == "HR":
        return ((0, outs, on1 + on2 + on3 + 1, 1.0),)

    if oc == "3B":
        return ((0b100, outs, on1 + on2 + on3, 1.0),)

    if oc == "2B":
        # Third and second score; the man on first scores or stops at third.
        if not on1:
            mask, runs = _place((HOME,) * (on3 + on2) + (2,))
            return ((mask, outs, runs, 1.0),)
        out: list[Branch] = []
        scores = DOUBLE_SCORES_1B[outs]
        for dest, prob in ((HOME, scores), (3, 1.0 - scores)):
            mask, runs = _place((HOME,) * (on3 + on2) + (dest, 2))
            out.append((mask, outs, runs, prob))
        return _merge(out)

    if oc == "1B":
        return _merge(_single_branches(on1, on2, on3, outs))

    if oc == "BB":
        mask, runs = _walk(on1, on2, on3)
        return ((mask, outs, runs, 1.0),)

    raise ValueError(f"unknown outcome {oc!r}")


def _walk(on1: int, on2: int, on3: int) -> tuple[int, int]:
    """A walk forces only: a runner moves up when the man behind takes his base."""
    if not on1:
        return ((on3 << 2) | (on2 << 1) | 1, 0)
    if not on2:
        return ((on3 << 2) | 0b011, 0)
    if not on3:
        return (0b111, 0)
    return (0b111, 1)


def _single_branches(on1: int, on2: int, on3: int, outs: int) -> list[Branch]:
    """The man on third scores; second scores or holds at third; first takes two or three."""
    out: list[Branch] = []
    p2, p1 = SINGLE_SCORES_2B[outs], SINGLE_ADVANCES_1B_TO_3B[outs]
    second = ((HOME, p2), (3, 1.0 - p2)) if on2 else ((0, 1.0),)
    first = ((3, p1), (2, 1.0 - p1)) if on1 else ((0, 1.0),)
    for d2, p2 in second:
        for d1, p1 in first:
            dests = (HOME,) * on3 + ((d2,) if on2 else ()) + ((d1,) if on1 else ()) + (1,)
            mask, runs = _place(dests)
            out.append((mask, outs, runs, p2 * p1))
    return out


def _out_branches(on1: int, on2: int, on3: int, outs: int, dp_rate: float) -> list[Branch]:
    """An out, which is where the deterministic rules were most wrong.

    With two outs already the inning ends before anyone can do anything, so the
    branch collapses; the measured 1.6% of runs there are errors and wild pitches
    the model has no business claiming.
    """
    base = (on3 << 2) | (on2 << 1) | on1
    if outs >= 2:
        return [(base, outs + 1, 0, 1.0)]

    out: list[Branch] = []
    if on1 and dp_rate > 0.0:
        # Batter and the man on first are erased; the others hold.
        for scored, p3 in _third(on3, SCORE_FROM_3B_ON_DP):
            mask, runs = _place(((HOME,) if scored else (3,)) * on3 + (2,) * on2)
            out.append((mask, outs + 2, runs, dp_rate * p3))

    rest = 1.0 - (dp_rate if on1 else 0.0)
    lead_rate = (SCORE_FROM_3B_ON_FORCE_OUT if on1 else SCORE_FROM_3B_ON_OUT)[outs]
    a2, a1 = OUT_ADVANCES_2B[outs], OUT_ADVANCES_1B[outs]
    second = ((3, a2), (2, 1.0 - a2)) if on2 else ((0, 1.0),)
    first = ((2, a1), (1, 1.0 - a1)) if on1 else ((0, 1.0),)
    for scored, p3 in _third(on3, lead_rate):
        for d2, p2 in second:
            for d1, p1 in first:
                dests = (
                    ((HOME,) if scored else (3,)) * on3
                    + ((d2,) if on2 else ())
                    + ((d1,) if on1 else ())
                )
                mask, runs = _place(dests)
                out.append((mask, outs + 1, runs, rest * p3 * p2 * p1))
    return out


def _third(on3: int, rate: float) -> tuple[tuple[bool, float], ...]:
    if not on3:
        return ((False, 1.0),)
    return ((True, rate), (False, 1.0 - rate))
