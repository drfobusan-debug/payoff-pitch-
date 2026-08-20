"""Set the simulator's run environment to the league it is actually pricing.

The per-PA rates the simulator is handed come from a log5 matchup whose pieces are
each measured well, but their product is not pinned to any league total: replayed
with two league-average lineups and league-average staffs, the simulator scores
``BASELINE_TOTAL`` runs a game against a league playing 8.95 (2026 season to date,
3,820 team-games). Nothing about that gap is a side error -- it lifts every over
and depresses every under in the book by construction, which is the uniform
asymmetry the graded ledger shows in every counting market at once.

This module closes the gap at its source rather than per market: the non-out
outcomes are scaled by one number and the in-play out absorbs the residual, so a
run environment is corrected without moving the *shape* of a PA (the strikeout
share is left alone) and without a market-specific patch. The scale is solved from
one measured elasticity, ``RUNS_PER_SCALE``, so the correction is a league
measurement rather than a fitted constant: feed it the league's runs per game and
it returns the scale that reproduces it.

Both constants are simulator properties, not league ones, so they are pinned by a
test (``tests/test_run_env.py``) that re-measures them -- a change to
``LEAGUE_RATES`` or to the run-scoring mechanics has to move them, and should not
be allowed to do so silently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Outcomes that put a runner on. Scaling these and letting the in-play out take the
# residual moves run scoring without touching the strikeout share, which is measured
# per matchup and is not what is wrong.
NON_OUT: tuple[str, ...] = ("1B", "2B", "3B", "HR", "BB")

# Total runs the simulator scores with two league-average teams at scale 1.0, and
# the runs a unit of scale is worth near 1.0 (20k games per point, seed-pinned:
# 0.96 -> 8.52, 1.00 -> 9.27, 1.04 -> 10.02, so the curve is near enough linear
# across the clamp to solve in one step; scripts/run_env_study.py re-measures both).
BASELINE_TOTAL = 9.27
RUNS_PER_SCALE = 18.8

# A league total outside ~8.3-10.1 runs is a measurement failure, not a league, and
# a scale far from 1.0 would be repricing the matchup model rather than the
# environment it sits in.
SCALE_CLAMP = (0.94, 1.04)

# Refuse to hand the simulator a PA that is nearly all outs (or has no room for
# them) -- the input was not what this correction assumes.
MIN_OUT_SHARE = 0.05


def scale_for_total(target_total: float, baseline: float = BASELINE_TOTAL) -> float:
    """The non-out scale that makes the simulator score ``target_total`` runs.

    One Newton step on a measured elasticity, clamped: the residual after one step
    is a few hundredths of a run, which is inside the standard error of any league
    total worth correcting to.
    """
    raw = 1.0 + (target_total - baseline) / RUNS_PER_SCALE
    lo, hi = SCALE_CLAMP
    return max(lo, min(hi, raw))


def scale_rates(rates: Mapping[str, float], scale: float) -> dict[str, float]:
    """Scale the non-out outcomes and give the residual to the in-play out.

    Returns the rates unchanged when there is no room for the residual, so a
    degenerate matchup row is never turned into a negative probability.
    """
    out = dict(rates)
    if scale == 1.0 or "OUT" not in out:
        return out
    for key in NON_OUT:
        if key in out:
            out[key] = out[key] * scale
    rest = sum(v for k, v in out.items() if k != "OUT")
    if 1.0 - rest < MIN_OUT_SHARE:
        return dict(rates)
    out["OUT"] = 1.0 - rest
    return out


def scale_all(rates_list: Iterable[Mapping[str, float]], scale: float) -> list[dict[str, float]]:
    """``scale_rates`` over a lineup's worth of per-slot matchup rates."""
    return [scale_rates(r, scale) for r in rates_list]
