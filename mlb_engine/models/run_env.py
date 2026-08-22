"""Set the simulator's run environment to the league it is actually pricing.

The per-PA rates the simulator is handed come from a log5 matchup whose pieces are
each measured well, but their product is not pinned to any league total: replayed
with two league-average lineups and league-average staffs, the simulator scores
``BASELINE_TOTAL`` runs a game against a league playing 8.95 (2026 season to date,
3,820 team-games). Nothing about that gap is a side error -- it lifts every over
and depresses every under in the book by construction, which is the uniform
asymmetry the graded ledger shows in every counting market at once.

The correction is one number: the non-out scale that makes the simulator score the
league's runs per game (``scale_for_total``), solved from one measured elasticity
so it is a league measurement rather than a fitted constant. Scaling the non-out
outcomes and letting the in-play out absorb the residual corrects a run environment
without moving the *shape* of a PA -- the strikeout share is left alone.

**Where it is applied is not where it is generated.** Scaling the rates the
simulator reads is the natural place, and it does not work: the isotonic
calibration map sits downstream, it is monotone, and the correction is very nearly
a constant log-odds shift per market (measured spread across a cold pitcher's park
and a hot hitter's park: 0.005 log odds), so the map re-maps a corrected raw back
toward the win rate the *uncorrected* raw predicted -- unevenly, which made that
version of the flag a wash to negative on the holdout (#252). Refitting the map on
corrected cards is the honest alternative, and it needs corrected cards to exist
first.

So the correction is applied *after* calibration, as the log-odds shift the scale
is worth to that market's over: ``LOGIT_PER_SCALE``, measured in the simulator and
not fitted to any market's outcomes. Graded that way it improves 14 of the 16
market-and-line slices the table moves, and the two it does not are worth 0.0002
Brier. The rate scaling stays here because it is what those coefficients are
measured *from*.

All of these constants are simulator properties, not league ones, so they are
pinned by a test (``tests/test_run_env.py``) that re-measures them -- a change to
``LEAGUE_RATES`` or to the run-scoring mechanics has to move them, and should not
be allowed to do so silently.
"""

from __future__ import annotations

import math
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

# Log odds of the *over* per unit of non-out scale, per market and line. Measured as
# a central difference across the clamp on common random numbers at 120k games a
# point (``scripts/run_env_study.py --coeffs`` is the generator) -- re-measure this
# table, do not hand-adjust it.
#
# One cell, read out: a game total's over is worth ~7 log odds per unit of scale, so
# the -0.017 scale a league playing 8.95 asks for takes an o8.5 priced at .537 down
# about 3 points, and a batter line about a third of that. The coefficient rises
# with the line in every market because a higher line sits further into the tail,
# where the run environment is worth more.
LOGIT_PER_SCALE: dict[str, dict[float, float]] = {
    "game_total": {
        6.5: 7.08,
        7.0: 6.95,
        7.5: 6.95,
        8.0: 6.98,
        8.5: 6.98,
        9.0: 7.20,
        9.5: 7.20,
        10.0: 7.41,
        10.5: 7.41,
        11.0: 7.79,
        11.5: 7.79,
        12.5: 8.09,
    },
    "batter_h": {0.5: 2.16, 1.5: 2.78, 2.5: 3.92},
    "batter_1b": {0.5: 1.85, 1.5: 2.80},
    "batter_2b": {0.5: 1.38},
    "batter_hr": {0.5: 1.41},
    "batter_r": {0.5: 2.62, 1.5: 4.09},
    "batter_rbi": {0.5: 2.25, 1.5: 2.84},
    "batter_tb": {0.5: 2.16, 1.5: 2.09, 2.5: 2.04, 3.5: 1.81},
    "batter_hrr": {0.5: 2.60, 1.5: 2.77, 2.5: 2.84, 3.5: 3.23},
}


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


def logit_shift(market: str, line: float | None, scale: float) -> float:
    """Log-odds move the scale is worth to this market's over; 0.0 when unmeasured.

    A line the grid does not carry takes its nearest measured neighbour: the
    coefficient moves slowly along the line (total bases 2.09 at 1.5, 2.04 at 2.5),
    so reading across a half-run is a smaller error than leaving a market
    uncorrected in a run environment that plainly moves it. A market absent from
    the table -- moneylines, run lines, the first five, pitcher props -- is left
    alone: the correction is applied only where it has been measured.
    """
    table = LOGIT_PER_SCALE.get(market)
    if table is None or line is None or scale == 1.0:
        return 0.0
    coef = table.get(line)
    if coef is None:
        coef = table[min(table, key=lambda x: abs(x - line))]
    return coef * (scale - 1.0)


def apply_shift(prob: float, market: str, line: float | None, scale: float) -> float:
    """Move a calibrated over-probability into the league's run environment."""
    d = logit_shift(market, line, scale)
    if d == 0.0:
        return prob
    p = min(max(prob, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + d)))
