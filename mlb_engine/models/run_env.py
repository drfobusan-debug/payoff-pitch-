"""Price the run environment the simulator sits in, and the game it prices hot.

Two corrections live here, on two different scales.

``RunEnvTilt`` (below) is a *within-slate* one: how far this game's simulated total
sits above the league's run level, fitted on graded batter props.

``scale_for_total`` / ``apply_shift`` are the *league-level* one: the simulator's
own run environment is not pinned to any league total, so it prices every game in a
league of its own. Replayed with two league-average lineups and league-average
staffs it scores ``BASELINE_TOTAL`` runs against a league that has played 8.95 for
the season and 8.58 over the trailing month, and every over in the book inherits
the gap. See ``TOTALS`` below for what is corrected, where, and what it is graded
on -- the two corrections are deliberately disjoint by market.

The batter-prop correction, applied in over-space (a prop is one opinion expressed
two ways, so marking an over down raises its under by the same amount):

  logit(p_over') = logit(p_over) - tilt - slope * elevation

``tilt`` is a constant: the simulator's batter overs are too high everywhere.
``elevation`` is how far the simulator's own game-total mean sits above the
league's run level, so the correction is largest on the games the simulator has
priced hot -- which is where its overs miss worst.

Measured on 112,795 graded batter prop rows from the ledger (35 slates), fitting
both terms on earlier slates and scoring them on later ones. Weekly walk-forward,
four blocks, refitting each week on everything before it:

    block        n     tilt  slope   logloss           Brier
    2026-07-26  20640  0.08  0.04   0.5523 -> 0.5512   0.1856 -> 0.1852
    2026-08-02  19656  0.10  0.03   0.5462 -> 0.5446   0.1825 -> 0.1817
    2026-08-09  18681  0.10  0.04   0.5658 -> 0.5593   0.1918 -> 0.1887
    2026-08-16  34312  0.15  0.05   0.5652 -> 0.5621   0.1915 -> 0.1899

better in 4/4 blocks on both scores, pooled 0.5585 -> 0.5554 and 0.1883 ->
0.1869 (``python -m scripts.run_env_study``). What it actually fixes is the overs
the engine *likes*: on the second half of the window the props it priced over .50
to the over said 59.0 and hit 50.5 (+8.52), and after the correction 58.0 against
52.7 (+5.33), while the unders it likes go from -1.98 to -0.28.

On the 27,115 of those rows that carry a devigged price it also, for the first
time in this ledger, adds something to the price rather than subtracting from it:
blending the corrected model into the market at the shipped 0.3 anchor scores
0.2106 against the market's own 0.2107 and the uncorrected blend's 0.2110.

Batter props only. The same fit on the (much thinner) pitcher families does not
support it: pitcher_bb gets worse under the same correction (Brier 0.2233 ->
0.2252), pitcher_outs and pitcher_k are flat, and their own fitted tilt is
negative -- the constant over-bias is a batter-prop phenomenon, not a slate-wide
one. Every batter family improves (batter_tb 0.1682 -> 0.1662, batter_hrr 0.2261
-> 0.2233, all eight better).

Three honest limits. This is forecast quality, not profit: graded ROI on priced
buys does not separate a threshold version of this from simply not buying overs,
so nothing here is allowed to gate a bet -- it moves the number and lets the
existing screens read it. The elevation is measured against a fixed league
baseline, which is what the trailing 90-day league total actually was across the
window (8.97 to 9.08, mean 9.02), so it needs refitting for a new season rather
than trusting the constant. And the study reconstructed the simulator's mean from
the calibrated total prices in the ledger while this reads the simulator's mean
directly, so the two differ by whatever the game-total calibration map does.

The same limit binds the league-level correction, and harder: it moves the total
markets' numbers and refuses nothing. Graded buys on the totals stayed a losing
book across the window whether corrected or not (game-total overs -25.1% on 310
bets, unders +17.2% on 312), so this is a better forecast of a market whose overs
should still not be bought, and the buy gates were deliberately left as they are.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# League runs per game. Matches ``filters.umpire.LEAGUE_RUNS_BASELINE`` and the
# trailing 90-day league total measured over the graded window.
LEAGUE_TOTAL_BASELINE = 9.0

# Ceiling on |elevation|, in runs. The fitted slope is small and linear in logit
# space, so a single wild simulated total cannot be allowed to move a prop by an
# unbounded amount; the graded window's slate means span -1.2 to +1.6.
MAX_ELEVATION = 3.0

EPS = 1e-6


@dataclass(frozen=True)
class RunEnvTilt:
    """The fitted over-tilt and run-environment slope, in logit units."""

    over_tilt: float
    env_slope: float

    @property
    def enabled(self) -> bool:
        return self.over_tilt != 0.0 or self.env_slope != 0.0

    @staticmethod
    def elevation(sim_total_mean: float | None) -> float | None:
        """Runs by which the simulator's game total sits above the league's."""
        if sim_total_mean is None:
            return None
        raw = float(sim_total_mean) - LEAGUE_TOTAL_BASELINE
        return max(-MAX_ELEVATION, min(MAX_ELEVATION, raw))

    def apply(self, p_over: float, elevation: float | None) -> float:
        """Correct a probability handed in on the over scale.

        Neutral on a game whose run environment is unreadable: both terms were
        fitted together on rows where the simulator's own total was recoverable,
        so charging the constant one where the elevation is unknown would be
        extrapolating half of a joint fit.
        """
        if elevation is None or not self.enabled:
            return p_over
        p = min(max(float(p_over), EPS), 1 - EPS)
        x = math.log(p / (1 - p)) - self.over_tilt - self.env_slope * elevation
        return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# The league-level correction: the simulator's own run environment.
#
# Outcomes that put a runner on. Scaling these and letting the in-play out absorb
# the residual moves run scoring without touching the *shape* of a plate
# appearance -- the strikeout share, which is measured per matchup and is not what
# is wrong, is left alone.
NON_OUT: tuple[str, ...] = ("1B", "2B", "3B", "HR", "BB")

# Runs two league-average teams score in the simulator at scale 1.0, and the runs a
# unit of scale is worth near it (seed-pinned, 0.94 -> 8.16, 1.00 -> 9.27,
# 1.04 -> 10.02, so the curve is near enough linear across the clamp to solve in
# one step). Both are properties of the simulator rather than of the league, so
# ``scripts/run_env_totals_study.py`` re-measures them -- re-run it whenever the
# run models change, and move these if it disagrees.
BASELINE_TOTAL = 9.27
RUNS_PER_SCALE = 18.77

# A league total outside ~8.3-10.1 runs is a measurement failure rather than a
# league, and a scale far from 1.0 would be repricing the matchup model instead of
# the environment it sits in.
SCALE_CLAMP = (0.94, 1.04)

# Refuse to hand the simulator a plate appearance that is nearly all outs (or has
# no room for them): the input was not what this correction assumes.
MIN_OUT_SHARE = 0.05

# Log odds of the *over* per unit of non-out scale, by market and line. Measured as
# a central difference across the clamp on common random numbers -- the game totals
# in the simulator, the first-five totals in the Markov chain that actually prices
# them. ``scripts/run_env_totals_study.py`` regenerates the table; do not
# hand-adjust it.
#
# Only the two total markets are here, and the omissions are the point:
#
# * batter props are corrected by ``RunEnvTilt`` above, whose constant term was
#   fitted on ledger rows priced *without* this shift, so charging both would mark
#   the same over down twice;
# * moneylines, run lines and pitcher props have no measured coefficient, and a run
#   environment does not move a side the way it moves a total.
#
# Graded walk-forward on 4,996 graded total rows (33 slates), the target being the
# league's trailing 30 days ending the day before each slate, so no outcome the
# shift is scored on is inside the number that set it: Brier 0.2431 -> 0.2401 and
# log loss 0.6820 -> 0.6754, better on all four game-total lines and both F5 lines,
# and better in 4 of 6 weekly blocks (flat in the two July ones, where the league
# was still playing the simulator's run level, so the shift was ~0). The overs it
# corrects were the miss: o8.5 said 50.6 and hit 44.2, o10.5 said 37.7 and hit 30.8.
TOTALS: dict[str, dict[float, float]] = {
    "game_total": {7.5: 6.95, 8.5: 6.98, 9.5: 7.20, 10.5: 7.41},
    "f5_total": {4.5: 5.56, 5.5: 5.80},
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
    degenerate matchup row is never turned into a negative probability. This is
    what ``TOTALS`` is measured from; nothing prices a card through it.
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


def logit_shift(market: str, line: float | None, scale: float) -> float:
    """Log odds the scale is worth to this market's over; 0.0 where unmeasured.

    A market absent from ``TOTALS`` is left alone, and so is a line the table does
    not carry: these are the lines the engine posts, so an unknown line means the
    board has moved somewhere this was never graded.
    """
    table = TOTALS.get(market)
    if table is None or line is None or scale == 1.0:
        return 0.0
    coef = table.get(float(line))
    return 0.0 if coef is None else coef * (scale - 1.0)


def apply_shift(p_over: float, market: str, line: float | None, scale: float) -> float:
    """Move a calibrated over-probability into the league's run environment.

    Applied after calibration, not to the rates the simulator reads. Scaling the
    rates is the natural place and it does not work: the isotonic map is monotone
    and the correction is very nearly a constant log-odds shift per market, so the
    map re-maps a corrected raw back toward the win rate the *uncorrected* raw
    predicted, which made that version a wash to negative on the holdout.
    """
    delta = logit_shift(market, line, scale)
    if delta == 0.0:
        return p_over
    p = min(max(float(p_over), EPS), 1 - EPS)
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1 - p)) + delta)))
