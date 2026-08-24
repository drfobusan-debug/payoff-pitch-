"""Mark a game's batter props toward the under when the simulator prices it hot.

Two corrections, both applied in over-space (a prop is one opinion expressed two
ways, so marking an over down raises its under by the same amount):

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
"""

from __future__ import annotations

import math
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
