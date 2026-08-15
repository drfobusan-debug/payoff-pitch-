"""Possession-based discrete score simulator.

The engine's core. A game is simulated as an alternating sequence of drives;
each drive ends in a touchdown, a field goal, a touchdown *for the defence*, or
no score, and the score is the accumulation. Nothing here draws a margin or a
total directly, which is the whole point: NFL points arrive in 3s and 7s and the
final margin piles up on their sums. Over 2015-2025 (n=3,028) 14.8% of games
ended by exactly 3 and 8.7% by exactly 7; a normal distribution with the same
mean and standard deviation puts 2.7% and 2.6% there. A model that cannot tell
-3 from -2.5 has given away the most reliable edge on the board before it has an
opinion about who wins.

Three measured mechanisms, all fitted on 2023-2025 play-by-play (46,918 drives)
and 2015-2025 results rather than assumed:

**Better offences score more touchdowns, not more field goals.** Across 96
team-seasons, per-drive touchdown rate tracks points per drive at r=0.976
(slope +0.137), while field-goal rate is flat (r=0.19, slope +0.014). Detroit's
2024 offence converted 37.0% of drives into touchdowns against the 2023 Jets'
8.4%, but their field-goal rates were 14.1% and 17.3%. So team quality moves the
touchdown rate and leaves the field-goal rate almost alone -- which is also why
scoring more does not flatten the key numbers.

**Late-game behaviour depends on the score, and that is where the key numbers
come from.** In the last seven minutes a team trailing by 5-8 scores a touchdown
on 28.5% of drives and kicks a field goal on 0.4% of them; trailing by 1-3 it is
12.5% and 20.1%; *leading* by 1-8 it is under 6% either way, because the leader's
job is the clock. Those three facts together compress margins toward the small
numbers and then quantise them onto 3s: the coarse version of this table, which
lumped "down 4" in with "down 1", manufactured 1-point margins at twice their
real rate because it let teams kick field goals from deficits where nobody kicks.

**Overtime is its own distribution.** Of 176 overtime games since 2015, 58.0%
finished with a 3-point margin and 34.1% with 6 -- six, not seven, because a
sudden-death touchdown ends the game before the extra point. 5.7% ended tied.

The remaining free parameters are the two the data cannot pin down -- how much a
team's scoring rate varies game to game around its rating, and how many drive
slots count as "late" -- and both are fitted to the historical margin histogram
by ``scripts/nfl/score_shape_study.py``, which is also the file that regenerates
every constant below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_engine.models.distribution import ScoreDistribution

# -- measured constants (2023-2025 play-by-play unless noted) ---------------
LG_DRIVES = 10.98  # drives per team per game
LG_PPD = 2.01  # league points per drive

# Per-drive outcome rates as a function of that team's points per drive.
TD_BASE, TD_SLOPE = 0.2183, 0.1368
FG_BASE, FG_SLOPE = 0.1558, 0.0141
# A drive that ends in a touchdown for the *defence* (pick-six, fumble return),
# slightly less likely the better the offence.
DEF_TD_BASE, DEF_TD_SLOPE = 0.0215, -0.0056

# What a touchdown is worth. The extra point is good 95.9% of the time (n=3,672)
# and a two-point conversion is made 47.2% (n=390), but the attempt rate is not
# a constant: 4.6% of touchdowns before the fourth quarter, 21.4% inside it, and
# near-certain when the touchdown leaves the team two points short. Spreading the
# aggregate 9.6% evenly over the game is what put simulated mass on 2- and
# 4-point margins that the real distribution keeps on 3 and 7.
XP_GOOD = 0.959
TWO_POINT_GOOD = 0.472
TWO_POINT_RATE = 0.046
TWO_POINT_RATE_LATE = 0.214
TWO_POINT_RATE_CHASING = 0.85

# Late-game drive rates by score differential from the offence's perspective,
# indexed from STATE_MIN to STATE_MAX inclusive and clipped at the ends.
# Empirical cells, shrunk toward their neighbours by sample size.
STATE_MIN, STATE_MAX = -11, 11
LATE_TD = np.array(
    [
        0.228, 0.235, 0.251, 0.259, 0.333, 0.285, 0.275, 0.225, 0.130, 0.082, 0.078,
        0.081,
        0.068, 0.036, 0.048, 0.064, 0.078, 0.053, 0.069, 0.053, 0.130, 0.118, 0.074,
    ]
)
LATE_FG = np.array(
    [
        0.023, 0.181, 0.180, 0.031, 0.002, 0.005, 0.023, 0.069, 0.195, 0.278, 0.322,
        0.280,
        0.139, 0.080, 0.079, 0.068, 0.064, 0.074, 0.091, 0.095, 0.093, 0.078, 0.092,
    ]
)
# The final drive of the game, which is far more constrained: a tied team kicks
# on 47.8% of them, a team leading by anything kicks on none.
FINAL_TD = np.array(
    [
        0.008, 0.007, 0.058, 0.254, 0.190, 0.094, 0.000, 0.025, 0.032, 0.017, 0.097,
        0.183,
        0.086, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    ]
)
FINAL_FG = np.array(
    [
        0.005, 0.042, 0.116, 0.000, 0.000, 0.000, 0.000, 0.000, 0.035, 0.191, 0.478,
        0.457,
        0.171, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    ]
)

# Overtime, from 176 overtime games 2015-2025.
OT_TIE_PROB = 0.057
OT_MARGINS = np.array([3.0, 6.0, 1.0, 7.0])
OT_MARGIN_PROBS = np.array([0.615, 0.362, 0.018, 0.005])
OT_HOME_WIN = 0.566

# -- fitted parameters ------------------------------------------------------
# Game-to-game dispersion of a team's scoring rate around its rating. The
# simulator's own drive-level randomness is not enough on its own: with this at
# zero the simulated margin sd is 12.4 against 14.16 actual.
FORM_SD = 0.16
# How many of the game's trailing drive slots use the late-game table. Six is
# three possessions each, a touch wider than the seven-minute window the table
# was measured over, and it is what reproduces the key numbers: at two slots
# P(margin = 3) is 10.1% against 14.8% actual, at six it is 14.4%.
LATE_SLOTS = 6
# The late-game tables move points around, so a requested mean has to be
# inverted through them before it is fed in. Two separate distortions:
#
# *Total*: the leader's clock-killing removes points the drive rates put in, so
# a requested per-team total comes back low -- ``realized = A + B * requested``.
#
# *Margin*: the same tables actively compress it, because the trailing team's
# late touchdown rate is four to five times the leader's. Real football does
# this too, but the closing spread we are handed already contains the effect: it
# is a forecast of the *final* margin, not of the margin before the comeback. So
# feeding the spread in raw double-counts the compression and returns a mean
# margin of 0.82 against 1.90 actual. ``MARGIN_B`` divides it back out.
#
# All three are refit by scripts/nfl/score_shape_study.py --fit whenever a table
# above changes.
ANCHOR_TOTAL_A, ANCHOR_TOTAL_B = 2.7011, 0.8300
ANCHOR_MARGIN_B = 0.8004
# A one-sided game also loses points to the clock, but only once the lead is
# beyond a score: the realized total is flat out to a 7-point margin and then
# falls 0.19 points for each point beyond it (0.6 low at 10, 1.3 at 14).
ANCHOR_TOTAL_MARGIN_KNEE = 7.0
ANCHOR_TOTAL_MARGIN_SLOPE = 0.19


@dataclass(frozen=True)
class ExpectedGame:
    """The means the simulator shapes: points and possessions per team.

    Drives come from pace, and pace is what gives a total its spread -- two
    fast-playing teams are not just higher-scoring, they are higher-variance.
    """

    home_points: float
    away_points: float
    home_drives: float = LG_DRIVES
    away_drives: float = LG_DRIVES

    def total(self) -> float:
        return self.home_points + self.away_points

    def margin(self) -> float:
        return self.home_points - self.away_points


def anchored_points(home_points: float, away_points: float) -> tuple[float, float]:
    """Invert the anchors so the realized means match what was requested."""
    total = home_points + away_points
    margin = home_points - away_points
    total += ANCHOR_TOTAL_MARGIN_SLOPE * max(0.0, abs(margin) - ANCHOR_TOTAL_MARGIN_KNEE)
    if ANCHOR_TOTAL_B > 0:
        total = (total - 2.0 * ANCHOR_TOTAL_A) / ANCHOR_TOTAL_B
    if ANCHOR_MARGIN_B > 0:
        margin = margin / ANCHOR_MARGIN_B
    return (total + margin) / 2.0, (total - margin) / 2.0


def _state_index(diff: np.ndarray) -> np.ndarray:
    clipped = np.clip(diff, STATE_MIN, STATE_MAX)
    return (clipped - STATE_MIN).astype(np.int64)


class DriveSim:
    """Simulate a game's score, drive by drive."""

    def __init__(self, *, n_sims: int = 40000, seed: int = 7, form_sd: float | None = None) -> None:
        self.n_sims = n_sims
        # Read the module default at call time rather than binding it here, so a
        # refit can set it and the study's grid search means something.
        self.form_sd = FORM_SD if form_sd is None else form_sd
        self.rng = np.random.default_rng(seed)

    # -- rates ------------------------------------------------------------
    def _base_rates(self, points: float, drives: float, n: int) -> tuple[np.ndarray, ...]:
        """Per-drive touchdown / field-goal / defensive-touchdown probabilities.

        One draw of ``form`` per trial: a team's scoring rate for *this* game,
        which is what stops the simulated margin being too tight.
        """
        form = np.clip(self.rng.normal(1.0, self.form_sd, n), 0.4, 2.0)
        ppd = (max(points, 3.0) / max(drives, 1.0)) * form
        excess = ppd - LG_PPD
        td = np.clip(TD_BASE + TD_SLOPE * excess, 0.01, 0.60)
        fg = np.clip(FG_BASE + FG_SLOPE * excess, 0.02, 0.30)
        def_td = np.clip(DEF_TD_BASE + DEF_TD_SLOPE * excess, 0.0, 0.10)
        return td, fg, def_td

    def _late_rates(
        self, td: np.ndarray, fg: np.ndarray, diff: np.ndarray, *, final: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score-aware rates, scaled by how good the offence is.

        The tables are league-wide, so a good offence is given proportionally
        more of the touchdown behaviour they describe and a bad one less.
        """
        idx = _state_index(diff)
        table_td = FINAL_TD if final else LATE_TD
        table_fg = FINAL_FG if final else LATE_FG
        q_td = table_td[idx] * np.clip(td / TD_BASE, 0.3, 2.5)
        q_fg = table_fg[idx] * np.clip(fg / FG_BASE, 0.3, 2.5)
        return np.clip(q_td, 0.0, 0.8), np.clip(q_fg, 0.0, 0.6)

    # -- simulation -------------------------------------------------------
    def simulate(self, exp: ExpectedGame) -> ScoreDistribution:
        """Simulate the game, half the trials with each team receiving last.

        Who has the ball at the end is worth about eight tenths of a point here,
        because the final-drive table denies a leading team any scoring at all,
        so a fixed possession order hands one side a systematic edge that has
        nothing to do with its rating.
        """
        half = max(1, self.n_sims // 2)
        first = self._simulate_order(exp, half, home_first=True)
        second = self._simulate_order(exp, self.n_sims - half, home_first=False)
        return ScoreDistribution(
            home=np.concatenate([first.home, second.home]),
            away=np.concatenate([first.away, second.away]),
        )

    def _simulate_order(self, exp: ExpectedGame, n: int, *, home_first: bool) -> ScoreDistribution:
        home_points, away_points = anchored_points(exp.home_points, exp.away_points)
        rates = (
            self._base_rates(home_points, exp.home_drives, n),
            self._base_rates(away_points, exp.away_drives, n),
        )

        home = np.zeros(n)
        away = np.zeros(n)
        slots = _drive_slots(exp.home_drives, exp.away_drives, home_first=home_first)
        for position, team in enumerate(slots):
            remaining = len(slots) - position - 1
            td, fg, def_td = rates[team]
            diff = (home - away) if team == 0 else (away - home)
            late = remaining < LATE_SLOTS
            if remaining == 0:
                q_td, q_fg = self._late_rates(td, fg, diff, final=True)
            elif late:
                q_td, q_fg = self._late_rates(td, fg, diff, final=False)
            else:
                q_td, q_fg = td, fg
            scored, conceded = self._drive(q_td, q_fg, def_td, diff, n, late=late)
            if team == 0:
                home = home + scored
                away = away + conceded
            else:
                away = away + scored
                home = home + conceded

        home, away = self._overtime(home, away)
        return ScoreDistribution(home=home, away=away)

    def _drive(
        self,
        q_td: np.ndarray,
        q_fg: np.ndarray,
        def_td: np.ndarray,
        diff: np.ndarray,
        n: int,
        *,
        late: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        roll = self.rng.random(n)
        td_points = np.where(roll < q_td, self._touchdown_points(diff, n, late=late), 0.0)
        fg_points = np.where((roll >= q_td) & (roll < q_td + q_fg), 3.0, 0.0)
        turnover_td = np.where(
            (roll >= q_td + q_fg) & (roll < q_td + q_fg + def_td), 7.0, 0.0
        )
        return td_points + fg_points, turnover_td

    def _touchdown_points(self, diff: np.ndarray, n: int, *, late: bool) -> np.ndarray:
        """6, 7 or 8, with the late two-point decision made on the scoreboard.

        Chasing the game is not a coin flip: a touchdown that leaves a team two
        points short late is followed by a conversion attempt far more often than
        the 9.6% league rate, and that decision is a large part of why final
        margins sit on 0, 3 and 7 rather than scattering.
        """
        base = TWO_POINT_RATE_LATE if late else TWO_POINT_RATE
        rate = (
            np.where(diff + 6.0 == -2.0, TWO_POINT_RATE_CHASING, base)
            if late
            else np.full(n, base)
        )
        go_for_two = self.rng.random(n) < rate
        conv = self.rng.random(n)
        two = np.where(conv < TWO_POINT_GOOD, 8.0, 6.0)
        one = np.where(conv < XP_GOOD, 7.0, 6.0)
        return np.where(go_for_two, two, one)

    def _overtime(self, home: np.ndarray, away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve regulation ties with the measured overtime distribution.

        Points are added to the winner rather than simulating another period:
        the totals market cares about the sum, and 3 or 6 more points is what
        overtime actually adds.
        """
        tied = np.flatnonzero(home == away)
        if tied.size == 0:
            return home, away
        keep_tied = self.rng.random(tied.size) < OT_TIE_PROB
        decided = tied[~keep_tied]
        if decided.size == 0:
            return home, away
        margin = self.rng.choice(OT_MARGINS, size=decided.size, p=OT_MARGIN_PROBS)
        home_wins = self.rng.random(decided.size) < OT_HOME_WIN
        home = home.copy()
        away = away.copy()
        home[decided] += np.where(home_wins, margin, 0.0)
        away[decided] += np.where(home_wins, 0.0, margin)
        return home, away


def _drive_slots(home_drives: float, away_drives: float, *, home_first: bool) -> list[int]:
    """Alternating possession order, 0 = home. ``home_first`` also decides who
    has the last possession. Unequal pace gives the faster team the extra drive.
    """
    counts = {0: int(max(4, min(18, round(home_drives)))), 1: int(max(4, min(18, round(away_drives))))}
    order = (0, 1) if home_first else (1, 0)
    slots: list[int] = []
    for i in range(max(counts.values())):
        for team in order:
            if i < counts[team]:
                slots.append(team)
    return slots
