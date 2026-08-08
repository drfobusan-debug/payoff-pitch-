"""Monte Carlo game simulator.

Simulates a full game plate-appearance by plate-appearance using matchup-adjusted
outcome probabilities, tracking team runs (full game and first-5) plus per-lineup
batting lines and starter pitching lines so batter/pitcher props can be priced off
the empirical distributions.

The starter's exposure -- which sets his out (innings-pitched) ceiling -- is
bounded by BOTH a batters-faced cap (the third-time-through hook) and a
pitch-count cap. Each plate appearance burns a number of pitches that depends on
the outcome (walks and strikeouts run deeper counts than balls in play) scaled by
the pitcher's own efficiency (``pitch_eff``, from P/PA and F-Strike%). Whichever
cap trips first hands the ball to the bullpen, at which point the starter stops
accruing stats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTCOMES = ["1B", "2B", "3B", "HR", "BB", "K", "OUT"]
IDX = {o: i for i, o in enumerate(OUTCOMES)}

# Run margin (absolute) at or below which a game is still "close" -- the state in
# which a manager deploys his high-leverage relievers. Beyond it the game is out
# of hand and lower-leverage / mop-up arms finish, so the offense faces the
# team's *aggregate* pen instead of its leverage arms.
CLOSE_MARGIN = 3

# First inning in which the high-leverage arms (setup man, closer) are the ones
# who appear. Before it a close game is bridged by middle relief, so the
# leverage profile must not be applied to the 6th and 7th.
LEVERAGE_INNING = 8

# League-average pitches thrown to resolve a plate appearance, by outcome. The
# blend averages ~3.9 P/PA under a typical outcome mix; the per-pitcher
# ``pitch_eff`` scaler recentres it for command/efficiency.
PITCH_COST = {
    "BB": 5.4,
    "K": 4.8,
    "HR": 3.4,
    "1B": 3.5,
    "2B": 3.5,
    "3B": 3.5,
    "OUT": 3.3,
}


@dataclass
class TeamSimConfig:
    """Everything needed to simulate one team's offense and its pitching."""

    # Offense: per lineup slot, matchup outcome probabilities vs the opposing
    # starter and vs a generic bullpen arm.
    bat_vs_starter: list[dict[str, float]]
    bat_vs_pen: list[dict[str, float]]
    # Optional: matchup vs the opposing team's *high-leverage* relievers (its
    # 8th/9th-inning arms). Used once the starter is out AND the game is still
    # close (|margin| <= CLOSE_MARGIN); when None the aggregate ``bat_vs_pen`` is
    # used in every post-starter inning (backward-compatible default).
    bat_vs_pen_close: list[dict[str, float]] | None = None
    # Optional: matchup vs the arms that bridge from the starter's hook to the
    # 8th. Used in a close game before ``LEVERAGE_INNING``; without it those
    # innings fall back to ``bat_vs_pen_close`` (the legacy behaviour, which
    # charges a 6th-inning hand-off the closer's rates).
    bat_vs_pen_bridge: list[dict[str, float]] | None = None
    # Pitching: this team's starter caps before the bullpen takes over.
    starter_bf_cap: int = 24
    starter_pitch_cap: int = 95
    # Per-PA pitch-cost scaler (>1 = inefficient / deep counts, <1 = efficient).
    pitch_eff: float = 1.0
    # Probability a ground-ball out turns into a double play (runner on first).
    gb_dp_rate: float = 0.0


def _cdf(prob: dict[str, float]) -> np.ndarray:
    arr = np.array([prob[o] for o in OUTCOMES], dtype=float)
    arr = arr / arr.sum()
    return np.cumsum(arr)


@dataclass
class GameSimResult:
    n_sims: int
    # team run totals
    home_runs_full: np.ndarray
    away_runs_full: np.ndarray
    home_runs_f5: np.ndarray
    away_runs_f5: np.ndarray
    # per-slot batting: shape (n_sims, 9) for each stat, indexed [home, away]
    bat: dict[str, dict[str, np.ndarray]]  # bat[team][stat] -> (n_sims, 9)
    # starter pitching lines: pit[team][stat] -> (n_sims,)
    pit: dict[str, dict[str, np.ndarray]]


class MonteCarlo:
    def __init__(self, n_sims: int, seed: int | None = None) -> None:
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)

    def simulate(self, home: TeamSimConfig, away: TeamSimConfig) -> GameSimResult:
        n = self.n_sims
        home_full = np.zeros(n, dtype=np.int16)
        away_full = np.zeros(n, dtype=np.int16)
        home_f5 = np.zeros(n, dtype=np.int16)
        away_f5 = np.zeros(n, dtype=np.int16)

        bat_stats = ["H", "1B", "2B", "3B", "HR", "BB", "K", "R", "RBI"]
        bat: dict[str, dict[str, np.ndarray]] = {
            "home": {s: np.zeros((n, 9), dtype=np.int16) for s in bat_stats},
            "away": {s: np.zeros((n, 9), dtype=np.int16) for s in bat_stats},
        }
        pit_stats = ["outs", "H", "BB", "K", "ER"]
        pit: dict[str, dict[str, np.ndarray]] = {
            "home": {s: np.zeros(n, dtype=np.int16) for s in pit_stats},
            "away": {s: np.zeros(n, dtype=np.int16) for s in pit_stats},
        }

        home_cdf_start = np.stack([_cdf(p) for p in home.bat_vs_starter])
        home_cdf_pen = np.stack([_cdf(p) for p in home.bat_vs_pen])
        away_cdf_start = np.stack([_cdf(p) for p in away.bat_vs_starter])
        away_cdf_pen = np.stack([_cdf(p) for p in away.bat_vs_pen])
        home_cdf_pen_close = (
            np.stack([_cdf(p) for p in home.bat_vs_pen_close])
            if home.bat_vs_pen_close is not None
            else None
        )
        away_cdf_pen_close = (
            np.stack([_cdf(p) for p in away.bat_vs_pen_close])
            if away.bat_vs_pen_close is not None
            else None
        )
        home_cdf_pen_bridge = (
            np.stack([_cdf(p) for p in home.bat_vs_pen_bridge])
            if home.bat_vs_pen_bridge is not None
            else None
        )
        away_cdf_pen_bridge = (
            np.stack([_cdf(p) for p in away.bat_vs_pen_bridge])
            if away.bat_vs_pen_bridge is not None
            else None
        )

        # Pitching caps/efficiency are keyed by the team doing the PITCHING. Away
        # pitching faces the home hitters and vice versa.
        pitch_caps = {"home": home.starter_bf_cap, "away": away.starter_bf_cap}
        pitch_count_caps = {"home": home.starter_pitch_cap, "away": away.starter_pitch_cap}
        pitch_eff = {"home": home.pitch_eff, "away": away.pitch_eff}
        gb_dp = {"home": home.gb_dp_rate, "away": away.gb_dp_rate}

        for s in range(n):
            self._sim_one(
                s,
                home_cdf_start,
                home_cdf_pen,
                away_cdf_start,
                away_cdf_pen,
                home_cdf_pen_close,
                away_cdf_pen_close,
                home_cdf_pen_bridge,
                away_cdf_pen_bridge,
                pitch_caps,
                pitch_count_caps,
                pitch_eff,
                gb_dp,
                home_full,
                away_full,
                home_f5,
                away_f5,
                bat,
                pit,
            )

        return GameSimResult(
            n_sims=n,
            home_runs_full=home_full,
            away_runs_full=away_full,
            home_runs_f5=home_f5,
            away_runs_f5=away_f5,
            bat=bat,
            pit=pit,
        )

    def _sim_one(
        self,
        s: int,
        home_cdf_start: np.ndarray,
        home_cdf_pen: np.ndarray,
        away_cdf_start: np.ndarray,
        away_cdf_pen: np.ndarray,
        home_cdf_pen_close: np.ndarray | None,
        away_cdf_pen_close: np.ndarray | None,
        home_cdf_pen_bridge: np.ndarray | None,
        away_cdf_pen_bridge: np.ndarray | None,
        bf_caps: dict[str, int],
        pitch_count_caps: dict[str, int],
        pitch_eff: dict[str, float],
        gb_dp: dict[str, float],
        home_full: np.ndarray,
        away_full: np.ndarray,
        home_f5: np.ndarray,
        away_f5: np.ndarray,
        bat: dict,
        pit: dict,
    ) -> None:
        rng = self.rng
        # lineup pointers
        ptr = {"home": 0, "away": 0}
        # batters faced / pitches thrown by each team's starter (drives the hook)
        bf = {"home": 0, "away": 0}
        pitches = {"home": 0.0, "away": 0.0}

        def half(team: str, inning: int, walkoff_deficit: int | None = None) -> int:
            """Simulate one half-inning for ``team`` batting. Returns runs.

            ``walkoff_deficit`` is the number of runs the batting team trails by
            (0 when tied) in a half-inning that ends the game the moment it takes
            the lead. The half stops there rather than playing out three outs, so
            walk-off margins stay at the realistic +1 (or the runners-plus-batter
            total on a walk-off home run).
            """
            if team == "home":
                cdf_start, cdf_pen = home_cdf_start, home_cdf_pen
                cdf_pen_close = home_cdf_pen_close
                cdf_pen_bridge = home_cdf_pen_bridge
                pitch_team = "away"
            else:
                cdf_start, cdf_pen = away_cdf_start, away_cdf_pen
                cdf_pen_close = away_cdf_pen_close
                cdf_pen_bridge = away_cdf_pen_bridge
                pitch_team = "home"
            bf_cap = bf_caps[pitch_team]
            pitch_cap = pitch_count_caps[pitch_team]
            eff = pitch_eff[pitch_team]
            dp_rate = gb_dp[pitch_team]

            outs = 0
            bases: list[int] = [-1, -1, -1]  # slot index of runner or -1
            runs = 0
            while outs < 3:
                slot = ptr[team]
                # Starter stays in until EITHER the batters-faced or pitch-count
                # hook trips; then the bullpen takes over. In a still-close game
                # (|margin| <= CLOSE_MARGIN) the offense faces the pitching team's
                # bridge arms until the 8th and its high-leverage arms from there;
                # once it is out of hand, the aggregate/mop-up pen finishes.
                starter_in = bf[pitch_team] < bf_cap and pitches[pitch_team] < pitch_cap
                close = abs(home_runs - away_runs) <= CLOSE_MARGIN
                if starter_in:
                    cdf = cdf_start[slot]
                elif close and cdf_pen_bridge is not None and inning < LEVERAGE_INNING:
                    cdf = cdf_pen_bridge[slot]
                elif close and cdf_pen_close is not None:
                    cdf = cdf_pen_close[slot]
                else:
                    cdf = cdf_pen[slot]
                r = rng.random()
                oc = OUTCOMES[int(np.searchsorted(cdf, r))]
                bf[pitch_team] += 1
                ptr[team] = (slot + 1) % 9

                scored, rbi, outs, bases, dp = _apply_pa(oc, slot, outs, bases, dp_rate)
                runs += scored

                # batting stats
                b = bat[team]
                if oc in ("1B", "2B", "3B", "HR"):
                    b["H"][s, slot] += 1
                    b[oc][s, slot] += 1
                elif oc == "BB":
                    b["BB"][s, slot] += 1
                elif oc == "K":
                    b["K"][s, slot] += 1
                b["RBI"][s, slot] += rbi
                for runner_slot in scored_runners_holder:
                    b["R"][s, runner_slot] += 1
                scored_runners_holder.clear()

                # pitching stats (attributed to starter only while he's in)
                if starter_in:
                    pitches[pitch_team] += PITCH_COST[oc] * eff
                    p = pit[pitch_team]
                    if oc == "K":
                        p["K"][s] += 1
                    if oc in ("1B", "2B", "3B", "HR"):
                        p["H"][s] += 1
                    if oc == "BB":
                        p["BB"][s] += 1
                    if oc in ("K", "OUT"):
                        p["outs"][s] += 2 if dp else 1
                    p["ER"][s] += scored

                if walkoff_deficit is not None and runs > walkoff_deficit:
                    break
            return runs

        scored_runners_holder: list[int] = []

        def _apply_pa(oc: str, slot: int, outs: int, bases: list[int], dp_rate: float):
            """Advance runners. Returns (runs, rbi, outs, bases, dp)."""
            runs = 0
            rbi = 0
            if oc == "OUT":
                # Ground-ball double play: runner on first erased, two outs on
                # one ball in play (only with a runner on first and <2 outs).
                if dp_rate > 0.0 and bases[0] >= 0 and outs < 2 and rng.random() < dp_rate:
                    bases[0] = -1
                    outs += 2
                    return 0, 0, outs, bases, True
                outs += 1
                return 0, 0, outs, bases, False
            if oc == "K":
                outs += 1
                return 0, 0, outs, bases, False

            b2, b1, b0 = bases[2], bases[1], bases[0]  # 3rd, 2nd, 1st

            def score(runner: int) -> None:
                nonlocal runs, rbi
                if runner >= 0:
                    runs += 1
                    rbi += 1
                    scored_runners_holder.append(runner)

            if oc == "BB":
                # force advance only
                if b0 >= 0:
                    if b1 >= 0:
                        if b2 >= 0:
                            score(b2)
                        b2 = b1
                    b1 = b0
                b0 = slot
            elif oc == "1B":
                # 3rd & 2nd score, 1st -> 3rd, batter -> 1st
                score(b2)
                score(b1)
                b2 = b0 if b0 >= 0 else -1
                b1 = -1
                b0 = slot
            elif oc == "2B":
                score(b2)
                score(b1)
                if b0 >= 0:
                    score(b0)
                b2 = -1
                b1 = slot
                b0 = -1
            elif oc == "3B":
                score(b2)
                score(b1)
                score(b0)
                b2 = slot
                b1 = -1
                b0 = -1
            elif oc == "HR":
                score(b2)
                score(b1)
                score(b0)
                runs += 1
                rbi += 1
                scored_runners_holder.append(slot)
                b2 = b1 = b0 = -1

            bases[2], bases[1], bases[0] = b2, b1, b0
            return runs, rbi, outs, bases, False

        # 9 innings (or more for tie in full game); F5 = first 5.
        away_runs = 0
        home_runs = 0
        f5_home = 0
        f5_away = 0
        inning = 1
        while True:
            away_r = half("away", inning)
            away_runs += away_r
            if inning <= 5:
                f5_away += away_r
            # Home does not bat in the bottom of the 9th+ if already leading.
            if inning >= 9 and home_runs > away_runs:
                break
            # In the bottom of the 9th+ the game ends the instant the home team
            # takes the lead, so the half-inning is walk-off truncated.
            deficit = (away_runs - home_runs) if inning >= 9 else None
            home_r = half("home", inning, walkoff_deficit=deficit)
            home_runs += home_r
            if inning <= 5:
                f5_home += home_r
            if inning >= 9 and home_runs != away_runs:
                break
            if inning >= 15:  # safety cap
                break
            inning += 1

        home_full[s] = home_runs
        away_full[s] = away_runs
        home_f5[s] = f5_home
        away_f5[s] = f5_away
