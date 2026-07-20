"""Monte Carlo game simulator.

Simulates a full game plate-appearance by plate-appearance using matchup-adjusted
outcome probabilities, tracking team runs (full game and first-5) plus per-lineup
batting lines and starter pitching lines so batter/pitcher props can be priced off
the empirical distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTCOMES = ["1B", "2B", "3B", "HR", "BB", "K", "OUT"]
IDX = {o: i for i, o in enumerate(OUTCOMES)}


@dataclass
class TeamSimConfig:
    """Everything needed to simulate one team's offense and its pitching."""

    # Offense: per lineup slot, matchup outcome probabilities vs the opposing
    # starter and vs a generic bullpen arm.
    bat_vs_starter: list[dict[str, float]]
    bat_vs_pen: list[dict[str, float]]
    # Pitching: this team's starter batters-faced cap before bullpen takes over.
    starter_bf_cap: int = 24


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

        for s in range(n):
            self._sim_one(
                s,
                home_cdf_start,
                home_cdf_pen,
                away_cdf_start,
                away_cdf_pen,
                away.starter_bf_cap,  # away pitching faces home hitters
                home.starter_bf_cap,  # home pitching faces away hitters
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
        away_pitch_cap: int,
        home_pitch_cap: int,
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
        # batters faced by each team's pitching (drives starter->pen switch)
        bf = {"home": 0, "away": 0}

        def half(team: str, inning: int) -> int:
            """Simulate one half-inning for ``team`` batting. Returns runs."""
            if team == "home":
                cdf_start, cdf_pen = home_cdf_start, home_cdf_pen
                pitch_team, cap = "away", away_pitch_cap
            else:
                cdf_start, cdf_pen = away_cdf_start, away_cdf_pen
                pitch_team, cap = "home", home_pitch_cap

            outs = 0
            bases: list[int] = [-1, -1, -1]  # slot index of runner or -1
            runs = 0
            while outs < 3:
                slot = ptr[team]
                starter_in = bf[pitch_team] < cap
                cdf = cdf_start[slot] if starter_in else cdf_pen[slot]
                r = rng.random()
                oc = OUTCOMES[int(np.searchsorted(cdf, r))]
                bf[pitch_team] += 1
                ptr[team] = (slot + 1) % 9

                scored, rbi, outs, bases = _apply_pa(oc, slot, outs, bases)
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
                    p = pit[pitch_team]
                    if oc == "K":
                        p["K"][s] += 1
                    if oc in ("1B", "2B", "3B", "HR"):
                        p["H"][s] += 1
                    if oc == "BB":
                        p["BB"][s] += 1
                    if oc in ("K", "OUT"):
                        p["outs"][s] += 1
                    p["ER"][s] += scored
            return runs

        scored_runners_holder: list[int] = []

        def _apply_pa(oc: str, slot: int, outs: int, bases: list[int]):
            """Advance runners. Returns (runs, rbi, outs, bases)."""
            runs = 0
            rbi = 0
            if oc == "OUT":
                outs += 1
                return 0, 0, outs, bases
            if oc == "K":
                outs += 1
                return 0, 0, outs, bases

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
            return runs, rbi, outs, bases

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
            home_r = half("home", inning)
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
