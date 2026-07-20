"""Markov base-out model for first-5-inning (F5) run distributions.

The half-inning is a Markov chain over the 24 base-out states. Two refinements
requested for F5 accuracy are supported:

* **Per-lineup-slot rates** — instead of one averaged team distribution, the
  chain is driven by the exact 9 batters' L/R-split rates, with a lineup pointer
  that carries across innings (leadoff of an inning is whoever is due up).
* **Non-stationary / times-through-order (TTO)** — the starter's suppression
  decays as the lineup turns over; offense outcomes get a small TTO boost the
  2nd/3rd time through, computed exactly by tracking the times-through counter as
  a state dimension.

A stationary, single-distribution wrapper (`f5_from_rates`) is kept for callers
that only have team-average rates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlb_engine.models.matchup import apply_multipliers
from mlb_engine.models.montecarlo import OUTCOMES

RUN_CAP = 20  # max runs tracked per inning
N_INNINGS = 5
ON_BASE = ("1B", "2B", "3B", "HR", "BB")

# Offense multiplier by times-through-the-order (pitcher fatigue / familiarity).
DEFAULT_TTO_FACTORS = (1.0, 1.03, 1.08, 1.10)

# base state is a 3-bit mask: bit0=1B, bit1=2B, bit2=3B


def _advance(base: int, outs: int, oc: str) -> tuple[int, int, int]:
    """Return (new_base, new_outs, runs) for an outcome from (base, outs)."""
    on1 = base & 1
    on2 = (base >> 1) & 1
    on3 = (base >> 2) & 1
    runs = 0

    if oc in ("K", "OUT"):
        return base, outs + 1, 0

    if oc == "BB":
        if on1:
            if on2:
                if on3:
                    runs += 1  # forced in
                on3 = 1
            on2 = 1
        on1 = 1
    elif oc == "1B":
        runs += on3 + on2
        on3 = on1
        on2 = 0
        on1 = 1
    elif oc == "2B":
        runs += on3 + on2 + on1
        on3 = 0
        on2 = 1
        on1 = 0
    elif oc == "3B":
        runs += on3 + on2 + on1
        on3 = 1
        on2 = 0
        on1 = 0
    elif oc == "HR":
        runs += on3 + on2 + on1 + 1
        on3 = on2 = on1 = 0

    new_base = (on3 << 2) | (on2 << 1) | on1
    return new_base, outs, runs


@dataclass
class F5Result:
    home_dist: np.ndarray  # P(home F5 runs = k)
    away_dist: np.ndarray
    total_dist: np.ndarray  # P(home+away F5 runs = k)
    p_home_ml: float  # P(home > away through 5)
    p_away_ml: float
    p_tie: float

    def p_total_over(self, line: float) -> float:
        # push handled by splitting at .5 lines; for X.5 lines no push.
        ks = np.arange(len(self.total_dist))
        return float(self.total_dist[ks > line].sum())

    def p_home_cover(self, spread: float) -> float:
        """P(home_margin > spread). spread e.g. -0.5 for home -0.5."""
        # rebuild joint quickly
        return _p_margin_gt(self.home_dist, self.away_dist, spread)


def _inning_distribution(prob: dict[str, float]) -> np.ndarray:
    """Exact run distribution for a single half-inning."""
    p = np.array([prob[o] for o in OUTCOMES], dtype=float)
    p = p / p.sum()
    # dp[base, outs] = vector over runs-so-far (len RUN_CAP+1)
    dp = np.zeros((8, 3, RUN_CAP + 1), dtype=float)
    dp[0, 0, 0] = 1.0
    out_dist = np.zeros(RUN_CAP + 1, dtype=float)  # runs when inning ends (3 outs)

    # Iterate PAs until mass is negligible.
    for _ in range(60):
        new_dp = np.zeros_like(dp)
        moved = False
        for base in range(8):
            for outs in range(3):
                mass = dp[base, outs]
                if mass.sum() < 1e-12:
                    continue
                moved = True
                for oi, oc in enumerate(OUTCOMES):
                    pr = p[oi]
                    if pr <= 0:
                        continue
                    nb, no, r = _advance(base, outs, oc)
                    shifted = _shift(mass, r) * pr
                    if no >= 3:
                        out_dist += shifted
                    else:
                        new_dp[nb, no] += shifted
        dp = new_dp
        if not moved:
            break
    # add any residual mass (safety)
    out_dist += dp.sum(axis=(0, 1))
    total = out_dist.sum()
    if total > 0:
        out_dist /= total
    return out_dist


def _shift(vec: np.ndarray, r: int) -> np.ndarray:
    if r == 0:
        return vec
    out = np.zeros_like(vec)
    if r <= RUN_CAP:
        out[r:] = vec[: len(vec) - r]
        # runs beyond cap accumulate at cap
        out[RUN_CAP] += vec[len(vec) - r :].sum()
    else:
        out[RUN_CAP] += vec.sum()
    return out


def _convolve_n(dist: np.ndarray, n: int) -> np.ndarray:
    out = dist.copy()
    for _ in range(n - 1):
        out = np.convolve(out, dist)
    return out


def _p_margin_gt(home: np.ndarray, away: np.ndarray, spread: float) -> float:
    tot = 0.0
    for h, ph in enumerate(home):
        if ph <= 0:
            continue
        for a, pa in enumerate(away):
            if pa <= 0:
                continue
            if (h - a) > spread:
                tot += ph * pa
    return tot


def _slot_prob_table(
    slots: list[dict[str, float]],
    tto_factors: tuple[float, ...],
) -> dict[tuple[int, int], np.ndarray]:
    """(batter_idx, tto) -> normalized prob vector over OUTCOMES, TTO-boosted."""
    table: dict[tuple[int, int], np.ndarray] = {}
    for bi, rates in enumerate(slots):
        for tto, factor in enumerate(tto_factors):
            adj = apply_multipliers(rates, {k: factor for k in ON_BASE}) if factor != 1.0 else rates
            arr = np.array([adj[o] for o in OUTCOMES], dtype=float)
            s = arr.sum()
            table[(bi, tto)] = arr / s if s > 0 else arr
    return table


def team_f5_distribution(
    slots: list[dict[str, float]],
    tto_factors: tuple[float, ...] = DEFAULT_TTO_FACTORS,
) -> np.ndarray:
    """F5 run distribution for a lineup, tracking batting order + TTO across innings."""
    if len(slots) != 9:
        raise ValueError("need exactly 9 lineup slots")
    table = _slot_prob_table(slots, tto_factors)
    max_tto = len(tto_factors) - 1

    zero = np.zeros(RUN_CAP + 1)
    start = zero.copy()
    start[0] = 1.0
    # state: (inning, outs, base, batter, tto)
    dp: dict[tuple[int, int, int, int, int], np.ndarray] = {(0, 0, 0, 0, 0): start}
    final = zero.copy()

    for _ in range(200):
        if not dp:
            break
        new_dp: dict[tuple[int, int, int, int, int], np.ndarray] = {}
        for (inn, outs, base, bi, tto), mass in dp.items():
            probs = table[(bi, tto)]
            nb_idx = (bi + 1) % 9
            ntto = min(tto + 1, max_tto) if nb_idx == 0 else tto
            for oi, oc in enumerate(OUTCOMES):
                pr = probs[oi]
                if pr <= 0:
                    continue
                nbase, nouts, runs = _advance(base, outs, oc)
                shifted = _shift(mass, runs) * pr
                if nouts >= 3:
                    ninn = inn + 1
                    if ninn >= N_INNINGS:
                        final += shifted
                    else:
                        key = (ninn, 0, 0, nb_idx, ntto)
                        _accum(new_dp, key, shifted)
                else:
                    key = (inn, nouts, nbase, nb_idx, ntto)
                    _accum(new_dp, key, shifted)
        dp = new_dp
    for mass in dp.values():  # residual safety
        final += mass
    s = final.sum()
    return final / s if s > 0 else final


def _accum(d: dict, key, vec: np.ndarray) -> None:
    if key in d:
        d[key] += vec
    else:
        d[key] = vec.copy()


def _combine(home5: np.ndarray, away5: np.ndarray) -> F5Result:
    total = np.convolve(home5, away5)
    p_home = _p_margin_gt(home5, away5, 0.0)
    p_away = _p_margin_gt(away5, home5, 0.0)
    p_tie = max(0.0, 1.0 - p_home - p_away)
    return F5Result(
        home_dist=home5,
        away_dist=away5,
        total_dist=total,
        p_home_ml=p_home,
        p_away_ml=p_away,
        p_tie=p_tie,
    )


def f5_from_lineups(
    home_slots: list[dict[str, float]],
    away_slots: list[dict[str, float]],
    tto_factors: tuple[float, ...] = DEFAULT_TTO_FACTORS,
) -> F5Result:
    """Non-stationary, per-slot F5 model (preferred)."""
    home5 = team_f5_distribution(home_slots, tto_factors)
    away5 = team_f5_distribution(away_slots, tto_factors)
    return _combine(home5, away5)


def f5_from_rates(home_pa: dict[str, float], away_pa: dict[str, float]) -> F5Result:
    """Stationary team-average F5 model (independent innings)."""
    home5 = _convolve_n(_inning_distribution(home_pa), 5)
    away5 = _convolve_n(_inning_distribution(away_pa), 5)
    return _combine(home5, away5)
