"""How many plate appearances the removal branch actually takes off a hitter.

The hazard is measured; this is the consequence. Runs the same lineup twice --
fixed nine to the last out, then with the branch on -- and reports expected plate
appearances per slot, the share of games he gets three or fewer, and what that
does to his 1+ hit probability. Three PA against four is 13.5 points of no-single
in the graded panel, so the PA column is the one that matters.

A hitter's own plate appearances are counted from the stats credited to him
(``(H + BB + K) / (p_h + p_bb + p_k)``, exact in expectation because a substitute's
production is never credited to him), which is also the number the props read.
"""

from __future__ import annotations

import numpy as np

from mlb_engine.features.removal import RemovalHazard
from mlb_engine.features.rolling import LEAGUE_RATES
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig

N_SIMS = 6000
SEED = 17
# A league-average nine, so the difference between the two runs is the branch and
# nothing else. Hands are a typical lineup: four left-handed bats, five right.
HANDS = ("L", "R", "L", "R", "R", "L", "R", "R", "L")
CREDITED = LEAGUE_RATES["1B"] + LEAGUE_RATES["2B"] + LEAGUE_RATES["3B"] + (
    LEAGUE_RATES["HR"] + LEAGUE_RATES["BB"] + LEAGUE_RATES["K"]
)


def _team(**kw) -> TeamSimConfig:
    rates = [dict(LEAGUE_RATES) for _ in range(9)]
    return TeamSimConfig(bat_vs_starter=rates, bat_vs_pen=[dict(r) for r in rates], **kw)


def _pa(res, slot: int) -> np.ndarray:
    """Plate appearances credited to the slot's original occupant, per sim."""
    got = sum(res.bat["home"][s][:, slot].astype(float) for s in ("H", "BB", "K"))
    return got / CREDITED


def main() -> None:
    for opp_hand in ("R", "L"):
        opp = _team(starter_hand=opp_hand)
        fixed = MonteCarlo(N_SIMS, seed=SEED).simulate(_team(bat_hands=HANDS), opp)
        rem = MonteCarlo(N_SIMS, seed=SEED).simulate(
            _team(bat_hands=HANDS, removal_hazard=RemovalHazard()), opp
        )
        print(f"\nvs {opp_hand}HP  ({N_SIMS:,} sims, league-average nine)")
        print(
            f"{'slot':<6}{'hand':<6}{'PA off':>8}{'PA on':>8}{'lost':>8}"
            f"{'<=3 PA off':>12}{'<=3 PA on':>11}{'p(1+H) off':>12}{'p(1+H) on':>11}"
        )
        for slot in range(9):
            off, on = _pa(fixed, slot), _pa(rem, slot)
            h_off = float((fixed.bat["home"]["H"][:, slot] >= 1).mean())
            h_on = float((rem.bat["home"]["H"][:, slot] >= 1).mean())
            bad = "*" if HANDS[slot] == opp_hand else " "
            print(
                f"{slot + 1:<6}{HANDS[slot] + bad:<6}{off.mean():8.2f}{on.mean():8.2f}"
                f"{off.mean() - on.mean():8.2f}"
                f"{float((off <= 3.4).mean()):12.1%}{float((on <= 3.4).mean()):11.1%}"
                f"{h_off:12.3f}{h_on:11.3f}"
            )
        tot_off = sum(_pa(fixed, s).mean() for s in range(9))
        tot_on = sum(_pa(rem, s).mean() for s in range(9))
        print(f"{'team':<12}{tot_off:8.2f}{tot_on:8.2f}{tot_off - tot_on:8.2f}")


if __name__ == "__main__":
    main()
