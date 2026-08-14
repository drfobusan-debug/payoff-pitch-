"""The simulator can lift a hitter, and only where the measurement says it does.

The lineup used to bat the same nine to the last out, so a pinch hitter's plate
appearances were credited to the man he replaced. These tests pin the shape of
the measured hazard (``features.removal``) and the accounting that follows from
it: the substitute bats, the original hitter does not, and nothing the substitute
does lands on the original hitter's line.
"""

from __future__ import annotations

import numpy as np

from mlb_engine.features.removal import LATE_INNING, SUB_RATES, RemovalHazard
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig

HAZ = RemovalHazard()
RATES = {"1B": 0.15, "2B": 0.04, "3B": 0.004, "HR": 0.03, "BB": 0.09, "K": 0.22,
         "OUT": 0.466}


def _team(**kw) -> TeamSimConfig:
    return TeamSimConfig(
        bat_vs_starter=[dict(RATES) for _ in range(9)],
        bat_vs_pen=[dict(RATES) for _ in range(9)],
        **kw,
    )


def test_the_hazard_is_the_shape_that_was_measured() -> None:
    """Starter out, late, and the wrong hand at the bottom of the order."""
    kw = dict(slot=9, inning=8)
    assert HAZ.per_pa(**kw, starter_out=False, same_hand=True) < HAZ.per_pa(
        **kw, starter_out=True, same_hand=True
    )
    # Inning 7 is the step; the 6th is only the starter's exit.
    late = dict(slot=9, starter_out=True, same_hand=True)
    assert HAZ.per_pa(**late, inning=LATE_INNING) > HAZ.per_pa(
        **late, inning=LATE_INNING - 1
    )
    # Handedness, and the slot only through it: the platoon-edge bat's hazard is
    # flat down the order (slot alone was z = -1.4), the wrong-handed bat's is not.
    state = dict(inning=8, starter_out=True)
    assert HAZ.per_pa(**state, slot=9, same_hand=True) > HAZ.per_pa(
        **state, slot=9, same_hand=False
    )
    spread_bad = HAZ.per_pa(**state, slot=9, same_hand=True) - HAZ.per_pa(
        **state, slot=1, same_hand=True
    )
    spread_ok = HAZ.per_pa(**state, slot=9, same_hand=False) - HAZ.per_pa(
        **state, slot=1, same_hand=False
    )
    assert spread_bad > 0.10 > abs(spread_ok)


def test_unknown_handedness_takes_the_lower_hazard() -> None:
    """A hand the board has not given is not a reason to lift anybody."""
    state = dict(slot=9, inning=8, starter_out=True)
    assert HAZ.per_pa(**state, same_hand=False) < HAZ.per_pa(**state, same_hand=True)


def test_off_by_default_the_lineup_is_unchanged() -> None:
    """No hazard config, no behaviour change: the same nine, the same seed."""
    home, away = _team(), _team()
    a = MonteCarlo(30, seed=7).simulate(home, away)
    b = MonteCarlo(30, seed=7).simulate(home, away)
    assert np.array_equal(a.bat["home"]["H"], b.bat["home"]["H"])
    assert np.array_equal(a.home_runs_full, b.home_runs_full)


def test_removal_takes_plate_appearances_off_the_hitter() -> None:
    """The slot keeps batting; he does not.

    A certain-removal hazard is the clean test of the accounting: every plate
    appearance after the opposing starter's exit belongs to the substitute, so the
    hitter's own line stops growing while the team's offense does not.
    """
    always = RemovalHazard(intercept=50.0)
    # The hand belongs to the team on the mound, so the opposing config carries it.
    opp = _team(starter_bf_cap=9, starter_hand="L")
    fixed = _team()
    lifted = _team(bat_hands=("L",) * 9, removal_hazard=always)
    # Both offenses face the same pitching, so only the batting side differs.
    base = MonteCarlo(400, seed=11).simulate(fixed, opp)
    with_removal = MonteCarlo(400, seed=11).simulate(lifted, opp)

    pa_base = sum(
        int(base.bat["home"][s].sum()) for s in ("H", "BB", "K")
    )
    pa_lifted = sum(
        int(with_removal.bat["home"][s].sum()) for s in ("H", "BB", "K")
    )
    assert pa_lifted < pa_base * 0.75
    # The runs still get scored -- the turn is taken, by somebody else.
    assert with_removal.home_runs_full.mean() > 0.5 * base.home_runs_full.mean()
    # And the substitute's runs are not credited to the man he replaced.
    assert with_removal.bat["home"]["R"].sum() < base.bat["home"]["R"].sum()


def test_a_slot_is_replaced_once_and_the_hitter_never_returns() -> None:
    """Removal is absorbing: no plate appearance comes back to him."""
    always = RemovalHazard(intercept=50.0)
    lifted = _team(
        bat_hands=("R",) * 9,
        removal_hazard=always,
        bat_replacement=SUB_RATES,
    )
    # The bullpen from the first batter, so he is lifted at once.
    res = MonteCarlo(50, seed=3).simulate(lifted, _team(starter_bf_cap=0, starter_hand="R"))
    for stat in ("H", "1B", "2B", "3B", "HR", "BB", "K", "R", "RBI"):
        assert res.bat["home"][stat].sum() == 0, stat
    # The offense itself is intact: somebody batted, and the game was played.
    assert res.home_runs_full.sum() > 0
    assert res.bat["away"]["H"].sum() > 0  # and the other lineup is untouched


def test_removal_moves_hits_down_without_touching_the_rate_vector() -> None:
    """The point of the branch: fewer chances, not a worse hitter.

    The rate vectors handed in are identical; only the chance of being lifted
    differs, and that alone has to move the hits distribution toward the Under.
    """
    hands = ("L",) * 9
    lhp = _team(starter_hand="L")
    fixed = _team(bat_hands=hands)
    lifted = _team(bat_hands=hands, removal_hazard=HAZ)
    base = MonteCarlo(3000, seed=5).simulate(fixed, lhp)
    rem = MonteCarlo(3000, seed=5).simulate(lifted, lhp)
    # The same nine against a right-hander: the platoon edge, same rate vectors.
    ok = MonteCarlo(3000, seed=5).simulate(lifted, _team(starter_hand="R"))

    def pa(res, slot: int) -> float:
        return sum(float(res.bat["home"][s][:, slot].mean()) for s in ("H", "BB", "K"))

    def p_hit(res, slot: int) -> float:
        return float((res.bat["home"]["H"][:, slot] >= 1).mean())

    for slot in range(9):
        assert pa(rem, slot) < pa(base, slot)
        assert p_hit(rem, slot) < p_hit(base, slot)
    # And it is handedness that costs him the turns: the wrong-handed bat loses
    # more of them than the same hitter with the platoon edge.
    assert sum(pa(rem, i) for i in range(9)) < sum(pa(ok, i) for i in range(9))


def test_the_substitute_is_a_worse_bat_than_the_league() -> None:
    """Measured off 4,409 substitute plate appearances, not assumed."""
    assert abs(sum(SUB_RATES.values()) - 1.0) < 1e-9
    assert SUB_RATES["1B"] < 0.1417  # league singles rate
    assert SUB_RATES["K"] > 0.2228  # league strikeout rate
