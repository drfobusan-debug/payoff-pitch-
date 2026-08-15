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
    # flat down the order (slot alone fits to -0.001), the wrong-handed bat's is not.
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
    """Removal is absorbing, and it starts after his first turn.

    Two facts in one test, because they are the same accounting. A man in the
    batting order takes his first appearance -- over 27,090 slot-games the starter
    has it 100.0% of the time, since being pinch-hit *for* presupposes he was due
    up -- and once he has been lifted nothing comes back to him. So with certain
    removal his line is exactly one plate appearance, every time.

    Every outcome here is a single or a strikeout, both of which the batter arrays
    record, so ``1B + K`` per slot IS that slot's credited plate appearances.
    """
    counted = {"1B": 0.3, "K": 0.7, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0,
               "OUT": 0.0}
    always = RemovalHazard(intercept=50.0)
    lifted = TeamSimConfig(
        bat_vs_starter=[dict(counted) for _ in range(9)],
        bat_vs_pen=[dict(counted) for _ in range(9)],
        bat_hands=("R",) * 9,
        removal_hazard=always,
        bat_replacement=dict(counted),
    )
    # The bullpen from the first batter, so he is lifted the moment he is at risk.
    res = MonteCarlo(200, seed=3).simulate(
        lifted, _team(starter_bf_cap=0, starter_hand="R")
    )
    pa = res.bat["home"]["1B"] + res.bat["home"]["K"]
    assert pa.max() == 1  # no second turn ever lands on him
    for slot in range(9):  # and no slot is erased before the game reaches it
        assert float(pa[:, slot].mean()) > 0.97, slot
    # The offense itself is intact: somebody batted, and the game was played.
    assert res.home_runs_full.sum() > 0
    assert res.bat["away"]["H"].sum() > 0  # and the other lineup is untouched


def test_removal_moves_hits_down_without_touching_the_rate_vector() -> None:
    """The point of the branch: fewer chances, not a worse hitter.

    The rate vectors handed in are identical; only the chance of being lifted
    differs, and that alone has to move the hits distribution toward the Under.

    Measured rather than sampled loosely, because the effect is small on purpose:
    the hazard costs a hitter ~4.5% of his turns, so at the top of the order a
    slot's loss is 0.05 of a plate appearance and Monte Carlo noise on a
    thousand-game sample is 0.03 of one. Every outcome here is a single or a
    strikeout, both recorded, so ``1B + K`` is the exact credited count and the
    comparison does not rest on a subset of outcomes.
    """
    counted = {"1B": 0.3, "K": 0.7, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0,
               "OUT": 0.0}

    def bats(**kw) -> TeamSimConfig:
        return TeamSimConfig(
            bat_vs_starter=[dict(counted) for _ in range(9)],
            bat_vs_pen=[dict(counted) for _ in range(9)],
            bat_replacement=dict(counted),  # same bat, so only attribution differs
            **kw,
        )

    hands = ("L",) * 9
    lhp = _team(starter_hand="L")
    base = MonteCarlo(20_000, seed=5).simulate(bats(bat_hands=hands), lhp)
    rem = MonteCarlo(20_000, seed=5).simulate(
        bats(bat_hands=hands, removal_hazard=HAZ), lhp
    )
    # The same nine against a right-hander: the platoon edge, same rate vectors.
    ok = MonteCarlo(20_000, seed=5).simulate(
        bats(bat_hands=hands, removal_hazard=HAZ), _team(starter_hand="R")
    )

    def pa(res, slot: int) -> float:
        return float((res.bat["home"]["1B"][:, slot] + res.bat["home"]["K"][:, slot]).mean())

    def p_hit(res, slot: int) -> float:
        return float((res.bat["home"]["1B"][:, slot] >= 1).mean())

    for slot in range(9):
        assert pa(rem, slot) < pa(base, slot), slot
        assert p_hit(rem, slot) < p_hit(base, slot), slot
    # The bottom of the order pays most of it: 8.0% of the 9-hole's appearances
    # go to somebody else against 3.1% of the leadoff man's.
    lost = [pa(base, s) - pa(rem, s) for s in range(9)]
    assert lost[8] > lost[0]
    # And it is handedness that costs him the turns: the wrong-handed bat loses
    # more of them than the same hitter with the platoon edge.
    assert sum(pa(rem, i) for i in range(9)) < sum(pa(ok, i) for i in range(9))


def test_the_substitute_is_a_worse_bat_than_the_league() -> None:
    """Measured off 1,915 substitute plate appearances, not assumed."""
    assert abs(sum(SUB_RATES.values()) - 1.0) < 1e-9
    assert SUB_RATES["1B"] < 0.1417  # league singles rate
    assert SUB_RATES["K"] > 0.2228  # league strikeout rate
