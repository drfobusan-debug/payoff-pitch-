"""The screen's simulated night: the distribution, the park, and the exposure.

The simulator itself is tested elsewhere; what is new here is the screen's use of
it, so the tests pin the parts a refactor could invert without failing -- that
the mode is the modal night rather than the rounded mean, that the park scales the
hit types it is measured on and nothing else, that a lineup slot's turns come from
the starter's hook, and that two runs of the same screen agree.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from mlb_engine.data.parks import get_park
from mlb_engine.features.rolling import LEAGUE_RATES, OutcomeRates
from mlb_engine.models.matchup import apply_multipliers
from mlb_engine.models.montecarlo import MonteCarlo
from mlb_engine.output import power_sim

_AS_OF = date(2026, 8, 22)
_EMPTY_FRAME = pd.DataFrame(
    columns=["batter", "pitcher", "game_date", "events", "inning", "inning_topbot",
             "p_throws", "description"]
)
_LEAGUE = OutcomeRates(
    pa=600.0,
    p_1b=LEAGUE_RATES["1B"],
    p_2b=LEAGUE_RATES["2B"],
    p_3b=LEAGUE_RATES["3B"],
    p_hr=LEAGUE_RATES["HR"],
    p_bb=LEAGUE_RATES["BB"],
    p_k=LEAGUE_RATES["K"],
    p_out=LEAGUE_RATES["OUT"],
)


def test_the_distribution_reports_the_modal_night_not_the_rounded_mean() -> None:
    """Six nights: four with one hit, one with none, one with four.

    Mean 1.17, median 1, mode 1 -- and the mode is the number that matters,
    because it is the night that actually happens.
    """
    arr = np.array([1, 1, 1, 1, 0, 3], dtype=float)
    d = power_sim._distribution(arr, (0.5, 1.5))

    assert math.isclose(d.mean, 7 / 6)
    assert d.median == 1.0
    assert d.mode == 1.0
    assert math.isclose(d.over[0.5], 5 / 6)
    assert math.isclose(d.over[1.5], 1 / 6)
    assert math.isclose(d.p_any, 5 / 6)


def test_a_park_scales_the_hit_types_it_is_measured_on_and_no_others() -> None:
    coors = get_park(19)
    assert coors is not None
    mult = power_sim._park_multipliers(coors)

    assert mult["1B"] == coors.singles_factor
    assert mult["2B"] == coors.xbh_factor
    assert mult["3B"] == coors.xbh_factor
    assert "HR" not in mult
    assert "K" not in mult
    assert power_sim._park_multipliers(None) == {}


def test_the_doubles_park_produces_more_doubles_than_the_pitchers_park() -> None:
    """End to end through the simulator, on the same hitters.

    Coors and Chase differ by 24 points of extra-base factor, which is the whole
    reason the screen prints a doubles probability rather than an xwOBA.
    """
    coors = get_park(19)
    chase = get_park(15)
    assert coors is not None and chase is not None

    def doubles(park_venue: int) -> float:
        park = get_park(park_venue)
        assert park is not None
        mult = power_sim._park_multipliers(park)
        lineup = [apply_multipliers(dict(LEAGUE_RATES), mult) for _ in range(9)]
        cfg = power_sim.TeamSimConfig(bat_vs_starter=lineup, bat_vs_pen=lineup)
        neutral = power_sim.TeamSimConfig(
            bat_vs_starter=power_sim._league_nine(),
            bat_vs_pen=power_sim._league_nine(),
        )
        res = MonteCarlo(4000, seed=7).simulate(cfg, neutral)
        return float(res.bat["home"]["2B"].sum())

    assert doubles(19) > doubles(15)


def test_the_hook_changes_the_game_the_screen_simulates() -> None:
    """A short-hooked starter hands more plate appearances to the bullpen.

    The screen already prints a batters-faced cap and a pitch budget; the sim has
    to use them, or the note's exposure table and its probabilities disagree
    about when the starter left.
    """
    lineup = [dict(LEAGUE_RATES) for _ in range(9)]
    short = power_sim.TeamSimConfig(
        bat_vs_starter=lineup, bat_vs_pen=lineup, starter_bf_cap=12, starter_pitch_cap=60
    )
    long = power_sim.TeamSimConfig(
        bat_vs_starter=lineup, bat_vs_pen=lineup, starter_bf_cap=30, starter_pitch_cap=110
    )
    neutral = power_sim.TeamSimConfig(
        bat_vs_starter=power_sim._league_nine(), bat_vs_pen=power_sim._league_nine()
    )
    mc = MonteCarlo(2000, seed=11)
    short_res = mc.simulate(short, neutral)
    long_res = MonteCarlo(2000, seed=11).simulate(long, neutral)

    # Same hitters, same rates, same seed: the only difference is the hook, so
    # the two runs must not be the same game.
    assert not np.array_equal(short_res.pit["away"]["outs"], long_res.pit["away"]["outs"])


def test_a_tired_pen_is_charged_on_top_of_the_park_not_instead_of_it() -> None:
    """Park and workload are separate corrections and both have to land.

    The pen's fatigue penalty and the park's extra-base factor touch the same
    outcome keys, so the easy bug is one silently replacing the other.
    """
    park = {"1B": 1.05, "2B": 1.20, "3B": 1.20}
    pen = {"HR": 1.08, "1B": 1.04, "2B": 1.04}
    lineup = [(1, 0)]
    vs_sp, vs_pen, _, _ = power_sim._slot_vectors(
        lineup,
        _EMPTY_FRAME,
        _AS_OF,
        form_days=30,
        starter_hand="R",
        starter=_LEAGUE,
        pen=_LEAGUE,
        pen_leverage=_LEAGUE,
        pen_bridge=_LEAGUE,
        is_home=True,
        park_mult=park,
        pen_mult=pen,
    )
    # Against the pen the doubles rate carries park x fatigue; against the
    # starter it carries the park alone.
    assert vs_pen[0]["2B"] > vs_sp[0]["2B"] > 0
    assert vs_pen[0]["HR"] > vs_sp[0]["HR"]


def test_the_same_screen_run_twice_prints_the_same_probability() -> None:
    lineup = [dict(LEAGUE_RATES) for _ in range(9)]
    cfg = power_sim.TeamSimConfig(bat_vs_starter=lineup, bat_vs_pen=lineup)
    neutral = power_sim.TeamSimConfig(
        bat_vs_starter=power_sim._league_nine(), bat_vs_pen=power_sim._league_nine()
    )
    first = MonteCarlo(1000, seed=power_sim.SEED).simulate(cfg, neutral)
    second = MonteCarlo(1000, seed=power_sim.SEED).simulate(cfg, neutral)

    assert np.array_equal(first.bat["home"]["H"], second.bat["home"]["H"])


def test_a_league_average_nine_is_a_full_normalised_order() -> None:
    nine = power_sim._league_nine()
    assert len(nine) == 9
    for slot in nine:
        assert math.isclose(sum(slot.values()), 1.0, abs_tol=1e-6)


def test_a_fair_price_is_the_price_the_probability_is_worth() -> None:
    assert power_sim.fair_price(0.5) == "-100"
    assert power_sim.fair_price(0.75) == "-300"
    assert power_sim.fair_price(0.25) == "+300"
    assert power_sim.fair_price(0.0) == "&mdash;"
    assert power_sim.fair_price(1.0) == "&mdash;"
    assert power_sim.fair_price(math.nan) == "&mdash;"


def test_total_bases_is_the_weighted_sum_of_the_hit_types() -> None:
    bat = {
        "1B": np.array([[1]], dtype=np.int16),
        "2B": np.array([[1]], dtype=np.int16),
        "3B": np.array([[0]], dtype=np.int16),
        "HR": np.array([[1]], dtype=np.int16),
    }
    arr = power_sim._stat_array(bat, "TB", 0)
    assert arr is not None
    assert arr[0] == 1 + 2 + 4
    assert power_sim._stat_array(bat, "R", 0) is None
