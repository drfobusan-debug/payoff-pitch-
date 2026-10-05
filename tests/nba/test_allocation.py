import math

import pytest

from nba_engine.models.allocation import allocate, vacate

REB = {"C": 0.30, "PF": 0.20, "SF": 0.15, "SG": 0.10, "PG": 0.10, "bench": 0.15}


def test_player_rebounds_sum_to_the_team_total():
    split = allocate(40.0, REB)
    assert math.isclose(sum(split.amounts.values()), 40.0)
    assert split.unabsorbed == 0.0
    assert math.isclose(split.amounts["C"], 12.0)


def test_one_player_up_takes_from_his_teammates():
    base = allocate(40.0, REB).amounts
    boosted = allocate(40.0, {**REB, "C": 0.45}).amounts
    assert boosted["C"] > base["C"]
    assert all(boosted[p] < base[p] for p in REB if p != "C")
    assert math.isclose(sum(boosted.values()), 40.0)


def test_vacated_usage_respects_ceilings_and_flows_to_others():
    usage = {"star": 0.32, "second": 0.26, "guard": 0.16, "wing": 0.14, "stopper": 0.12}
    survivors = vacate(usage, {"star", "second"})
    fga = allocate(88.0, survivors, ceilings={"guard": 30.0, "wing": 26.0, "stopper": 14.0})
    assert fga.amounts["stopper"] <= 14.0 and "stopper" in fga.capped
    assert math.isclose(sum(fga.amounts.values()) + fga.unabsorbed, 88.0)
    assert fga.unabsorbed == pytest.approx(88.0 - 30.0 - 26.0 - 14.0)


def test_excess_is_redistributed_before_anything_is_unabsorbed():
    split = allocate(20.0, {"a": 0.5, "b": 0.5}, ceilings={"a": 5.0})
    assert split.amounts == pytest.approx({"a": 5.0, "b": 15.0})
    assert split.unabsorbed == 0.0


def test_zero_weight_and_zero_ceiling_get_nothing():
    split = allocate(10.0, {"a": 1.0, "b": 0.0, "c": 1.0}, ceilings={"c": 0.0})
    assert split.amounts == pytest.approx({"a": 10.0, "b": 0.0, "c": 0.0})


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        allocate(-1.0, REB)
    with pytest.raises(ValueError):
        allocate(1.0, {"a": -0.1})
