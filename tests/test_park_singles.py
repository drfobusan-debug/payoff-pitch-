"""The ballpark on the singles line, which used to be a runs factor or nothing."""

from __future__ import annotations

import numpy as np

from mlb_engine.data.parks import PARKS, get_park
from mlb_engine.models.matchup import apply_multipliers
from mlb_engine.pipeline import Pipeline

YANKEE, BUSCH, COORS, DODGER = 3313, 2889, 19, 22


def test_every_park_carries_a_singles_factor_near_neutral() -> None:
    f = np.array([p.singles_factor for p in PARKS.values()])
    assert len(f) == 30
    # Shrunk to a 0.40 split-half reliability, so the spread is a few percent.
    assert 0.94 <= f.min() and f.max() <= 1.04
    assert abs(f.mean() - 1.0) < 0.01


def test_the_singles_factor_is_not_the_runs_factor() -> None:
    """Runs are mostly home runs; the two disagree at the extremes."""
    runs = np.array([p.park_factor for p in PARKS.values()])
    singles = np.array([p.singles_factor for p in PARKS.values()])
    assert abs(np.corrcoef(runs, singles)[0, 1]) < 0.3
    # Yankee Stadium scores hitter-friendly on runs and suppresses singles.
    assert PARKS[YANKEE].park_factor > 100.0
    assert PARKS[YANKEE].singles_factor < 1.0
    # Busch is the reverse.
    assert PARKS[BUSCH].park_factor < 100.0
    assert PARKS[BUSCH].singles_factor > 1.0


def test_the_park_moves_the_simulated_singles_rate() -> None:
    """Coors and Dodger Stadium used to price a single identically."""
    rates = {"1B": 0.15, "2B": 0.05, "3B": 0.005, "HR": 0.04,
             "BB": 0.08, "K": 0.22, "OUT": 0.455}
    coors = apply_multipliers(rates, {"1B": PARKS[COORS].singles_factor})
    dodger = apply_multipliers(rates, {"1B": PARKS[DODGER].singles_factor})
    assert coors["1B"] > rates["1B"] > dodger["1B"]
    assert coors["1B"] / dodger["1B"] > 1.04
    # Renormalised, so the extra singles come out of the other outcomes.
    assert abs(sum(coors.values()) - 1.0) < 1e-9
    assert coors["HR"] < rates["HR"]


def test_the_gate_context_reads_the_singles_factor() -> None:
    ctx = Pipeline._hits_context(get_park(YANKEE))
    assert ctx == PARKS[YANKEE].singles_factor
    # The old behaviour bought an average bat here on a 102 runs factor.
    assert ctx < 1.0
    assert Pipeline._hits_context(get_park(BUSCH)) > 1.0


def test_an_unknown_park_leaves_the_gate_neutral() -> None:
    assert Pipeline._hits_context(None) is None


def test_the_switch_turns_the_park_term_off(monkeypatch) -> None:
    from mlb_engine.config import Config

    monkeypatch.delenv("MLBE_PARK_SINGLES", raising=False)
    assert Config().park_singles is True
    monkeypatch.setenv("MLBE_PARK_SINGLES", "0")
    assert Config().park_singles is False
