"""The ballpark on the doubles line, which nothing reached before.

The HR channel has its own park multiplier and #110 gave singles theirs, so
2B/3B were the one hit type priced identically everywhere -- despite outfield
geometry varying more between parks than fence distance does.
"""

from __future__ import annotations

import numpy as np

from mlb_engine.config import Config
from mlb_engine.data.parks import PARKS
from mlb_engine.models.matchup import apply_multipliers

COORS, WRIGLEY, KAUFFMAN, FENWAY = 19, 17, 7, 3


def test_every_park_carries_an_extra_base_factor() -> None:
    f = np.array([p.xbh_factor for p in PARKS.values()])
    assert len(f) == 30
    # Shrunk to a 0.59 alternate-day reliability, so wider than singles' few
    # percent but still well inside the raw 0.77..1.42 measurement.
    assert 0.85 <= f.min() and f.max() <= 1.26
    assert abs(f.mean() - 1.0) < 0.02


def test_it_is_wider_than_the_singles_factor() -> None:
    """Doubles depend on the outfield; singles mostly on where fielders stand."""
    xbh = np.array([p.xbh_factor for p in PARKS.values()])
    singles = np.array([p.singles_factor for p in PARKS.values()])
    assert xbh.std() > 3 * singles.std()


def test_it_is_not_a_restatement_of_the_runs_factor() -> None:
    runs = np.array([p.park_factor for p in PARKS.values()])
    xbh = np.array([p.xbh_factor for p in PARKS.values()])
    assert abs(np.corrcoef(runs, xbh)[0, 1]) < 0.6
    # Kauffman: a neutral runs factor hiding the deepest alleys in baseball.
    assert PARKS[KAUFFMAN].park_factor == 100.0
    assert PARKS[KAUFFMAN].xbh_factor > 1.10
    # The two extremes, and the runs factor has them almost level (112 vs 103).
    assert PARKS[COORS].xbh_factor > 1.20
    assert PARKS[WRIGLEY].xbh_factor < 0.90


def test_the_wall_and_the_altitude_both_add_doubles() -> None:
    assert PARKS[FENWAY].xbh_factor > 1.0
    assert PARKS[COORS].xbh_factor > PARKS[FENWAY].xbh_factor


def test_the_park_moves_the_simulated_extra_base_rate() -> None:
    rates = {"1B": 0.15, "2B": 0.05, "3B": 0.005, "HR": 0.04,
             "BB": 0.08, "K": 0.22, "OUT": 0.455}
    coors = apply_multipliers(rates, {"2B": PARKS[COORS].xbh_factor,
                                      "3B": PARKS[COORS].xbh_factor})
    wrigley = apply_multipliers(rates, {"2B": PARKS[WRIGLEY].xbh_factor,
                                        "3B": PARKS[WRIGLEY].xbh_factor})
    assert coors["2B"] > rates["2B"] > wrigley["2B"]
    assert coors["2B"] / wrigley["2B"] > 1.35
    # Renormalised, so the extra doubles come out of the other outcomes.
    assert abs(sum(coors.values()) - 1.0) < 1e-9
    assert coors["1B"] < rates["1B"]


def test_the_switch_turns_the_park_term_off(monkeypatch) -> None:
    monkeypatch.delenv("MLBE_PARK_XBH", raising=False)
    assert Config().park_xbh is True
    monkeypatch.setenv("MLBE_PARK_XBH", "0")
    assert Config().park_xbh is False
