"""The validation harness itself: what "matches the histogram" is measured as."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.nfl.score_shape_study import (
    _push_by_spread,
    _summary,
    actual_summary,
    simulate_slate,
)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "spread_line": [3.0, -3.0, 6.5],
            "total_line": [44.5, 47.0, 41.0],
            "result": [3.0, -7.0, 6.0],
            "total": [45.0, 47.0, 41.0],
        }
    )


def test_a_push_needs_the_sign_not_just_the_number():
    """Half of |margin| = 3 games push a 3-point spread; the other half win it.

    Counting both signs doubles the measured push rate, which would make every
    half-point look twice as valuable as it is.
    """
    margin = np.array([3.0, -3.0, 3.0, -3.0])
    spread = np.array([3.0, 3.0, 3.0, 3.0])
    assert _push_by_spread(margin, spread)[3] == 0.5


def test_half_point_spreads_can_never_push():
    margin = np.array([3.0, -3.0])
    assert np.isnan(_push_by_spread(margin, np.array([3.5, 3.5]))[3])


def test_the_actual_summary_reads_the_history_the_same_way():
    games = _games()
    actual = actual_summary(games)
    # spread_line is the home handicap with the sign flipped: +3.0 means the home
    # team is favoured by 3, and a 3-point home win is the push.
    assert actual["push_by_spread"][3] == 0.5
    assert actual["hist"][3] == 1 / 3
    # 45 clears 44.5; the other two land exactly on their whole number, so they
    # push rather than going over.
    assert actual["over"] == 1 / 3
    assert abs(actual["whole_total_push"] - 1.0) < 1e-9


def test_the_simulated_slate_is_read_by_the_same_summary():
    sim = simulate_slate(_games(), engine="drives", n_sims=400, seed=3)
    out = _summary(
        sim.margin.to_numpy(),
        sim.total.to_numpy(),
        sim.spread_line.to_numpy(),
        sim.total_line.to_numpy(),
    )
    assert out["push_by_spread"][3] > 0.0
    assert np.isnan(out["push_by_spread"][7])  # no 7-point game in the sample
    assert 30.0 < out["total_mean"] < 60.0
