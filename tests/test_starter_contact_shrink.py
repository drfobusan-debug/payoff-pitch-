"""Empirical-Bayes shrinkage on the contact quality a starter allows.

Split-half across adjacent six-week blocks (112 pitcher-pairs): xwOBA repeats at
0.31, hard-hit 0.24, BABIP 0.10, barrel 0.09 — against K% 0.52 and CSW 0.50 for
the command signals. The contact group drives the hit and HR multipliers, so it
is the group that gets pulled toward league; the command group is left raw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_engine.config import Config, RollingWindows
from mlb_engine.features.regression import (
    BL_BABIP,
    BL_BARREL_ALLOWED,
    BL_HARD_HIT,
    STARTER_PRIOR_BBE,
    build_pitcher_regression,
    shrink_starter_rate,
)


def test_shrink_is_a_weighted_walk_toward_the_baseline() -> None:
    k = STARTER_PRIOR_BBE["xwoba"]
    assert shrink_starter_rate(0.400, 0.300, 100, k, strength=0.0) == 0.400
    # strength 1.0 at n == k keeps exactly half the deviation.
    assert shrink_starter_rate(0.400, 0.300, int(k), k) == 0.350
    # More batted balls -> more of the deviation survives.
    assert shrink_starter_rate(0.400, 0.300, 400, k) > shrink_starter_rate(0.400, 0.300, 50, k)
    # It is symmetric: a good rate looks less good, a bad one less bad.
    assert shrink_starter_rate(0.200, 0.300, 106, k) > 0.200
    assert shrink_starter_rate(0.400, 0.300, 106, k) < 0.400
    # An empty sample has nothing to shrink.
    assert shrink_starter_rate(0.400, 0.300, 0, k) == 0.400


def test_measured_weights_reproduce_the_split_half_reliabilities() -> None:
    """At the mean six-week sample the kept share IS the metric's reliability."""
    for metric, n, expected in (("xwoba", 106, 0.31), ("hard_hit", 106, 0.24),
                                ("babip", 106, 0.10), ("barrel", 106, 0.09)):
        keep = n / (n + STARTER_PRIOR_BBE[metric])
        assert abs(keep - expected) < 0.01, metric


def _pitch_rows(n: int, *, barrel_frac: float, hard_frac: float, xwoba: float) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "launch_speed": np.where(rng.random(n) < hard_frac, 100.0, 85.0),
            "launch_angle": np.full(n, 15.0),
            "launch_speed_angle": np.where(rng.random(n) < barrel_frac, 6, 3),
            "estimated_woba_using_speedangle": np.full(n, xwoba),
            "woba_value": np.full(n, xwoba),
            "events": ["single"] * n,
            "description": ["hit_into_play"] * n,
            "type": ["X"] * n,
            "balls": np.zeros(n, dtype=int),
            "strikes": np.zeros(n, dtype=int),
            "stand": ["R"] * n,
            "pfx_z": np.full(n, 1.2),
            "release_extension": np.full(n, 6.4),
            "release_pos_x": np.full(n, -1.5),
            "release_pos_z": np.full(n, 5.9),
            "release_spin_rate": np.full(n, 2300.0),
        }
    )


def test_shrinkage_moves_contact_rates_and_leaves_command_alone() -> None:
    rows = _pitch_rows(120, barrel_frac=0.20, hard_frac=0.60, xwoba=0.420)
    raw = build_pitcher_regression(rows, shrink=0.0)
    shrunk = build_pitcher_regression(rows, shrink=1.0)

    assert raw.barrel_allowed > shrunk.barrel_allowed > BL_BARREL_ALLOWED
    assert raw.hard_hit_allowed > shrunk.hard_hit_allowed > BL_HARD_HIT
    assert shrunk.xwoba_allowed < raw.xwoba_allowed
    assert abs(shrunk.babip_allowed - BL_BABIP) < abs(raw.babip_allowed - BL_BABIP)
    # Barrel is the least reliable of the group, so it moves the furthest.
    def moved(field: str) -> float:
        return abs(getattr(raw, field) - getattr(shrunk, field)) / abs(
            getattr(raw, field) - {"barrel_allowed": BL_BARREL_ALLOWED,
                                   "hard_hit_allowed": BL_HARD_HIT}[field]
        )
    assert moved("barrel_allowed") > moved("hard_hit_allowed")

    # Command and stuff signals are untouched.
    for field_name in ("csw", "k_pct", "bb_pct", "zone_pct", "chase", "whiff", "swstr"):
        assert getattr(raw, field_name) == getattr(shrunk, field_name)

    # The raw rates survive for reporting.
    assert shrunk.raw_contact["barrel"] == raw.barrel_allowed


def test_shrinkage_damps_the_hit_and_hr_multipliers() -> None:
    rows = _pitch_rows(120, barrel_frac=0.20, hard_frac=0.60, xwoba=0.420)
    raw = build_pitcher_regression(rows, shrink=0.0).allowed_multipliers()
    shrunk = build_pitcher_regression(rows, shrink=1.0).allowed_multipliers()
    assert raw and shrunk
    # A six-week barrel spike stops being priced as a home-run projection.
    assert abs(shrunk["HR"] - 1.0) < abs(raw["HR"] - 1.0)


def test_knob_defaults_off_and_is_env_readable(monkeypatch) -> None:
    assert RollingWindows().starter_contact_shrink == 0.0
    assert Config().windows.starter_contact_shrink == 0.0
    monkeypatch.setenv("MLBE_STARTER_CONTACT_SHRINK", "1.0")
    assert RollingWindows().starter_contact_shrink == 1.0
