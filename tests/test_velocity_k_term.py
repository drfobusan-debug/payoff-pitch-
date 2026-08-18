"""The four-seam velocity term on a starter's strikeout rate.

Two reads: how hard he throws against the league, and how his most recent start
sat against his own window. Both are off unless ``vfa_k`` is set, because the
term ships quoted before it is bought.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from mlb_engine.config import Config
from mlb_engine.features.regression import (
    BL_VFA,
    MIN_VFA_START,
    PitcherRegression,
    build_pitcher_regression,
)


def _rows(days: dict[str, tuple[float, int]]) -> pd.DataFrame:
    """One frame of four-seamers: ``{game_date: (velocity, pitches)}``."""
    out = []
    for day, (velo, n) in days.items():
        for i in range(n):
            out.append(
                {
                    "game_date": day,
                    "pitch_type": "FF",
                    "release_speed": velo,
                    "description": "foul",
                    "events": "field_out" if i % 5 == 0 else None,
                    "woba_denom": 1.0 if i % 5 == 0 else None,
                    "woba_value": 0.0,
                    "bb_type": None,
                    "launch_speed": None,
                    "pfx_z": None,
                    "release_extension": None,
                    "release_pos_x": None,
                    "release_pos_z": None,
                    "release_spin_rate": None,
                }
            )
    return pd.DataFrame(out)


def _reg(days: dict[str, tuple[float, int]], vfa_k: float) -> PitcherRegression:
    return build_pitcher_regression(_rows(days), vfa_k=vfa_k)


def test_the_velocity_term_is_off_until_it_is_switched_on() -> None:
    days = {"2026-07-20": (98.0, 200), "2026-08-01": (98.0, 90)}
    assert Config().windows.vfa_k_weight == 0.0
    assert _reg(days, 0.0).velocity_k_multiplier() == 1.0
    assert _reg(days, 1.0).velocity_k_multiplier() > 1.0


def test_harder_is_more_strikeouts_and_softer_is_fewer() -> None:
    hard = _reg({"2026-07-20": (98.0, 200), "2026-08-01": (98.0, 90)}, 1.0)
    soft = _reg({"2026-07-20": (90.0, 200), "2026-08-01": (90.0, 90)}, 1.0)
    league = _reg({"2026-07-20": (BL_VFA, 200), "2026-08-01": (BL_VFA, 90)}, 1.0)
    assert hard.velocity_k_multiplier() > 1.0 > soft.velocity_k_multiplier()
    assert abs(league.velocity_k_multiplier() - 1.0) < 1e-9
    # And the level is clipped, so no arm gets an unbounded strikeout bonus.
    absurd = _reg({"2026-07-20": (108.0, 200), "2026-08-01": (108.0, 90)}, 1.0)
    assert absurd.velocity_k_multiplier() < 1.13


def test_the_last_start_moves_the_read_on_its_own() -> None:
    """Same window, one start down a tick: the arm is priced for fewer Ks."""
    steady = _reg({"2026-07-20": (95.0, 200), "2026-08-01": (95.0, 90)}, 1.0)
    dipped = _reg({"2026-07-20": (95.0, 200), "2026-08-01": (92.0, 90)}, 1.0)
    spiked = _reg({"2026-07-20": (95.0, 200), "2026-08-01": (98.0, 90)}, 1.0)
    assert spiked.velocity_k_multiplier() > steady.velocity_k_multiplier()
    assert dipped.velocity_k_multiplier() < steady.velocity_k_multiplier()
    # The deviation is measured against the window his own start is part of.
    assert dipped.vfa_dev < 0 < spiked.vfa_dev


def test_a_start_he_barely_threw_the_fastball_in_is_not_a_reading() -> None:
    thin = _reg({"2026-07-20": (95.0, 200), "2026-08-01": (88.0, MIN_VFA_START - 1)}, 1.0)
    assert thin.vfa_dev != thin.vfa_dev  # NaN
    assert thin.vfa == thin.vfa
    # The level still prices; only the one-start arrow is withheld.
    assert thin.velocity_k_multiplier() < 1.0


def test_a_thin_window_prices_nothing() -> None:
    reg = _reg({"2026-08-01": (99.0, 30)}, 1.0)
    assert reg.vfa != reg.vfa  # NaN
    assert reg.velocity_k_multiplier() == 1.0


def test_the_term_was_fitted_on_starters_so_it_stays_off_bullpens() -> None:
    pen = build_pitcher_regression(
        _rows({"2026-07-20": (98.0, 200), "2026-08-01": (98.0, 90)}),
        bullpen=True,
        vfa_k=1.0,
    )
    assert pen.velocity_k_multiplier() == 1.0


def test_the_weight_scales_the_whole_correction() -> None:
    days = {"2026-07-20": (97.0, 200), "2026-08-01": (99.0, 90)}
    full = _reg(days, 1.0).velocity_k_multiplier() - 1.0
    half = _reg(days, 0.5).velocity_k_multiplier() - 1.0
    assert 0 < half < full
    assert abs(half / full - 0.5) < 0.02


def test_the_slate_article_reads_velocity_off_the_last_start() -> None:
    from mlb_engine.features.trend import pitcher_trends

    rows = _rows({"2026-07-10": (95.0, 200), "2026-07-20": (95.0, 200), "2026-08-01": (93.0, 90)})
    trends = pitcher_trends(rows, dt.date(2026, 8, 5), 42)
    assert trends.vfa.recent == 93.0
    assert trends.vfa.delta is not None and trends.vfa.delta < -0.3
