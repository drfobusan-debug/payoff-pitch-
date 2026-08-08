"""Bridge innings: the arms between the starter's hook and the 8th.

The pen's leverage profile describes its setup man and closer, so applying it
from the 6th onward priced every bridge inning as if the closer were in it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from mlb_engine.features.pitch_mix import (
    build_arsenal,
    build_batter_pitch_profile,
)
from mlb_engine.features.rolling import LEAGUE_RATES, build_bullpen_profile
from mlb_engine.models.montecarlo import MonteCarlo, TeamSimConfig
from mlb_engine.pipeline import _pen_arsenal_mult

AS_OF = date(2024, 7, 19)
GD = date(2024, 7, 10)


def _relief_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(
        rows, columns=["game_date", "pitcher", "inning", "inning_topbot", "events"]
    )
    df["batter"] = 1
    df["home_team"] = "NYY"
    df["away_team"] = "BOS"
    return df


def _pen(rows):
    return build_bullpen_profile(_relief_frame(rows), "NYY", AS_OF, 21, min_inning=6)


# ---- profile construction --------------------------------------------------
def test_bridge_profile_isolates_the_middle_men() -> None:
    rows = (
        [(GD, 201, 6, "Top", "single") for _ in range(20)]
        + [(GD, 203, 7, "Top", "single") for _ in range(20)]
        + [(GD, 200, 8, "Top", "strikeout") for _ in range(20)]
        + [(GD, 202, 9, "Top", "strikeout") for _ in range(20)]
    )
    pen = _pen(rows)
    # The bridge arms allow the singles the leverage arms do not, and the
    # aggregate sits between the two.
    assert pen.bridge.p_1b > pen.allowed.p_1b > pen.allowed_leverage.p_1b
    assert pen.bridge.p_k < pen.allowed_leverage.p_k


def test_bridge_falls_back_to_the_aggregate_when_thin() -> None:
    rows = (
        # 5 pre-8th relief PAs: below MIN_BRIDGE_PA -> no separate profile
        [(GD, 201, 7, "Top", "single") for _ in range(5)]
        + [(GD, 200, 8, "Top", "strikeout") for _ in range(30)]
    )
    pen = _pen(rows)
    assert pen.allowed_bridge is None
    assert pen.bridge.p_k == pen.allowed.p_k


# ---- simulation ------------------------------------------------------------
def _sim_cfg(**kw) -> TeamSimConfig:
    bat = [dict(LEAGUE_RATES) for _ in range(9)]
    hot = {"1B": 0.5, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0, "OUT": 0.5}
    # Hook the starter after one time through so the pen covers the 6th on.
    return TeamSimConfig(
        bat_vs_starter=bat,
        bat_vs_pen=[dict(hot) for _ in range(9)],
        starter_bf_cap=9,
        starter_pitch_cap=200,
        **kw,
    )


def _cold() -> list[dict[str, float]]:
    cold = {"1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.5, "OUT": 0.5}
    return [dict(cold) for _ in range(9)]


def _hot() -> list[dict[str, float]]:
    hot = {"1B": 0.5, "2B": 0.0, "3B": 0.0, "HR": 0.0, "BB": 0.0, "K": 0.0, "OUT": 0.5}
    return [dict(hot) for _ in range(9)]


def test_bridge_innings_are_not_priced_off_the_closer() -> None:
    legacy = _sim_cfg(bat_vs_pen_close=_cold())
    bridge = _sim_cfg(bat_vs_pen_close=_cold(), bat_vs_pen_bridge=_hot())
    res_legacy = MonteCarlo(600, seed=7).simulate(legacy, legacy)
    res_bridge = MonteCarlo(600, seed=7).simulate(bridge, bridge)
    # Legacy gave the 6th and 7th the shutdown leverage arms; charging them to
    # the (leaky) bridge arms that actually pitch them scores more runs.
    assert res_bridge.home_runs_full.mean() > res_legacy.home_runs_full.mean()


def test_leverage_arms_still_own_the_eighth_and_ninth() -> None:
    # Bridge and aggregate identical, so any gap is the leverage profile's work
    # in the 8th/9th of a close game.
    with_lev = _sim_cfg(bat_vs_pen_close=_cold(), bat_vs_pen_bridge=_hot())
    without = _sim_cfg(bat_vs_pen_bridge=_hot())
    res_with = MonteCarlo(600, seed=11).simulate(with_lev, with_lev)
    res_without = MonteCarlo(600, seed=11).simulate(without, without)
    assert res_with.home_runs_full.mean() < res_without.home_runs_full.mean()


# ---- pen arsenal matchup ---------------------------------------------------
def _pitch_rows(pitch_type: str, description: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitch_type": [pitch_type] * n,
            "description": [description] * n,
            "estimated_woba_using_speedangle": [None] * n,
        }
    )


def test_pen_arsenal_matchup_prices_a_whiff_prone_hitter() -> None:
    # A pen that lives on sliders against a hitter who cannot touch one.
    pen = build_arsenal(_pitch_rows("SL", "swinging_strike", 60))
    hitter = build_batter_pitch_profile(_pitch_rows("SL", "swinging_strike", 40))
    mult = _pen_arsenal_mult(pen, hitter)
    assert mult["K"] > 1.0


def test_pen_arsenal_is_neutral_without_a_readable_mix() -> None:
    hitter = build_batter_pitch_profile(_pitch_rows("SL", "swinging_strike", 40))
    assert _pen_arsenal_mult(None, hitter) == {}
    # A pen with too few tracked pitches per class reads as no arsenal at all.
    thin = build_arsenal(_pitch_rows("SL", "swinging_strike", 5))
    assert _pen_arsenal_mult(thin, hitter) == {}
