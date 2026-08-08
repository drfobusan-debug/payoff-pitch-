"""Foul balls carry an exit velocity but are not balls in play."""

from __future__ import annotations

import pandas as pd

from mlb_engine.data.statcast import batted_balls
from mlb_engine.features.regression import build_batter_regression


def _pitch(desc: str, ev: float, la: float = 20.0, **kw: object) -> dict[str, object]:
    row: dict[str, object] = {
        "batter": 1,
        "pitcher": 2,
        "description": desc,
        "type": "X" if desc == "hit_into_play" else "S",
        "events": "single" if desc == "hit_into_play" else None,
        "launch_speed": ev,
        "launch_angle": la,
        "launch_speed_angle": 4 if desc == "hit_into_play" else None,
        "bb_type": "line_drive" if desc == "hit_into_play" else None,
        "estimated_ba_using_speedangle": 0.35,
        "estimated_woba_using_speedangle": 0.40,
        "woba_value": 0.9,
        "zone": 5,
        "bat_speed": 72.0,
    }
    row.update(kw)
    return row


def _slice(n_bip: int, n_foul: int, *, bip_ev: float = 100.0, foul_ev: float = 70.0):
    rows = [_pitch("hit_into_play", bip_ev) for _ in range(n_bip)]
    rows += [_pitch("foul", foul_ev) for _ in range(n_foul)]
    return pd.DataFrame(rows)


def test_fouls_are_excluded_from_the_batted_ball_pool() -> None:
    df = _slice(10, 10)
    assert len(batted_balls(df)) == 10
    assert set(batted_balls(df)["description"]) == {"hit_into_play"}


def test_hard_hit_is_not_diluted_by_weak_foul_contact() -> None:
    """Every ball in play here is 100 mph, so hard-hit% is 1.0 whatever the fouls do."""
    clean = build_batter_regression(_slice(20, 0))
    fouled = build_batter_regression(_slice(20, 20))
    assert clean.hard_hit == 1.0
    assert fouled.hard_hit == 1.0
    assert fouled.bbe == 20  # not 40


def test_the_batted_ball_count_drives_barrel_per_pa() -> None:
    """Counting fouls as batted balls inflates barrels per plate appearance."""
    df = _slice(20, 20)
    reg = build_batter_regression(df)
    assert reg.bbe == 20
    assert reg.pa == 20  # only the balls in play carry an event here
    assert reg.barrel_per_pa == reg.barrel_rate  # bbe == pa


def test_air_contact_reads_only_balls_in_play() -> None:
    """A batter who fouls off weak pitches still gets credit for his real contact."""
    rows = [_pitch("hit_into_play", 100.0, la=20.0) for _ in range(15)]
    rows += [_pitch("foul", 60.0, la=25.0) for _ in range(30)]
    reg = build_batter_regression(pd.DataFrame(rows))
    assert reg.fb_ld_ev == 100.0
    assert reg.fb_ld_hard_hit == 1.0


def test_type_column_is_preferred_and_description_is_the_fallback() -> None:
    df = _slice(5, 5)
    assert len(batted_balls(df.drop(columns=["description"]))) == 5
    assert len(batted_balls(df.drop(columns=["type"]))) == 5
    # Neither column: fall back to exit velocity so reduced frames still work.
    bare = df.drop(columns=["type", "description"])
    assert len(batted_balls(bare)) == 10


def test_soft_air_contact_brake_no_longer_fires_on_league_average_contact() -> None:
    """League median air EV is ~89.4 mph; the floor must sit below it."""
    from mlb_engine.features.regression import FB_LD_EV_FLOOR

    assert FB_LD_EV_FLOOR < 89.4
    median_bat = build_batter_regression(_slice(30, 0, bip_ev=89.4))
    assert median_bat.fb_ld_ev > FB_LD_EV_FLOOR  # brake does not fire
    soft = build_batter_regression(_slice(30, 0, bip_ev=80.0))
    assert soft.fb_ld_ev < FB_LD_EV_FLOOR
    assert soft.multipliers()["HR"] < median_bat.multipliers()["HR"]
