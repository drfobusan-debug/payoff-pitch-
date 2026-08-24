"""The delivery behind the luck term: what it reads, and what it refuses to guess.

The arm model's failure modes are all silent ones -- a left-hander's release
point read with the wrong sign, a metric averaged over a sample too thin to mean
anything, a column our own ingestion used to drop coming back as league average
instead of as unknown. Each of those is pinned here.
"""

from __future__ import annotations

import math
from datetime import date as Date
from datetime import timedelta

import pandas as pd

from mlb_engine.features.arm import (
    CONFIRMED,
    CONTRADICTED,
    DRIFT_SD,
    HOLDING,
    LEAGUE,
    MIN_LEVEL_PITCHES,
    PITCHES_FOR_READABLE,
    SHEDDING,
    UNMEASURED,
    WINDOW,
    ArmProfile,
    build_arm_profile,
    fastballs_of,
    reliability,
    stage_two,
    velo_trend,
)

DAY0 = Date(2026, 5, 1)


def _pitches(
    n: int,
    *,
    pitch_type: str = "FF",
    throws: str = "R",
    velo: float = 96.0,
    ext: float = 6.6,
    rel_x: float = -1.9,  # Statcast's sign for a right-hander, from the catcher's view
    rel_z: float = 5.9,
    spin: float = 2300.0,
    pfx_z: float = 1.25,
    pfx_x: float = -0.85,
    day0: int = 0,
) -> pd.DataFrame:
    """``n`` identical tracked fastballs, one per day so the order is unambiguous."""
    return pd.DataFrame(
        {
            "pitcher": [10] * n,
            "pitch_type": [pitch_type] * n,
            "p_throws": [throws] * n,
            "game_date": [DAY0 + timedelta(days=day0 + i) for i in range(n)],
            "release_speed": [velo] * n,
            "release_extension": [ext] * n,
            "release_pos_x": [rel_x] * n,
            "release_pos_z": [rel_z] * n,
            "release_spin_rate": [spin] * n,
            "pfx_z": [pfx_z] * n,
            "pfx_x": [pfx_x] * n,
        }
    )


# --- reliability ----------------------------------------------------------


def test_every_arm_measure_half_repeats_inside_one_pitch() -> None:
    """The finding the window rests on, and the reason it is not a reliability window.

    A radar reading is a physical property measured directly rather than a rate
    estimated from outcomes, so the r=.50 rule that sized the hitter windows
    would size these at a single pitch. If this ever stops being true the window
    below has to be rederived rather than kept.
    """
    assert set(PITCHES_FOR_READABLE) == set(LEAGUE) - {"scatter"}
    assert all(n <= 1.0 for n in PITCHES_FOR_READABLE.values())
    assert WINDOW > max(PITCHES_FOR_READABLE.values())


def test_reliability_interpolates_and_holds_at_the_measured_ends() -> None:
    assert reliability("pvelo", 1) == 0.913
    assert reliability("pvelo", 0.2) == 0.913  # below the grid, no extrapolation
    assert reliability("pvelo", 10_000) == 0.974
    mid = reliability("ivb", 30)
    assert 0.944 < mid < 0.979
    assert math.isclose(reliability("no_such_metric", 50), 1.0)


# --- the levels themselves ------------------------------------------------


def test_perceived_velocity_is_release_speed_plus_extension() -> None:
    """The user's form, and the level the verdict is read on."""
    prof = build_arm_profile(_pitches(60, velo=97.0, ext=7.0))
    assert math.isclose(prof.velo, 97.0)
    assert math.isclose(prof.pvelo, 97.0 + 1.1 * 7.0 - 6.0)
    assert prof.pvelo > prof.velo  # a long stride buys the arm mph it does not throw


def test_only_the_last_window_fastballs_are_read() -> None:
    """A level is a level *now*: an arm that lost 4 mph reads at the new velocity."""
    old = _pitches(WINDOW * 2, velo=99.0)
    new = _pitches(WINDOW, velo=95.0, day0=WINDOW * 2 + 5)
    prof = build_arm_profile(pd.concat([old, new], ignore_index=True))
    assert prof.pitches == WINDOW * 3
    assert math.isclose(prof.velo, 95.0)


def test_the_slice_is_ordered_before_it_is_windowed() -> None:
    """Rows arriving newest-first must not read the oldest fastballs as current."""
    old = _pitches(WINDOW, velo=99.0)
    new = _pitches(WINDOW, velo=95.0, day0=WINDOW + 5)
    shuffled = pd.concat([new, old], ignore_index=True)
    assert math.isclose(build_arm_profile(shuffled).velo, 95.0)


def test_only_fastballs_count_toward_the_delivery() -> None:
    """A curveball's velocity and break are not comparable with a fastball's."""
    rows = pd.concat(
        [_pitches(40, velo=96.0), _pitches(40, pitch_type="CU", velo=79.0, day0=40)],
        ignore_index=True,
    )
    assert len(fastballs_of(rows)) == 40
    assert math.isclose(build_arm_profile(rows).velo, 96.0)


def test_a_left_handers_release_point_and_break_are_mirrored() -> None:
    """Arm side positive for either hand, or a lefty looks like a submariner.

    Statcast measures ``release_pos_x`` from the catcher's view, so the two hands
    sit on opposite sides of zero and pooling them raw would put the league mean
    near the plate's centre line and make every arm an outlier.
    """
    righty = build_arm_profile(_pitches(60, throws="R", rel_x=-1.9, pfx_x=-0.85))
    lefty = build_arm_profile(_pitches(60, throws="L", rel_x=1.9, pfx_x=0.85))
    assert math.isclose(righty.rel_x, lefty.rel_x)
    assert math.isclose(righty.hb, lefty.hb)
    assert righty.rel_x > 0 and righty.hb > 0


def test_break_is_reported_in_inches() -> None:
    """``pfx_z`` arrives in feet; a 15-inch ride must not print as 1.25."""
    prof = build_arm_profile(_pitches(60, pfx_z=1.25))
    assert math.isclose(prof.ivb, 15.0)


# --- missingness ----------------------------------------------------------


def test_a_thin_arm_is_unmeasured_rather_than_league_average() -> None:
    prof = build_arm_profile(_pitches(MIN_LEVEL_PITCHES - 1))
    assert prof.pitches == MIN_LEVEL_PITCHES - 1
    assert all(v != v for v in prof.levels().values())
    assert stage_two(0.02, prof) == UNMEASURED


def test_an_empty_slice_reads_as_no_pitches_and_no_levels() -> None:
    prof = build_arm_profile(pd.DataFrame())
    assert prof.pitches == 0
    assert math.isnan(prof.stuff_z)
    assert stage_two(0.02, prof) == UNMEASURED
    assert stage_two(0.02, None) == UNMEASURED


def test_a_cache_predating_the_horizontal_break_column_still_reads() -> None:
    """``pfx_x`` was in the feed and our ingestion dropped it until now.

    Every frame cached before that carries no column at all, and the model has to
    read the rest of the delivery off it rather than raising or inventing a break.
    """
    old = _pitches(60).drop(columns=["pfx_x"])
    prof = build_arm_profile(old)
    assert math.isnan(prof.hb)
    assert not math.isnan(prof.pvelo)
    assert stage_two(0.02, prof) != UNMEASURED


def test_an_untracked_arm_angle_column_is_simply_not_required() -> None:
    """Arm angle earns nothing out of time, so its absence cannot matter."""
    prof = build_arm_profile(_pitches(60))
    assert "arm_angle" not in prof.levels()
    assert not math.isnan(prof.stuff_z)


def test_null_readings_are_dropped_rather_than_averaged_in() -> None:
    rows = _pitches(60)
    rows.loc[rows.index[:20], "release_speed"] = None
    rows.loc[rows.index[:59], "release_spin_rate"] = None
    prof = build_arm_profile(rows)
    assert math.isclose(prof.velo, 96.0)  # 40 valid readings, none of them zero
    assert math.isnan(prof.spin)  # one valid reading is not a level


def test_an_untracked_extension_costs_the_perceived_velocity_and_nothing_else() -> None:
    """Perceived velocity needs both halves; the radar reading survives alone."""
    prof = build_arm_profile(_pitches(60).drop(columns=["release_extension"]))
    assert math.isnan(prof.pvelo) and math.isnan(prof.ext)
    assert math.isclose(prof.velo, 96.0)
    assert stage_two(0.02, prof) == UNMEASURED  # the verdict rides on pvelo


def test_a_slice_without_handedness_reads_the_right_hand() -> None:
    prof = build_arm_profile(_pitches(60).drop(columns=["p_throws"]))
    assert prof.rel_x < 0 or prof.rel_x > 0  # a number, not a crash
    assert not math.isnan(prof.pvelo)


# --- the verdict ----------------------------------------------------------


def _profile(pvelo: float) -> ArmProfile:
    return ArmProfile(pitches=WINDOW, pvelo=pvelo)


def test_the_verdict_crosses_the_luck_term_with_the_arm() -> None:
    """Hot results and a good arm disagree; cold results and a good arm agree."""
    mu = LEAGUE["pvelo"][0]
    good, weak = _profile(mu + 3.0), _profile(mu - 3.0)
    assert stage_two(+0.030, weak) == CONFIRMED  # due to correct, nothing holding it up
    assert stage_two(+0.030, good) == CONTRADICTED  # lucky and good at once
    assert stage_two(-0.030, good) == CONFIRMED  # the rebound has an arm behind it
    assert stage_two(-0.030, weak) == CONTRADICTED  # the ugly line is the level


def test_ride_is_reported_beside_the_verdict_and_never_folded_into_it() -> None:
    """Ride points at home runs and against hits, so one number cannot carry both."""
    mu, sd = LEAGUE["ivb"]
    flat = ArmProfile(pitches=WINDOW, pvelo=LEAGUE["pvelo"][0] + 3.0, ivb=mu - 2 * sd)
    rides = ArmProfile(pitches=WINDOW, pvelo=LEAGUE["pvelo"][0] + 3.0, ivb=mu + 2 * sd)
    assert stage_two(0.03, flat) == stage_two(0.03, rides) == CONTRADICTED
    assert flat.ride_z < 0 < rides.ride_z
    assert math.isclose(flat.stuff_z, rides.stuff_z)


def test_perceived_velocity_is_the_only_measure_with_a_trend() -> None:
    """Every other delta earned nothing out of time, so none of them is readable.

    Guards against a well-meaning revival of the spin, break, release-point and
    scatter deltas (t -0.4 to +1.5 on the next fortnight's wOBA), which the panel
    refused while it kept the velocity one.
    """
    prof = build_arm_profile(_pitches(2 * MIN_LEVEL_PITCHES))
    assert not math.isnan(prof.d_pvelo)
    assert not any("d_" in f or "trend" in f for f in prof.levels())
    assert [f for f in vars(prof) if f.startswith("d_")] == ["d_pvelo"]


def test_the_trend_is_the_last_block_against_the_one_before_it() -> None:
    """An arm that has shed a mile reads as shedding exactly that mile."""
    was = _pitches(WINDOW, velo=96.0)
    now = _pitches(WINDOW, velo=95.0, day0=WINDOW)
    prof = build_arm_profile(pd.concat([was, now]))
    assert math.isclose(prof.d_pvelo, -1.0, abs_tol=1e-9)
    assert velo_trend(prof) == SHEDDING
    assert math.isclose(prof.trend_z, -1.0 / DRIFT_SD)


def test_a_level_can_be_readable_while_the_trend_is_not() -> None:
    """The trend needs two whole blocks; half of one is a different quantity."""
    prof = build_arm_profile(_pitches(2 * MIN_LEVEL_PITCHES - 1))
    assert not math.isnan(prof.pvelo)
    assert math.isnan(prof.d_pvelo) and math.isnan(prof.trend_z)
    assert velo_trend(prof) == UNMEASURED
    assert velo_trend(None) == UNMEASURED
    assert stage_two(0.03, prof) != UNMEASURED  # the verdict never needed the trend


def test_the_trend_does_not_move_the_verdict() -> None:
    """It graded on the fade side only, so it qualifies a verdict rather than casting one."""
    mu = LEAGUE["pvelo"][0]
    shedding = ArmProfile(pitches=WINDOW, pvelo=mu + 3.0, d_pvelo=-1.5)
    holding = ArmProfile(pitches=WINDOW, pvelo=mu + 3.0, d_pvelo=+1.5)
    assert velo_trend(shedding) == SHEDDING
    assert velo_trend(holding) == HOLDING
    assert stage_two(0.03, shedding) == stage_two(0.03, holding) == CONTRADICTED
    assert stage_two(-0.03, shedding) == stage_two(-0.03, holding) == CONFIRMED
    assert velo_trend(ArmProfile(pitches=WINDOW, pvelo=mu, d_pvelo=0.0)) == HOLDING
